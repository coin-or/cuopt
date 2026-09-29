/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <branch_and_bound/branch_and_bound.hpp>
#include <chrono>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/user_problem.hpp>
#include <linear_algebra/sparse_matrix.hpp>
#include <math_optimization/tic_toc.hpp>
#include <mip_heuristics/feasibility_jump/cpu/climber.hpp>
#include <stdexcept>
#include <utilities/timer.hpp>
#include "interface.hpp"

namespace cuopt::hive_lns {
namespace mip     = cuopt::mathematical_optimization::mip;
namespace simplex = cuopt::mathematical_optimization::simplex;
enum class repair_backend_t { cpufj, submip };

// Eliminate fixed columns into row bounds. All state is local to this request.
struct repair_problem_t {
  model_t reduced;
  std::vector<int> free_columns;
  std::vector<double> full, start;
  bool possible = true;
};
inline repair_problem_t make_repair_problem(const model_t& model, const repair_request_t& request)
{
  const size_t n = model.lower.size();
  if (request.start.size() != n || request.lower.size() != n || request.upper.size() != n)
    throw std::invalid_argument("Repair vectors must cover every model variable");
  repair_problem_t p;
  p.full = request.start;
  std::vector<int> index(n, -1);
  for (size_t j = 0; j < n; ++j) {
    double lo = request.lower[j], hi = request.upper[j];
    if (!std::isfinite(p.full[j]) || std::isnan(lo) || std::isnan(hi) || lo < model.lower[j] ||
        hi > model.upper[j]) {
      // Reject the neighborhood without widening the original model's domains.
      p.possible = false;
      return p;
    }
    if (model.integer[j]) {
      lo = std::ceil(lo);
      hi = std::floor(hi);
    }
    if (lo > hi || lo == std::numeric_limits<double>::infinity() ||
        hi == -std::numeric_limits<double>::infinity()) {
      p.possible = false;
      return p;
    }
    if (lo == hi) {
      p.full[j] = lo;
      continue;
    }
    index[j] = p.free_columns.size();
    p.free_columns.push_back(j);
    p.reduced.lower.push_back(lo);
    p.reduced.upper.push_back(hi);
    p.reduced.integer.push_back(model.integer[j]);
    p.reduced.objective.push_back(model.objective[j]);
    double value = std::clamp(p.full[j], lo, hi);
    if (model.integer[j]) value = std::clamp(std::round(value), lo, hi);
    p.start.push_back(value);
  }
  p.reduced.offsets.push_back(0);
  p.reduced.feasibility_tolerance = model.feasibility_tolerance;
  p.reduced.relative_tolerance    = model.relative_tolerance;
  p.reduced.integrality_tolerance = model.integrality_tolerance;
  for (size_t r = 0; r < model.row_lower.size(); ++r) {
    double fixed = 0, correction = 0;
    const size_t begin = p.reduced.columns.size();
    for (int k = model.offsets[r]; k < model.offsets[r + 1]; ++k) {
      const int j = model.columns[k];
      if (index[j] >= 0) {
        p.reduced.columns.push_back(index[j]);
        p.reduced.coefficients.push_back(model.coefficients[k]);
      } else {
        const double term = model.coefficients[k] * p.full[j] - correction;
        const double next = fixed + term;
        correction        = (next - fixed) - term;
        fixed             = next;
      }
    }
    const double lo = model.row_lower[r] - fixed, hi = model.row_upper[r] - fixed;
    if (!std::isfinite(fixed)) {
      p.possible = false;
      return p;
    }
    if (begin == p.reduced.columns.size()) {
      if (lo > model.row_tolerances[r] || hi < -model.row_tolerances[r]) {
        p.possible = false;
        return p;
      }
      continue;
    }
    p.reduced.row_lower.push_back(lo);
    p.reduced.row_upper.push_back(hi);
    p.reduced.row_tolerances.push_back(model.row_tolerances[r]);
    p.reduced.offsets.push_back(p.reduced.columns.size());
  }
  return p;
}

inline repair_result_t repair_neighborhood(const model_t& model,
                                           const repair_request_t& request,
                                           repair_backend_t backend,
                                           const stop_fn& stop,
                                           const raft::handle_t* handle,
                                           double remaining_seconds,
                                           std::atomic<bool>& preemption)
{
  if (!std::isfinite(request.time_limit_seconds) || request.time_limit_seconds < 0 ||
      request.max_iterations <= 0 || request.max_nodes <= 0)
    throw std::invalid_argument("Invalid repair time/work budget");
  const double budget = std::min(request.time_limit_seconds, remaining_seconds);
  const cuopt::timer_t timer(std::max(0.0, budget));
  repair_result_t result;
  if (budget <= 0 || stop()) return result;
  auto p        = make_repair_problem(model, request);
  auto consider = [&](const std::vector<double>& reduced) {
    if (stop() || timer.check_time_limit() || reduced.size() != p.free_columns.size()) return;
    auto full = p.full;
    for (size_t j = 0; j < reduced.size(); ++j)
      full[p.free_columns[j]] = reduced[j];
    for (size_t j = 0; j < full.size(); ++j)
      if (full[j] < request.lower[j] - model.integrality_tolerance ||
          full[j] > request.upper[j] + model.integrality_tolerance)
        return;
    if (!model.feasible(full)) return;
    const double objective = model.cost(full);
    if (stop() || timer.check_time_limit()) return;
    if (!result.feasible || objective < result.objective) {
      result.feasible   = true;
      result.assignment = std::move(full);
      result.objective  = objective;
    }
  };
  if (!p.possible || stop() || timer.check_time_limit()) {
    result.elapsed_seconds = timer.elapsed_time();
    return result;
  }
  if (p.free_columns.empty()) {
    consider({});
    result.elapsed_seconds = timer.elapsed_time();
    return result;
  }
  auto& m = p.reduced;
  // Search-free neighborhoods need no engine and avoid empty-matrix corner cases.
  if (m.row_lower.empty()) {
    auto x = p.start;
    for (size_t j = 0; j < x.size(); ++j) {
      double value = m.objective[j] >= 0 ? m.lower[j] : m.upper[j];
      if (std::isfinite(value)) x[j] = value;
    }
    consider(x);
    result.elapsed_seconds = timer.elapsed_time();
    return result;
  }
  omp_set_num_threads(1);  // This worker's OpenMP ICV, never the main solver's team.
  if (backend == repair_backend_t::cpufj) {
    std::vector<cuopt::mathematical_optimization::var_t> types;
    for (bool integer : m.integer)
      types.push_back(integer ? cuopt::mathematical_optimization::var_t::INTEGER
                              : cuopt::mathematical_optimization::var_t::CONTINUOUS);
    cuopt::mathematical_optimization::mip_solver_settings_t<int, double>::tolerances_t tolerances;
    tolerances.absolute_tolerance    = model.feasibility_tolerance;
    tolerances.relative_tolerance    = model.relative_tolerance;
    tolerances.integrality_tolerance = model.integrality_tolerance;
    mip::fj_settings_t settings;
    settings.seed            = request.seed;
    settings.iteration_limit = request.max_iterations;
    auto climber             = mip::init_fj_cpu_from_host_model<int, double>(m.lower.size(),
                                                                 m.row_lower.size(),
                                                                 m.coefficients.size(),
                                                                 false,
                                                                 1.0,
                                                                 0.0,
                                                                 m.coefficients,
                                                                 m.columns,
                                                                 m.offsets,
                                                                 m.objective,
                                                                 m.lower,
                                                                 m.upper,
                                                                 m.row_lower,
                                                                 m.row_upper,
                                                                             {},
                                                                             {},
                                                                 std::move(types),
                                                                 tolerances,
                                                                 preemption,
                                                                 settings);
    // The factory initializes its own seed; select the caller's seed before search.
    climber->settings.seed          = request.seed;
    climber->h_assignment           = p.start;
    climber->h_best_assignment      = p.start;
    climber->suppress_incumbent_log = true;
    climber->improvement_callback   = [&](double, const std::vector<double>& x, double) {
      consider(x);
    };
    if (!stop() && !timer.check_time_limit())
      mip::cpufj_solve(climber.get(), timer.remaining_time());
  } else {
    simplex::user_problem_t<int, double> problem(handle);
    problem.num_cols     = m.lower.size();
    problem.num_rows     = m.row_lower.size();
    problem.objective    = m.objective;
    problem.lower        = m.lower;
    problem.upper        = m.upper;
    problem.problem_name = "LNS-private-repair";
    cuopt::mathematical_optimization::csr_matrix_t<int, double> csr(
      problem.num_rows, problem.num_cols, m.coefficients.size());
    csr.x         = m.coefficients;
    csr.j         = m.columns;
    csr.row_start = m.offsets;
    csr.to_compressed_col(problem.A);
    for (int r = 0; r < problem.num_rows; ++r) {
      const double lo = m.row_lower[r], hi = m.row_upper[r];
      if (lo == hi) {
        problem.row_sense.push_back('E');
        problem.rhs.push_back(lo);
      } else if (!std::isfinite(hi)) {
        problem.row_sense.push_back('G');
        problem.rhs.push_back(lo);
      } else if (!std::isfinite(lo)) {
        problem.row_sense.push_back('L');
        problem.rhs.push_back(hi);
      } else {
        problem.row_sense.push_back('E');
        problem.rhs.push_back(lo);
        problem.range_rows.push_back(r);
        problem.range_value.push_back(hi - lo);
      }
    }
    problem.num_range_rows = problem.range_rows.size();
    for (bool integer : m.integer)
      problem.var_types.push_back(integer ? simplex::variable_type_t::INTEGER
                                          : simplex::variable_type_t::CONTINUOUS);
    simplex::simplex_solver_settings_t<int, double> settings;
    settings.time_limit                               = timer.remaining_time();
    settings.num_threads                              = 1;
    settings.node_limit                               = request.max_nodes;
    settings.branch_and_bound_simplex_iteration_limit = request.max_iterations;
    settings.random_seed                              = request.seed;
    settings.print_presolve_stats                     = false;
    settings.log.log                                  = false;
    settings.primal_tol                               = model.feasibility_tolerance;
    settings.integer_tol                              = model.integrality_tolerance;
    settings.absolute_mip_gap_tol                     = 0;
    settings.relative_mip_gap_tol                     = 0;
    settings.reliability_branching                    = 0;
    settings.max_cut_passes                           = 0;
    settings.clique_cuts                              = 0;
    settings.zero_half_cuts                           = 0;
    settings.inside_submip                            = 1;
    settings.num_gpus                                 = 0;
    settings.submip_settings.rins                     = 0;
    settings.submip_settings.rens                     = 0;
    settings.submip_settings.enable_cpufj             = false;
    settings.strong_branching_simplex_iteration_limit = 200;
    settings.solution_callback = [&](std::vector<double>& x, double) { consider(x); };
    if (!stop() && !timer.check_time_limit()) {
      mip::probing_implied_bound_t<int, double> probing(problem.num_cols);
      mip::branch_and_bound_t<int, double> solver(
        problem, settings, cuopt::mathematical_optimization::tic(), probing);
      solver.set_initial_guess(p.start);
      solver.set_submip_halt_callback(
        [&](double, double) { return stop() || timer.check_time_limit(); });
      simplex::mip_solution_t<int, double> solution(problem.num_cols);
      solver.solve(solution);
    }
  }
  result.elapsed_seconds = timer.elapsed_time();
  return result;
}
}  // namespace cuopt::hive_lns
