/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "repair_lns.cuh"

#include <branch_and_bound/branch_and_bound.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/user_problem.hpp>
#include <linear_algebra/sparse_matrix.hpp>
#include <mip_heuristics/feasibility_jump/cpu/climber.hpp>
#include <mip_heuristics/lns/cpufj_validation.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/utils.hpp>

#include <raft/core/handle.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <utility>

namespace cuopt::mathematical_optimization::mip {

namespace {
enum class repair_operator_t { CPUFJ, SUBMIP, BP };
}  // namespace

template <typename i_t, typename f_t>
struct repair_lns_t<i_t, f_t>::bandit_arm_t {
  int pulls     = 0;
  double reward = 0.0;
  double ucb() const
  {
    if (pulls == 0) return 1e6;
    const double exploitation = reward / std::max(1, pulls);
    const double exploration =
      std::sqrt(2.0 * std::log((double)std::max(1, pulls)) / std::max(1, pulls));
    return exploitation + exploration;
  }
};

template <typename i_t, typename f_t>
struct repair_lns_t<i_t, f_t>::bound_change_t {
  i_t variable;
  f_t old_lo;
  f_t old_hi;
};

// References preserve the search state across repair backends within one iteration.
template <typename i_t, typename f_t>
struct repair_lns_t<i_t, f_t>::search_state_t {
  std::vector<f_t>& current;
  f_t& current_cost;
  std::vector<f_t>& best_known;
  f_t& best_known_cost;
  bandit_arm_t* bandit_arms;
  std::mt19937_64& rng;
  raft::handle_t& repair_handle;
  const submit_fn& submit;
  bool& improvement_found;
};

template <typename i_t, typename f_t>
struct repair_lns_t<i_t, f_t>::backend_config_t {
  lns_repair_backend_t backend;
  int arm;
  f_t time_limit;
  int max_iterations;
  int max_nodes;
  double new_best_reward;
  double improve_reward;
  double penalty;
};

// Each branch-and-propagate episode reuses its bounds and trail throughout recursion.
template <typename i_t, typename f_t>
struct repair_lns_t<i_t, f_t>::branch_state_t {
  const std::vector<i_t>& rows;
  const std::vector<bool>& chosen;
  const std::vector<i_t>& ruined;
  int max_propagate_passes;
  int node_budget;
  f_t fixed_objective;
  std::vector<f_t>& lo;
  std::vector<f_t>& hi;
  std::vector<bound_change_t>& trail;
  int& nodes;
  const cuopt::timer_t& search_timer;
};

template <typename i_t, typename f_t>
repair_lns_t<i_t, f_t>::repair_lns_t(const fj_cpu_climber_t<i_t, f_t>& anchor,
                                     std::atomic<bool>& preemption,
                                     uint64_t seed,
                                     cuopt::timer_t timer)
  : repair_lns_t(anchor,
                 anchor.h_var_bounds.underlying(),
                 anchor.problem->h_var_types,
                 preemption,
                 seed,
                 timer)
{
}

template <typename i_t, typename f_t>
repair_lns_t<i_t, f_t>::repair_lns_t(const fj_cpu_climber_t<i_t, f_t>& anchor,
                                     std::vector<typename type_2<f_t>::type> bounds,
                                     std::vector<var_t> types,
                                     std::atomic<bool>& preemption,
                                     uint64_t seed,
                                     cuopt::timer_t timer)
  : problem_(anchor.problem),
    bounds_(std::move(bounds)),
    types_(std::move(types)),
    preemption_(preemption),
    seed_(seed),
    timer_(timer)
{
  const auto& p = *problem_;
  cuopt_assert(bounds_.size() == (size_t)p.n_variables && types_.size() == bounds_.size(),
               "repair domains must cover every variable");
  row_tolerances_.reserve(p.n_constraints);
  for (i_t r = 0; r < p.n_constraints; ++r) {
    row_tolerances_.push_back(get_cstr_tolerance<i_t, f_t>(p.cstr_lb[r],
                                                           p.cstr_ub[r],
                                                           p.tolerances.absolute_tolerance,
                                                           p.tolerances.relative_tolerance));
  }
}

template <typename i_t, typename f_t>
bool repair_lns_t<i_t, f_t>::feasible(const std::vector<f_t>& x) const
{
  return verify_cpufj_lns_feasible(*problem_, bounds_, types_, x);
}

template <typename i_t, typename f_t>
f_t repair_lns_t<i_t, f_t>::cost(const std::vector<f_t>& x) const
{
  return compensated_dot2(problem_->h_obj_coeffs.data(), x.data(), x.size());
}

template <typename i_t, typename f_t>
bool repair_lns_t<i_t, f_t>::is_integer_var(i_t j) const
{
  return types_[j] == var_t::INTEGER;
}

template <typename i_t, typename f_t>
lns_neighborhood_t<i_t, f_t> repair_lns_t<i_t, f_t>::make_neighborhood(
  const lns_repair_request_t<f_t>& request) const
{
  const auto& p = *problem_;
  const i_t n   = p.n_variables;
  cuopt_assert(request.start.size() == (size_t)n && request.lower.size() == (size_t)n &&
                 request.upper.size() == (size_t)n,
               "repair vectors must cover every variable");
  lns_neighborhood_t<i_t, f_t> nb;
  nb.full = request.start;
  std::vector<i_t> index(n, -1);
  for (i_t j = 0; j < n; ++j) {
    f_t lo = request.lower[j], hi = request.upper[j];
    if (!std::isfinite(nb.full[j]) || std::isnan(lo) || std::isnan(hi) || lo < lower(j) ||
        hi > upper(j)) {
      nb.possible = false;
      return nb;
    }
    if (is_integer_var(j)) {
      lo = std::ceil(lo);
      hi = std::floor(hi);
    }
    if (lo > hi || lo == std::numeric_limits<f_t>::infinity() ||
        hi == -std::numeric_limits<f_t>::infinity()) {
      nb.possible = false;
      return nb;
    }
    if (lo == hi) {
      nb.full[j] = lo;
      continue;
    }
    index[j] = nb.free_columns.size();
    nb.free_columns.push_back(j);
    nb.lower.push_back(lo);
    nb.upper.push_back(hi);
    nb.types.push_back(types_[j]);
    nb.objective.push_back(p.h_obj_coeffs[j]);
    f_t value = std::clamp(nb.full[j], lo, hi);
    if (is_integer_var(j)) value = std::clamp(std::round(value), lo, hi);
    nb.start.push_back(value);
  }
  for (i_t r = 0; r < p.n_constraints; ++r) {
    f_t fixed = 0, correction = 0;
    const size_t begin = nb.columns.size();
    for (i_t k = p.offsets[r]; k < p.offsets[r + 1]; ++k) {
      const i_t j = p.variables[k];
      if (index[j] >= 0) {
        nb.columns.push_back(index[j]);
        nb.coefficients.push_back(p.coefficients[k]);
      } else {
        const f_t term = p.coefficients[k] * nb.full[j] - correction;
        const f_t next = fixed + term;
        correction     = (next - fixed) - term;
        fixed          = next;
      }
    }
    if (!std::isfinite(fixed)) {
      nb.possible = false;
      return nb;
    }
    const f_t lo = p.cstr_lb[r] - fixed, hi = p.cstr_ub[r] - fixed;
    if (begin == nb.columns.size()) {
      if (lo > row_tolerances_[r] || hi < -row_tolerances_[r]) {
        nb.possible = false;
        return nb;
      }
      continue;
    }
    nb.row_lower.push_back(lo);
    nb.row_upper.push_back(hi);
    nb.offsets.push_back(nb.columns.size());
  }
  return nb;
}

template <typename i_t, typename f_t>
void repair_lns_t<i_t, f_t>::consider_repair_candidate(const lns_repair_request_t<f_t>& request,
                                                       const lns_neighborhood_t<i_t, f_t>& nb,
                                                       const std::vector<f_t>& reduced,
                                                       const cuopt::timer_t& timer,
                                                       lns_repair_result_t<f_t>& result) const
{
  if (stopped() || timer.check_time_limit() || reduced.size() != nb.free_columns.size()) return;
  auto full = nb.full;
  for (size_t j = 0; j < reduced.size(); ++j)
    full[nb.free_columns[j]] = reduced[j];
  const f_t int_tol = problem_->tolerances.integrality_tolerance;
  for (size_t j = 0; j < full.size(); ++j)
    if (full[j] < request.lower[j] - int_tol || full[j] > request.upper[j] + int_tol) return;
  if (!feasible(full)) return;
  const f_t objective = cost(full);
  if (stopped() || timer.check_time_limit()) return;
  if (!result.feasible || objective < result.objective) {
    result.feasible   = true;
    result.assignment = std::move(full);
    result.objective  = objective;
  }
}

template <typename i_t, typename f_t>
lns_repair_result_t<f_t> repair_lns_t<i_t, f_t>::repair(const lns_repair_request_t<f_t>& request,
                                                        lns_repair_backend_t backend,
                                                        const raft::handle_t* handle)
{
  cuopt_assert(std::isfinite(request.time_limit_seconds) && request.time_limit_seconds >= 0 &&
                 request.max_iterations > 0 && request.max_nodes > 0,
               "invalid repair time/work budget");
  const double budget = std::min((double)request.time_limit_seconds, timer_.remaining_time());
  const cuopt::timer_t timer(budget);
  lns_repair_result_t<f_t> result;
  if (budget <= 0 || stopped()) return result;
  auto nb             = make_neighborhood(request);
  const auto consider = [&](const std::vector<f_t>& reduced) {
    consider_repair_candidate(request, nb, reduced, timer, result);
  };
  if (!nb.possible || timer.check_time_limit()) return result;
  if (nb.free_columns.empty()) {
    consider({});
    return result;
  }
  // Search-free neighborhoods need no engine and avoid empty-matrix corner cases.
  if (nb.row_lower.empty()) {
    auto x = nb.start;
    for (size_t j = 0; j < x.size(); ++j) {
      const f_t value = nb.objective[j] >= 0 ? nb.lower[j] : nb.upper[j];
      if (std::isfinite(value)) x[j] = value;
    }
    consider(x);
    return result;
  }
  if (backend == lns_repair_backend_t::cpufj) {
    fj_settings_t settings;
    settings.seed            = request.seed;
    settings.iteration_limit = request.max_iterations;
    auto climber             = init_fj_cpu_from_host_model<i_t, f_t>((i_t)nb.lower.size(),
                                                         (i_t)nb.row_lower.size(),
                                                         (i_t)nb.coefficients.size(),
                                                         false,
                                                         f_t{1},
                                                         f_t{0},
                                                         nb.coefficients,
                                                         nb.columns,
                                                         nb.offsets,
                                                         nb.objective,
                                                         nb.lower,
                                                         nb.upper,
                                                         nb.row_lower,
                                                         nb.row_upper,
                                                                     {},
                                                                     {},
                                                         nb.types,
                                                         problem_->tolerances,
                                                         preemption_,
                                                         settings);
    // The factory initializes its own seed; select the caller's seed before search.
    climber->settings.seed          = request.seed;
    climber->h_assignment           = nb.start;
    climber->h_best_assignment      = nb.start;
    climber->suppress_incumbent_log = true;
    climber->improvement_callback   = [&](f_t, const std::vector<f_t>& x, double) { consider(x); };
    if (!stopped() && !timer.check_time_limit()) cpufj_solve(climber.get(), timer.remaining_time());
    return result;
  }

  simplex::user_problem_t<i_t, f_t> sub_problem(handle);
  sub_problem.num_cols     = nb.lower.size();
  sub_problem.num_rows     = nb.row_lower.size();
  sub_problem.objective    = nb.objective;
  sub_problem.lower        = nb.lower;
  sub_problem.upper        = nb.upper;
  sub_problem.problem_name = "LNS-repair";
  csr_matrix_t<i_t, f_t> csr(sub_problem.num_rows, sub_problem.num_cols, nb.coefficients.size());
  csr.x         = nb.coefficients;
  csr.j         = nb.columns;
  csr.row_start = nb.offsets;
  csr.to_compressed_col(sub_problem.A);
  for (i_t r = 0; r < sub_problem.num_rows; ++r) {
    const f_t lo = nb.row_lower[r], hi = nb.row_upper[r];
    if (lo == hi) {
      sub_problem.row_sense.push_back('E');
      sub_problem.rhs.push_back(lo);
    } else if (!std::isfinite(hi)) {
      sub_problem.row_sense.push_back('G');
      sub_problem.rhs.push_back(lo);
    } else if (!std::isfinite(lo)) {
      sub_problem.row_sense.push_back('L');
      sub_problem.rhs.push_back(hi);
    } else {
      sub_problem.row_sense.push_back('E');
      sub_problem.rhs.push_back(lo);
      sub_problem.range_rows.push_back(r);
      sub_problem.range_value.push_back(hi - lo);
    }
  }
  sub_problem.num_range_rows = sub_problem.range_rows.size();
  for (auto type : nb.types)
    sub_problem.var_types.push_back(type == var_t::INTEGER ? simplex::variable_type_t::INTEGER
                                                           : simplex::variable_type_t::CONTINUOUS);
  simplex::simplex_solver_settings_t<i_t, f_t> settings;
  settings.time_limit                               = timer.remaining_time();
  settings.num_threads                              = 1;
  settings.node_limit                               = request.max_nodes;
  settings.branch_and_bound_simplex_iteration_limit = request.max_iterations;
  settings.random_seed                              = request.seed;
  settings.print_presolve_stats                     = false;
  settings.log.log                                  = false;
  settings.primal_tol                               = problem_->tolerances.absolute_tolerance;
  settings.integer_tol                              = problem_->tolerances.integrality_tolerance;
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
  settings.solution_callback = [&](std::vector<f_t>& x, f_t) { consider(x); };
  if (!stopped() && !timer.check_time_limit()) {
    probing_implied_bound_t<i_t, f_t> probing(sub_problem.num_cols);
    // The sub-MIP's remaining-time budget starts here, after neighborhood setup.
    const cuopt::timer_t submip_timer(settings.time_limit);
    branch_and_bound_t<i_t, f_t> solver(
      sub_problem, settings, submip_timer.get_tic_start(), probing);
    solver.set_initial_guess(nb.start);
    solver.set_submip_halt_callback(
      [&](f_t, f_t) { return stopped() || timer.check_time_limit(); });
    simplex::mip_solution_t<i_t, f_t> solution(sub_problem.num_cols);
    solver.solve(solution);
  }
  return result;
}

template <typename i_t, typename f_t>
size_t repair_lns_t<i_t, f_t>::refresh_scores(
  const std::vector<f_t>& current,
  const std::vector<i_t>& movable,
  const std::vector<std::vector<std::pair<i_t, f_t>>>& column_rows,
  std::vector<std::pair<double, i_t>>& variable_scores) const
{
  const auto& p = *problem_;
  variable_scores.clear();
  variable_scores.reserve(movable.size());
  for (i_t j : movable) {
    const f_t val            = current[j];
    const double obj_contrib = std::abs(p.h_obj_coeffs[j] * val);
    double activity          = 0.0;
    for (const auto& [r, norm_coeff] : column_rows[j])
      activity += std::abs(norm_coeff * val);
    const double combined_score = 0.5 * obj_contrib + 0.5 * activity;
    variable_scores.emplace_back(combined_score, j);
  }
  const size_t top_count = std::max(size_t{1}, variable_scores.size() / 5);
  if (!variable_scores.empty()) {
    std::nth_element(variable_scores.begin(),
                     variable_scores.begin() + std::min(top_count, variable_scores.size() - 1),
                     variable_scores.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
  }
  return top_count;
}

template <typename i_t, typename f_t>
void repair_lns_t<i_t, f_t>::run_backend(search_state_t& state,
                                         const std::vector<f_t>& lo_bounds,
                                         const std::vector<f_t>& hi_bounds,
                                         const backend_config_t& config)
{
  lns_repair_request_t<f_t> request;
  request.start              = state.current;
  request.lower              = lo_bounds;
  request.upper              = hi_bounds;
  request.time_limit_seconds = config.time_limit;
  request.seed               = state.rng();
  request.max_iterations     = config.max_iterations;
  request.max_nodes          = config.max_nodes;
  auto result                = repair(request, config.backend, &state.repair_handle);
  state.bandit_arms[config.arm].pulls++;
  if (result.feasible && result.objective < state.current_cost) {
    state.current           = result.assignment;
    state.current_cost      = result.objective;
    state.improvement_found = true;
    state.bandit_arms[config.arm].reward +=
      state.current_cost < state.best_known_cost ? config.new_best_reward : config.improve_reward;
    if (result.objective < state.best_known_cost) {
      state.submit(result.assignment, result.objective);
      state.best_known      = result.assignment;
      state.best_known_cost = result.objective;
    }
  } else {
    state.bandit_arms[config.arm].reward -= config.penalty;
  }
}

// Tighten every touched row to a fixpoint, recording only actual bound changes so the
// branch-and-propagate search can undo them without copying the bound vectors.
template <typename i_t, typename f_t>
bool repair_lns_t<i_t, f_t>::propagate(branch_state_t& episode) const
{
  const auto& p     = *problem_;
  const f_t int_tol = p.tolerances.integrality_tolerance;
  auto& lo          = episode.lo;
  auto& hi          = episode.hi;
  auto& trail       = episode.trail;
  bool changed      = true;
  int passes        = 0;
  while (changed && passes++ < episode.max_propagate_passes) {
    changed = false;
    for (i_t r : episode.rows) {
      f_t min_activity = 0, max_activity = 0;
      for (i_t k = p.offsets[r]; k < p.offsets[r + 1]; ++k) {
        const i_t v = p.variables[k];
        const f_t a = p.coefficients[k];
        min_activity += a * (a >= 0 ? lo[v] : hi[v]);
        max_activity += a * (a >= 0 ? hi[v] : lo[v]);
      }
      if (min_activity > p.cstr_ub[r] + row_tolerances_[r] ||
          max_activity < p.cstr_lb[r] - row_tolerances_[r])
        return false;
      for (i_t k = p.offsets[r]; k < p.offsets[r + 1]; ++k) {
        const i_t v = p.variables[k];
        const f_t a = p.coefficients[k];
        if (!episode.chosen[v] || a == 0 || lo[v] == hi[v]) continue;
        const f_t other_min = min_activity - a * (a >= 0 ? lo[v] : hi[v]);
        const f_t other_max = max_activity - a * (a >= 0 ? hi[v] : lo[v]);
        f_t new_lo          = (p.cstr_lb[r] - other_max) / a;
        f_t new_hi          = (p.cstr_ub[r] - other_min) / a;
        if (a < 0) std::swap(new_lo, new_hi);
        if (is_integer_var(v)) {
          new_lo = std::ceil(new_lo - int_tol);
          new_hi = std::floor(new_hi + int_tol);
        }
        new_lo = std::max(lo[v], new_lo);
        new_hi = std::min(hi[v], new_hi);
        if (new_lo > new_hi) return false;
        if (new_lo > lo[v] || new_hi < hi[v]) {
          trail.push_back({v, lo[v], hi[v]});
          lo[v]   = new_lo;
          hi[v]   = new_hi;
          changed = true;
        }
      }
    }
  }
  return !changed;
}

template <typename i_t, typename f_t>
void repair_lns_t<i_t, f_t>::rollback(branch_state_t& episode, size_t mark) const
{
  while (episode.trail.size() > mark) {
    const bound_change_t change = episode.trail.back();
    episode.trail.pop_back();
    episode.lo[change.variable] = change.old_lo;
    episode.hi[change.variable] = change.old_hi;
  }
}

template <typename i_t, typename f_t>
void repair_lns_t<i_t, f_t>::search(search_state_t& state, branch_state_t& episode)
{
  const auto& p          = *problem_;
  const auto& ruined     = episode.ruined;
  auto& lo               = episode.lo;
  auto& hi               = episode.hi;
  auto& trail            = episode.trail;
  const size_t node_mark = trail.size();
  if (episode.nodes++ >= episode.node_budget || stopped() ||
      episode.search_timer.check_time_limit() || !propagate(episode)) {
    rollback(episode, node_mark);
    return;
  }
  f_t objective_bound = episode.fixed_objective;
  for (i_t j : ruined)
    objective_bound += p.h_obj_coeffs[j] * (p.h_obj_coeffs[j] >= 0 ? lo[j] : hi[j]);
  if (objective_bound >= state.current_cost - f_t{1e-9}) {
    rollback(episode, node_mark);
    return;
  }

  i_t branch          = -1;
  f_t smallest_domain = std::numeric_limits<f_t>::infinity();
  for (i_t j : ruined) {
    if (hi[j] <= lo[j]) continue;
    const f_t domain = is_integer_var(j) ? hi[j] - lo[j] + 1 : f_t{3};
    if (domain < smallest_domain) {
      smallest_domain = domain;
      branch          = j;
    }
  }
  if (branch < 0) {
    // Propagation arithmetic alone does not certify the full assignment.
    f_t leaf_cost = episode.fixed_objective;
    for (i_t j : ruined)
      leaf_cost += p.h_obj_coeffs[j] * lo[j];
    if (leaf_cost < state.current_cost - f_t{1e-9} && feasible(lo)) {
      state.current           = lo;
      state.current_cost      = leaf_cost;
      state.improvement_found = true;
      state.bandit_arms[2].pulls++;
      state.bandit_arms[2].reward += state.current_cost < state.best_known_cost ? 1.5 : 0.5;
      if (leaf_cost < state.best_known_cost) {
        state.submit(lo, leaf_cost);
        state.best_known      = lo;
        state.best_known_cost = leaf_cost;
      }
    }
    rollback(episode, node_mark);
    return;
  }

  std::vector<f_t> values;
  if (is_integer_var(branch)) {
    values = lns_integer_values(lo[branch], hi[branch]);
  } else {
    values = {lo[branch], std::clamp(state.current[branch], lo[branch], hi[branch]), hi[branch]};
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
  }
  if (p.h_obj_coeffs[branch] < 0) std::reverse(values.begin(), values.end());
  for (f_t value : values) {
    if (episode.nodes >= episode.node_budget || stopped()) break;
    const size_t child_mark = trail.size();
    trail.push_back({branch, lo[branch], hi[branch]});
    lo[branch] = hi[branch] = value;
    search(state, episode);
    rollback(episode, child_mark);
  }
  rollback(episode, node_mark);
}

template <typename i_t, typename f_t>
void repair_lns_t<i_t, f_t>::run(const seed_fn& seeds, const submit_fn& submit)
{
  const auto& p = *problem_;
  const i_t n   = p.n_variables;
  if (!n) return;
  std::mt19937_64 rng(seed_);
  raft::handle_t repair_handle;

  std::vector<std::vector<std::pair<i_t, f_t>>> column_rows(n);
  std::vector<f_t> row_l1_norm(p.n_constraints, f_t{1e-10});
  for (i_t r = 0; r < p.n_constraints; ++r) {
    for (i_t k = p.offsets[r]; k < p.offsets[r + 1]; ++k)
      row_l1_norm[r] += std::abs(p.coefficients[k]);
  }
  for (i_t r = 0; r < p.n_constraints; ++r) {
    for (i_t k = p.offsets[r]; k < p.offsets[r + 1]; ++k)
      column_rows[p.variables[k]].emplace_back(r, p.coefficients[k] / row_l1_norm[r]);
  }

  std::vector<i_t> movable;
  for (i_t j = 0; j < n; ++j) {
    if (lower(j) < upper(j)) movable.push_back(j);
  }
  if (movable.empty()) return;

  bandit_arm_t bandit_arms[3];

  const size_t calibrated_ruin_size = 15;
  const int failure_threshold       = 3;
  const size_t max_ruin_size        = std::min(calibrated_ruin_size * 4, movable.size());

  // Bound-propagation cost per branch-and-bound node scales with row density.
  double avg_row_nnz = 1.0;
  if (p.n_constraints > 0) avg_row_nnz = (double)p.coefficients.size() / p.n_constraints;
  const double density_scale     = std::min(1.0, 30.0 / std::max(1.0, avg_row_nnz));
  const int base_node_budget     = std::max(100, (int)(1500 * density_scale));
  const int max_propagate_passes = std::max(5, (int)(40 * density_scale));

  int consecutive_failures = 0;
  std::vector<f_t> best_known;
  f_t best_known_cost = std::numeric_limits<f_t>::infinity();
  std::vector<std::vector<f_t>> population;

  while (!stopped()) {
    population.clear();
    seeds(population);
    population.erase(
      std::remove_if(
        population.begin(), population.end(), [this](const auto& x) { return !feasible(x); }),
      population.end());
    for (const auto& sol : population) {
      const f_t c = cost(sol);
      if (c < best_known_cost) {
        best_known_cost = c;
        best_known      = sol;
      }
    }
    if (best_known.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    // Mostly elitist, so a local improvement is likely a new global best. Occasionally
    // diversify from a fresh population member to avoid stagnation.
    std::vector<f_t> current;
    if (!population.empty() && (int)(rng() % 100) < 10) {
      current = population[(size_t)rng() % population.size()];
    } else {
      current = best_known;
    }
    if (!clamp_and_validate_cpufj_lns_seed(p, bounds_, types_, current)) continue;
    f_t current_cost = cost(current);

    // Seed-selection scores are O(nnz). Recompute them on improvement or periodically.
    std::vector<std::pair<double, i_t>> variable_scores;
    size_t top_count             = refresh_scores(current, movable, column_rows, variable_scores);
    int iterations_since_refresh = 0;
    const int refresh_period     = 30;

    // Bound each restart by wall clock as well, so dense instances still refresh seeds and
    // respond to stop requests.
    const int max_inner_iterations = 400;
    const cuopt::timer_t inner_loop_timer(2.0);
    for (int iteration = 0;
         iteration < max_inner_iterations && !stopped() && !inner_loop_timer.check_time_limit();
         ++iteration) {
      size_t current_ruin_size = calibrated_ruin_size;
      if (consecutive_failures >= failure_threshold) {
        current_ruin_size = std::min(
          max_ruin_size, current_ruin_size + (consecutive_failures - failure_threshold + 1) * 5);
      }
      const size_t ruin_size = std::min(current_ruin_size, movable.size());

      if (++iterations_since_refresh >= refresh_period) {
        top_count                = refresh_scores(current, movable, column_rows, variable_scores);
        iterations_since_refresh = 0;
      }

      const size_t selection_idx = (size_t)rng() % top_count;
      const i_t seed_var         = variable_scores[selection_idx].second;

      std::vector<i_t> ruined{seed_var};
      std::vector<bool> chosen(n, false);
      chosen[seed_var] = true;

      while (ruined.size() < ruin_size) {
        const i_t last = ruined.back();
        std::vector<std::pair<double, i_t>> candidates;
        std::vector<i_t> candidate_vars;
        for (const auto& [r, norm_coeff] : column_rows[last]) {
          for (i_t k = p.offsets[r]; k < p.offsets[r + 1]; ++k) {
            const i_t j = p.variables[k];
            if (!chosen[j] && lower(j) < upper(j)) candidate_vars.push_back(j);
          }
        }
        std::sort(candidate_vars.begin(), candidate_vars.end());
        candidate_vars.erase(std::unique(candidate_vars.begin(), candidate_vars.end()),
                             candidate_vars.end());
        if (candidate_vars.size() > 100) candidate_vars.resize(100);

        for (i_t j : candidate_vars) {
          const auto& left_col  = column_rows[last];
          const auto& right_col = column_rows[j];
          size_t a = 0, b = 0, shared_constraints = 0;
          while (a < left_col.size() && b < right_col.size()) {
            if (left_col[a].first < right_col[b].first) {
              ++a;
            } else if (right_col[b].first < left_col[a].first) {
              ++b;
            } else {
              ++shared_constraints;
              ++a;
              ++b;
            }
          }
          const double value_similarity = std::exp(-std::abs(current[last] - current[j]));
          candidates.emplace_back(0.7 * shared_constraints + 0.3 * value_similarity, j);
        }
        if (candidates.empty()) break;
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
          return a.first > b.first;
        });

        i_t selected_candidate;
        if ((int)(rng() % 100) < 80 && candidates[0].first > 1e-6) {
          selected_candidate = candidates[0].second;
        } else {
          const size_t selection_range = std::min(size_t{3}, candidates.size());
          selected_candidate           = candidates[(size_t)rng() % selection_range].second;
        }
        ruined.push_back(selected_candidate);
        chosen[selected_candidate] = true;
      }

      // Expand each ruined domain relative to its own range.
      auto lo_bounds = current, hi_bounds = current;
      std::vector<i_t> rows;
      bool valid_domain = true;
      for (i_t j : ruined) {
        const f_t domain_width = upper(j) - lower(j);
        f_t expansion;
        if (is_integer_var(j)) {
          expansion = 2;
        } else if (std::isfinite(domain_width) && domain_width < f_t{1e12}) {
          expansion = std::max(f_t{2}, f_t{0.2} * domain_width);
        } else {
          expansion = std::max(f_t{2}, f_t{0.1} * std::abs(current[j]) + 1);
        }
        lo_bounds[j] = std::max(lower(j), current[j] - expansion);
        hi_bounds[j] = std::min(upper(j), current[j] + expansion);
        if (is_integer_var(j)) {
          lo_bounds[j] = std::ceil(lo_bounds[j]);
          hi_bounds[j] = std::floor(hi_bounds[j]);
        }
        if (lo_bounds[j] > hi_bounds[j]) {
          valid_domain = false;
          break;
        }
        for (const auto& entry : column_rows[j])
          rows.push_back(entry.first);
      }
      if (!valid_domain || rows.empty()) continue;
      std::sort(rows.begin(), rows.end());
      rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

      repair_operator_t selected_operator = repair_operator_t::BP;
      if (ruined.size() <= 25) {
        double best_ucb = -1e6;
        for (int op = 0; op < 3; ++op) {
          const double ucb = bandit_arms[op].ucb();
          if (ucb > best_ucb) {
            best_ucb          = ucb;
            selected_operator = (repair_operator_t)op;
          }
        }
      }

      bool improvement_found = false;
      search_state_t state{current,
                           current_cost,
                           best_known,
                           best_known_cost,
                           bandit_arms,
                           rng,
                           repair_handle,
                           submit,
                           improvement_found};

      if (selected_operator == repair_operator_t::CPUFJ) {
        run_backend(state,
                    lo_bounds,
                    hi_bounds,
                    {lns_repair_backend_t::cpufj, 0, f_t{0.15}, 3000, 200, 2.0, 1.0, 0.1});
      }
      if (selected_operator == repair_operator_t::SUBMIP) {
        if (ruined.size() <= 20) {
          run_backend(state,
                      lo_bounds,
                      hi_bounds,
                      {lns_repair_backend_t::submip, 1, f_t{0.2}, 5000, 500, 3.0, 1.5, 0.15});
        } else {
          selected_operator = repair_operator_t::BP;
        }
      }

      // One propagate() call costs O(passes * touched rows * row nnz) and is not bounded by
      // the per-node deadline, so skip BP for oversized touched-row sets.
      if (selected_operator == repair_operator_t::BP && rows.size() > 4000) {
        bandit_arms[2].pulls++;
        bandit_arms[2].reward -= 0.05;
      } else if (selected_operator == repair_operator_t::BP) {
        int nodes = 0;
        const int node_budget =
          (int)(base_node_budget * calibrated_ruin_size / std::max(current_ruin_size, size_t{1}));
        auto& lo = lo_bounds;
        auto& hi = hi_bounds;
        std::vector<bound_change_t> trail;
        trail.reserve(ruined.size() * 4);

        // The non-ruined variables are fixed for the whole episode, so only the ruined
        // variables' contribution changes per node.
        f_t fixed_objective = 0;
        for (i_t j = 0; j < n; ++j)
          fixed_objective += p.h_obj_coeffs[j] * current[j];
        for (i_t j : ruined)
          fixed_objective -= p.h_obj_coeffs[j] * current[j];
        const cuopt::timer_t search_timer(0.03);
        branch_state_t episode{rows,
                               chosen,
                               ruined,
                               max_propagate_passes,
                               node_budget,
                               fixed_objective,
                               lo,
                               hi,
                               trail,
                               nodes,
                               search_timer};
        search(state, episode);

        if (!improvement_found) {
          bandit_arms[2].pulls++;
          bandit_arms[2].reward -= 0.05;
        }
      }

      if (improvement_found) {
        // A backend can return tolerance-feasible values. Recheck its normalized copy before
        // the next iteration uses it for exact fixings.
        if (!clamp_and_validate_cpufj_lns_seed(p, bounds_, types_, current)) break;
        current_cost             = cost(current);
        consecutive_failures     = 0;
        top_count                = refresh_scores(current, movable, column_rows, variable_scores);
        iterations_since_refresh = 0;
      } else {
        consecutive_failures++;
      }
    }
  }
}

#if MIP_INSTANTIATE_DOUBLE
template class repair_lns_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
