/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../benchmarks/linear_programming/cuopt/c_api_check.hpp"

#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/feasibility_jump/early_cpufj.cuh>
#include <mip_heuristics/lns/cpufj_validation.cuh>
#include <mip_heuristics/lns/early.cuh>
#include <mip_heuristics/lns/repair_lns.cuh>
#include <mip_heuristics/lns/thread_budget.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <limits>
#include <numeric>
#include <thread>

namespace cuopt::lns::test {
namespace mip = cuopt::mathematical_optimization::mip;
namespace opt = cuopt::mathematical_optimization;

using tolerances_t = opt::mip_solver_settings_t<int, double>::tolerances_t;

struct host_model_t {
  std::vector<double> coefficients, lower, upper, objective, row_lower, row_upper;
  std::vector<int> columns, offsets;
  std::vector<opt::var_t> types;
};

tolerances_t test_tolerances()
{
  tolerances_t tolerances;
  tolerances.absolute_tolerance    = 1e-7;
  tolerances.relative_tolerance    = 2e-8;
  tolerances.integrality_tolerance = 1e-4;
  return tolerances;
}

host_model_t single_variable(
  bool integer, double lower, double upper, double row_lower, double row_upper)
{
  return {{1.0},
          {lower},
          {upper},
          {1.0},
          {row_lower},
          {row_upper},
          {0},
          {0, 1},
          {integer ? opt::var_t::INTEGER : opt::var_t::CONTINUOUS}};
}

// min x0 + 2 x1 subject to 1 <= x0 + x1 <= 2 over binaries.
host_model_t covering_pair()
{
  return {{1.0, 1.0},
          {0.0, 0.0},
          {1.0, 1.0},
          {1.0, 2.0},
          {1.0},
          {2.0},
          {0, 1},
          {0, 2},
          {opt::var_t::INTEGER, opt::var_t::INTEGER}};
}

std::unique_ptr<mip::fj_cpu_climber_t<int, double>> make_anchor(const host_model_t& m,
                                                                std::atomic<bool>& preemption,
                                                                tolerances_t tolerances)
{
  return mip::init_fj_cpu_from_host_model<int, double>((int)m.lower.size(),
                                                       (int)m.row_lower.size(),
                                                       (int)m.coefficients.size(),
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
                                                       m.types,
                                                       tolerances,
                                                       preemption,
                                                       mip::fj_settings_t{});
}

TEST(Lns, NeighborhoodKeepsStrictDomainsAndRejectsEmptyIntegerDomains)
{
  std::atomic<bool> preemption{false};
  auto anchor = make_anchor(single_variable(false, 0, 1, -1, 1), preemption, test_tolerances());
  mip::repair_lns_t<int, double> lns(*anchor, preemption, 42);
  mip::lns_repair_request_t<double> request;
  request.start = request.lower = request.upper = {0.0};
  EXPECT_TRUE(lns.make_neighborhood(request).possible);
  request.lower = request.upper = {-5e-5};
  EXPECT_FALSE(lns.make_neighborhood(request).possible);
  request.lower = {0.0};
  request.upper = {1.0 + 1e-10};
  EXPECT_FALSE(lns.make_neighborhood(request).possible);

  auto integer_anchor =
    make_anchor(single_variable(true, 0, 1, 0, 1), preemption, test_tolerances());
  mip::repair_lns_t<int, double> integer_lns(*integer_anchor, preemption, 42);
  request.lower = {.2};
  request.upper = {.8};
  EXPECT_FALSE(integer_lns.make_neighborhood(request).possible);
}

TEST(Lns, NeighborhoodEliminatesFixedColumnsWithRelativeRowTolerance)
{
  std::atomic<bool> preemption{false};
  // x0 - x1 = 1e6 is not tightened by the climber's bound propagation.
  const host_model_t shifted{{1.0, -1.0},
                             {1e6, 0.0},
                             {1e6 + 1, 1.0},
                             {1.0, 0.0},
                             {1e6},
                             {1e6},
                             {0, 1},
                             {0, 2},
                             {opt::var_t::CONTINUOUS, opt::var_t::CONTINUOUS}};
  auto anchor = make_anchor(shifted, preemption, test_tolerances());
  mip::repair_lns_t<int, double> lns(*anchor, preemption, 42);
  EXPECT_TRUE(lns.feasible({1e6 + .01, 0}));
  EXPECT_FALSE(lns.feasible({1e6 + .03, 0}));
  mip::lns_repair_request_t<double> request;
  request.start = request.lower = request.upper = {1e6 + .01, 0};
  EXPECT_TRUE(lns.make_neighborhood(request).possible);
  request.start = request.lower = request.upper = {1e6 + .03, 0};
  EXPECT_FALSE(lns.make_neighborhood(request).possible);

  auto pair_anchor = make_anchor(covering_pair(), preemption, test_tolerances());
  mip::repair_lns_t<int, double> pair_lns(*pair_anchor, preemption, 42);
  request.start           = {1, 1};
  request.lower           = {1, 0};
  request.upper           = {1, 1};
  const auto neighborhood = pair_lns.make_neighborhood(request);
  ASSERT_TRUE(neighborhood.possible);
  EXPECT_EQ(neighborhood.free_columns, (std::vector<int>{1}));
  EXPECT_EQ(neighborhood.row_lower, (std::vector<double>{0}));
  EXPECT_EQ(neighborhood.row_upper, (std::vector<double>{1}));
  EXPECT_EQ(neighborhood.full, (std::vector<double>{1, 1}));
}

TEST(Lns, BothRepairBackendsContainWidenedNeighborhoods)
{
  raft::handle_t handle;
  std::atomic<bool> preemption{false};
  auto anchor = make_anchor(single_variable(false, 0, 1, 0, 1), preemption, test_tolerances());
  mip::repair_lns_t<int, double> lns(*anchor, preemption, 42);
  mip::lns_repair_request_t<double> request;
  request.start = {0.0};
  request.lower = {-1e-10};
  request.upper = {1.0};
  for (auto backend : {mip::lns_repair_backend_t::cpufj, mip::lns_repair_backend_t::submip}) {
    auto result = lns.repair(request, backend, &handle);
    EXPECT_FALSE(result.feasible);
    EXPECT_TRUE(result.assignment.empty());
    auto valid_request  = request;
    valid_request.lower = valid_request.upper = valid_request.start;
    auto recovered                            = lns.repair(valid_request, backend, &handle);
    EXPECT_TRUE(recovered.feasible);
    EXPECT_EQ(recovered.assignment, valid_request.start);
  }
}

TEST(Lns, BothRepairBackendsImproveWithinNeighborhood)
{
  raft::handle_t handle;
  std::atomic<bool> preemption{false};
  auto anchor = make_anchor(covering_pair(), preemption, test_tolerances());
  mip::repair_lns_t<int, double> lns(*anchor, preemption, 42);
  mip::lns_repair_request_t<double> request;
  request.start              = {1, 1};
  request.lower              = {0, 0};
  request.upper              = {1, 1};
  request.time_limit_seconds = 2;
  for (auto backend : {mip::lns_repair_backend_t::cpufj, mip::lns_repair_backend_t::submip}) {
    auto result = lns.repair(request, backend, &handle);
    ASSERT_TRUE(result.feasible);
    EXPECT_TRUE(lns.feasible(result.assignment));
    EXPECT_DOUBLE_EQ(result.objective, 1);
  }
}

TEST(Lns, RunImprovesFromToleranceFeasibleSeedWithoutMutatingSource)
{
  host_model_t model;
  const int n = 40;
  model.lower.assign(n, 0.0);
  model.upper.assign(n, 1.0);
  model.objective.assign(n, 1.0);
  model.coefficients.assign(n, 1.0);
  model.row_lower.assign(n, 0.0);
  model.row_upper.assign(n, 1.0);
  model.types.assign(n, opt::var_t::INTEGER);
  model.columns.resize(n);
  model.offsets.resize(n + 1);
  std::iota(model.columns.begin(), model.columns.end(), 0);
  std::iota(model.offsets.begin(), model.offsets.end(), 0);
  std::atomic<bool> preemption{false};
  auto anchor = make_anchor(model, preemption, test_tolerances());
  mip::repair_lns_t<int, double> lns(*anchor, preemption, 42);
  const std::vector<double> source(n, 1.0 - 5e-5);
  ASSERT_TRUE(lns.feasible(source));
  double last         = lns.cost(source);
  int submissions     = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  lns.run(
    [&](auto& seeds) {
      if (std::chrono::steady_clock::now() > deadline) lns.halted = true;
      seeds.push_back(source);
    },
    [&](const auto& x, double objective) {
      ++submissions;
      EXPECT_TRUE(lns.feasible(x));
      EXPECT_DOUBLE_EQ(objective, lns.cost(x));
      EXPECT_LT(objective, last);
      last = objective;
      if (objective <= 0) lns.halted = true;
    });
  EXPECT_GT(submissions, 0);
  EXPECT_LE(last, 0);
  EXPECT_EQ(source, std::vector<double>(n, 1.0 - 5e-5));
}

TEST(Lns, MainSolveBudgetMatchesPresolvePair)
{
  for (int size = 2; size <= 48; ++size) {
    const int workers = mip::lns_worker_count(size, false);
    EXPECT_EQ(workers, size >= 7 ? 2 : 0);
    EXPECT_EQ(mip::lns_worker_count(size, true), 0);
  }
}

TEST(Lns, CpufjLnsRevalidatesWithSolverTolerances)
{
  mip::fj_cpu_problem_t<int, double> problem;
  problem.n_variables = problem.n_constraints = 1;
  problem.offsets                             = {0, 1};
  problem.variables                           = {0};
  problem.coefficients = problem.h_obj_coeffs = {1.0};
  problem.h_var_types                         = {opt::var_t::CONTINUOUS};
  problem.cstr_lb = problem.cstr_ub        = {1e6};
  problem.tolerances.absolute_tolerance    = 1e-7;
  problem.tolerances.relative_tolerance    = 2e-8;
  problem.tolerances.integrality_tolerance = 1e-4;
  std::vector<double2> bounds{make_double2(1e6, 1e6 + 1)};
  EXPECT_TRUE(mip::verify_cpufj_lns_feasible(problem, bounds, {1e6 + .01}));
  EXPECT_FALSE(mip::verify_cpufj_lns_feasible(problem, bounds, {1e6 + .03}));
  EXPECT_FALSE(
    mip::verify_cpufj_lns_feasible(problem, bounds, {std::numeric_limits<double>::quiet_NaN()}));
  problem.h_var_types = {opt::var_t::INTEGER};
  bounds              = {make_double2(0, 1)};
  problem.cstr_lb     = {0};
  problem.cstr_ub     = {2};
  EXPECT_TRUE(mip::verify_cpufj_lns_feasible(problem, bounds, {1 + 5e-5}));
  EXPECT_FALSE(mip::verify_cpufj_lns_feasible(problem, bounds, {1 + 2e-4}));
  auto seed = std::vector<double>{1 - 5e-5};
  ASSERT_TRUE(mip::normalize_cpufj_lns_seed(problem, bounds, seed));
  EXPECT_EQ(seed[0], 1);
  problem.cstr_lb = problem.cstr_ub = {1 - 5e-5};
  seed                              = {1 - 5e-5};
  EXPECT_FALSE(mip::normalize_cpufj_lns_seed(problem, bounds, seed));
  bounds          = {make_double2(.2, .99999)};
  problem.cstr_lb = {0};
  problem.cstr_ub = {2};
  seed            = {.99999};
  EXPECT_FALSE(mip::normalize_cpufj_lns_seed(problem, bounds, seed));
}

void init_early_lns_test_problem(opt::optimization_problem_t<int, double>& op, bool integer = false)
{
  const std::vector<double> coefficients{1, 1}, lower{0, 0}, upper{1, 1}, objective{1, 2};
  const std::vector<double> row_lower{0}, row_upper{2};
  const std::vector<int> columns{0, 1}, offsets{0, 2};
  const std::vector<opt::var_t> types(2, integer ? opt::var_t::INTEGER : opt::var_t::CONTINUOUS);
  op.set_csr_constraint_matrix(coefficients.data(), 2, columns.data(), 2, offsets.data(), 2);
  op.set_variable_lower_bounds(lower.data(), 2);
  op.set_variable_upper_bounds(upper.data(), 2);
  op.set_variable_types(types.data(), 2);
  op.set_objective_coefficients(objective.data(), 2);
  op.set_constraint_lower_bounds(row_lower.data(), 1);
  op.set_constraint_upper_bounds(row_upper.data(), 1);
}

TEST(Lns, PresolveWorkerFailureIsCapturedAfterPublishingValidatedImprovement)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op, true);
  const double row_lower = 1;
  op.set_constraint_lower_bounds(&row_lower, 1);
  opt::mip_solver_settings_t<int, double> settings;
  std::atomic<bool> preemption{false};
  auto anchor =
    mip::init_fj_cpu_from_optimization_problem(op, settings.get_tolerances(), preemption);
  auto shared = std::make_shared<mip::fj_cpu_shared_incumbent_t<int, double>>();
  shared->publish(3, 3, {1, 1});
  std::atomic<bool> reported{false};
  cuopt::lns::task_errors_t task_errors;
#pragma omp parallel num_threads(7)
  {
#pragma omp masked
    {
      mip::early_lns_t<int, double> workers(
        *anchor,
        shared,
        preemption,
        task_errors,
        [&](double objective, const auto& x, const char*) {
          EXPECT_TRUE(omp_in_parallel());
          EXPECT_EQ(omp_get_max_threads(), 1);
          EXPECT_LT(objective, 3);
          EXPECT_GE(x[0] + x[1], 1);
          reported = true;
          throw std::runtime_error("injected presolve LNS failure");
        },
        42);
      workers.start();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!reported.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      EXPECT_NO_THROW(workers.finish());
      EXPECT_NO_THROW(workers.finish());
    }
  }
  EXPECT_TRUE(reported.load());
  EXPECT_THROW(task_errors.rethrow_if_error(), std::runtime_error);
  EXPECT_LT(shared->objective.load(), 3);
  EXPECT_FALSE(preemption.load());
}

TEST(Lns, PresolveSearchImprovesAFeasibleCpuIncumbent)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op, true);
  const double row_lower = 1;
  op.set_constraint_lower_bounds(&row_lower, 1);
  opt::mip_solver_settings_t<int, double> settings;
  std::atomic<bool> preemption{false};
  auto anchor =
    mip::init_fj_cpu_from_optimization_problem(op, settings.get_tolerances(), preemption);
  auto shared = std::make_shared<mip::fj_cpu_shared_incumbent_t<int, double>>();
  shared->publish(3, 3, {1, 1});
  std::atomic<int> reports{0};
  cuopt::lns::task_errors_t task_errors;
#pragma omp parallel num_threads(7)
  {
#pragma omp masked
    {
      mip::early_lns_t<int, double> workers(
        *anchor,
        shared,
        preemption,
        task_errors,
        [&](double objective, const auto& x, const char*) {
          ++reports;
          EXPECT_GE(objective, 1);
          EXPECT_GE(x[0] + x[1], 1);
        },
        42);
      workers.start();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (shared->objective.load() > 1 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      workers.finish();
    }
  }
  EXPECT_EQ(shared->objective.load(), 1);
  EXPECT_EQ(shared->assignment, (std::vector<double>{1, 0}));
  EXPECT_GT(reports.load(), 0);
  EXPECT_NO_THROW(task_errors.rethrow_if_error());
}

TEST(Lns, PresolvePortfolioBudgetAndLifetime)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op, true);
  opt::mip_solver_settings_t<int, double> settings;
  cuopt::lns::task_errors_t task_errors;
  for (int team_size : {2, 6, 7, 10}) {
#pragma omp parallel num_threads(team_size)
    {
#pragma omp masked
      {
        mip::early_cpufj_t<int, double> portfolio(
          op, settings.get_tolerances(), {}, 42, &task_errors);
        for (int restart = 0; restart < 2; ++restart) {
          // Even an oversized request must honor the presolve reservation.
          portfolio.start(team_size);
          EXPECT_EQ(portfolio.lane_count(), std::max(1, team_size - 4));
          EXPECT_EQ(portfolio.improvement_lane_count(), team_size >= 7 ? 2 : 0);
          EXPECT_NO_THROW(portfolio.stop());
          EXPECT_EQ(portfolio.lane_count(), 0);
          EXPECT_NO_THROW(portfolio.stop());
        }
      }
    }
  }
  EXPECT_NO_THROW(task_errors.rethrow_if_error());
  // The short initialization probe precedes the OMP team and keeps its single lane.
  mip::early_cpufj_t<int, double> probe(op, settings.get_tolerances(), {}, 42);
  probe.start(20, true);
  EXPECT_EQ(probe.lane_count(), 1);
  EXPECT_EQ(probe.improvement_lane_count(), 0);
  probe.stop();
}

TEST(Lns, PresolveIncumbentIsHandedToMainPopulationOnce)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op, true);
  const double row_lower = 1;
  op.set_constraint_lower_bounds(&row_lower, 1);
  opt::mip_solver_settings_t<int, double> settings;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  context.initial_incumbent_assignment = {1, 0};
  mip::diversity_manager_t<int, double> dm(context);
  std::vector<mip::solution_t<int, double>> solutions;
  dm.add_user_given_solutions(solutions);
  EXPECT_TRUE(solutions.empty());
  context.initial_incumbent_from_papilo_model = true;
  dm.add_user_given_solutions(solutions);
  ASSERT_EQ(solutions.size(), 1);
  EXPECT_TRUE(solutions[0].get_feasible());
  EXPECT_DOUBLE_EQ(solutions[0].get_user_objective(), 1);
}

TEST(Lns, BenchmarkApiErrorsAreDistinctFromNoSolution)
{
  EXPECT_NO_THROW(cuopt_bench::check_c_api(CUOPT_SUCCESS, "cuOptSolve"));
  try {
    cuopt_bench::check_c_api(CUOPT_RUNTIME_ERROR, "cuOptSolve");
    FAIL() << "API error was ignored";
  } catch (const cuopt_bench::c_api_error_t& error) {
    EXPECT_EQ(error.code, CUOPT_RUNTIME_ERROR);
    EXPECT_STREQ(error.operation, "cuOptSolve");
  }
}
TEST(Lns, PresolveBudgetsIncludePapiloAndAuxiliaryWorkers)
{
  for (int team_size = 2; team_size <= 128; ++team_size) {
    const int gpu_workers = team_size >= CUOPT_MIP_EARLY_GPUFJ_REQUIRED_THREAD_COUNT ? 1 : 0;
    for (bool recognized_structure : {false, true}) {
      const int structural_cpu_budget =
        mip::presolve_early_worker_budget(team_size, CUOPT_MIP_PAPILO_THREAD_LIMIT, gpu_workers, 1);
      const int structural_workers =
        recognized_structure &&
        mip::early_structural_has_capacity(team_size, structural_cpu_budget, gpu_workers);
      const int cpu_workers = mip::presolve_early_worker_budget(
        team_size, CUOPT_MIP_PAPILO_THREAD_LIMIT, gpu_workers, structural_workers);
      const int lns_workers = mip::presolve_lns_worker_count(cpu_workers);
      EXPECT_GE(cpu_workers - lns_workers, 1);
      EXPECT_EQ(lns_workers, team_size >= 7 ? 2 : 0);
      const int papilo_workers =
        mip::papilo_thread_budget(team_size, cpu_workers, gpu_workers, structural_workers);
      EXPECT_GE(papilo_workers, 1);
      EXPECT_LE(papilo_workers, CUOPT_MIP_PAPILO_THREAD_LIMIT);
      EXPECT_LE(cpu_workers + gpu_workers + structural_workers + papilo_workers, team_size);
      if (team_size >= 9) {
        EXPECT_EQ(papilo_workers, CUOPT_MIP_PAPILO_THREAD_LIMIT);
        EXPECT_EQ(cpu_workers + gpu_workers + structural_workers + papilo_workers, team_size);
      }

      // The replacement portfolio keeps the cuOpt caller/probing reservation.
      const int reduced_structural_budget =
        mip::presolve_early_worker_budget(team_size, CUOPT_MIP_EARLY_CPUFJ_RESERVED_THREADS, 0, 1);
      const int reduced_structural_workers =
        recognized_structure &&
        mip::early_structural_has_capacity(team_size, reduced_structural_budget, 0);
      const int reduced_workers = mip::presolve_early_worker_budget(
        team_size, CUOPT_MIP_EARLY_CPUFJ_RESERVED_THREADS, 0, reduced_structural_workers);
      const int probing_workers =
        mip::probing_thread_budget(team_size, reduced_workers, reduced_structural_workers);
      EXPECT_EQ(mip::presolve_lns_worker_count(reduced_workers), team_size >= 7 ? 2 : 0);
      EXPECT_LE(reduced_workers + reduced_structural_workers + 1, team_size);
      // At tiny team sizes the caller can execute the single probing task itself.
      EXPECT_LE(reduced_workers + reduced_structural_workers + probing_workers, team_size);
      if (team_size >= 8) {
        EXPECT_EQ(
          reduced_workers + reduced_structural_workers + CUOPT_MIP_EARLY_CPUFJ_RESERVED_THREADS,
          team_size);
      }
    }
    EXPECT_EQ(mip::papilo_thread_budget(team_size, 0, 0, 0),
              std::min(team_size, CUOPT_MIP_PAPILO_THREAD_LIMIT));
  }
  EXPECT_EQ(mip::presolve_early_worker_budget(28, 4, 1, 0), 23);
  EXPECT_EQ(mip::presolve_early_worker_budget(28, 4, 1, 1), 22);
  EXPECT_EQ(mip::presolve_early_worker_budget(28, 4, 0, 0), 24);
  EXPECT_EQ(mip::presolve_early_worker_budget(28, 4, 0, 1), 23);
  EXPECT_FALSE(mip::early_structural_has_capacity(2, 1, 0));
  EXPECT_FALSE(mip::early_structural_has_capacity(3, 1, 1));
}

}  // namespace cuopt::lns::test
