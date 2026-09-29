/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../benchmarks/linear_programming/cuopt/c_api_check.hpp"
#include "../../../experiments/hive_lns/bridge.cuh"

#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/feasibility_jump/early_cpufj.cuh>
#include <mip_heuristics/feasibility_jump/early_lns.cuh>
#include <mip_heuristics/feasibility_jump/persistent_lns_bridge.cuh>
#include <mip_heuristics/local_search/cpufj_lns_validation.cuh>

#include <gtest/gtest.h>

#include <atomic>
#include <limits>

namespace cuopt::hive_lns::test {
namespace mip = cuopt::mathematical_optimization::mip;
namespace opt = cuopt::mathematical_optimization;

model_t make_model(bool integer = false)
{
  model_t model;
  model.lower                 = {0.0};
  model.upper                 = {1.0};
  model.integer               = {integer};
  model.objective             = {1.0};
  model.offsets               = {0, 1};
  model.columns               = {0};
  model.coefficients          = {1.0};
  model.row_lower             = {0.0};
  model.row_upper             = {1.0};
  model.feasibility_tolerance = 1e-7;
  model.relative_tolerance    = 2e-8;
  model.integrality_tolerance = 1e-4;
  model.row_tolerances        = {mip::get_cstr_tolerance<int, double>(
    0.0, 1.0, model.feasibility_tolerance, model.relative_tolerance)};
  return model;
}

TEST(HiveLns, NormalizesPrivateSeedAndKeepsStrictDomains)
{
  auto model      = make_model();
  model.row_lower = {-1.0};
  const std::vector<double> population_member{-5e-5};
  auto seed = population_member;
  ASSERT_TRUE(model.feasible(seed));
  ASSERT_TRUE(model.normalize_seed(seed));
  EXPECT_EQ(seed[0], 0.0);
  EXPECT_EQ(population_member[0], -5e-5);

  repair_request_t request;
  request.start = request.lower = request.upper = seed;
  EXPECT_TRUE(make_repair_problem(model, request).possible);
  request.lower = request.upper = population_member;
  EXPECT_FALSE(make_repair_problem(model, request).possible);
  request.lower = {0.0};
  request.upper = {1.0 + 1e-10};
  EXPECT_FALSE(make_repair_problem(model, request).possible);
}

TEST(HiveLns, RoundsIntegersAndRejectsEmptyIntegerDomains)
{
  auto model = make_model(true);
  std::vector<double> seed{1.0 - 5e-5};
  ASSERT_TRUE(model.normalize_seed(seed));
  EXPECT_EQ(seed[0], 1.0);
  repair_request_t request;
  request.start = request.lower = request.upper = seed;
  EXPECT_TRUE(make_repair_problem(model, request).possible);

  model.lower = {.2};
  model.upper = {.99999};
  seed        = {.99999};
  ASSERT_TRUE(model.feasible(seed));
  EXPECT_FALSE(model.normalize_seed(seed));
}

TEST(HiveLns, RevalidatesRowsAfterRoundingOrClamping)
{
  auto model      = make_model(true);
  model.row_lower = model.row_upper = {1.0 - 5e-5};
  std::vector<double> seed{1.0 - 5e-5};
  ASSERT_TRUE(model.feasible(seed));
  EXPECT_FALSE(model.normalize_seed(seed));

  model.integer   = {false};
  model.row_lower = model.row_upper = {-5e-5};
  seed                              = {-5e-5};
  ASSERT_TRUE(model.feasible(seed));
  EXPECT_FALSE(model.normalize_seed(seed));
}

TEST(HiveLns, HonorsConfiguredBoundsIntegralityAndRelativeRowTolerance)
{
  auto model      = make_model(true);
  model.row_upper = {2.0};
  EXPECT_TRUE(model.feasible({1.0 + 5e-5}));
  model.integrality_tolerance = 1e-6;
  EXPECT_FALSE(model.feasible({1.0 + 5e-5}));
  EXPECT_FALSE(model.feasible({std::numeric_limits<double>::quiet_NaN()}));

  model.integer   = {false};
  model.lower     = {1e6};
  model.upper     = {1e6 + 1.0};
  model.row_lower = model.row_upper = {1e6};
  model.row_tolerances              = {mip::get_cstr_tolerance<int, double>(
    1e6, 1e6, model.feasibility_tolerance, model.relative_tolerance)};
  EXPECT_TRUE(model.feasible({1e6 + .01}));
  EXPECT_FALSE(model.feasible({1e6 + .03}));
  repair_request_t request;
  request.start = request.lower = request.upper = {1e6 + .01};
  EXPECT_TRUE(make_repair_problem(model, request).possible);
}

TEST(HiveLns, BothRepairBackendsContainWidenedNeighborhoods)
{
  auto model = make_model();
  repair_request_t request;
  request.start = {0.0};
  request.lower = {-1e-10};
  request.upper = {1.0};
  std::atomic<bool> preemption{false};
  for (auto backend : {repair_backend_t::cpufj, repair_backend_t::submip}) {
    auto result =
      repair_neighborhood(model, request, backend, [] { return false; }, nullptr, 1.0, preemption);
    EXPECT_FALSE(result.feasible);
    EXPECT_TRUE(result.assignment.empty());
    auto valid_request  = request;
    valid_request.lower = valid_request.upper = valid_request.start;
    auto recovered                            = repair_neighborhood(
      model, valid_request, backend, [] { return false; }, nullptr, 1.0, preemption);
    EXPECT_TRUE(recovered.feasible);
    EXPECT_EQ(recovered.assignment, valid_request.start);
  }
}

TEST(HiveLns, RenormalizesBackendOutputBeforeNextFixing)
{
  auto model = make_model(true);
  model.lower.assign(40, 0.0);
  model.upper.assign(40, 1.0);
  model.integer.assign(40, true);
  model.objective.assign(40, 1.0);
  model.offsets.resize(41);
  model.columns.resize(40);
  std::iota(model.offsets.begin(), model.offsets.end(), 0);
  std::iota(model.columns.begin(), model.columns.end(), 0);
  model.coefficients.assign(40, 1.0);
  model.row_lower.assign(40, 0.0);
  model.row_upper.assign(40, 1.0);
  model.row_tolerances.assign(40, model.row_tolerances.front());
  const population_t population{std::vector<double>(40, 1.0 - 5e-5)};
  int repairs = 0;
  auto repair = [&](const repair_request_t& request) {
    ++repairs;
    for (size_t j = 0; j < request.start.size(); ++j) {
      EXPECT_EQ(request.start[j], 1.0);
      EXPECT_GE(request.lower[j], model.lower[j]);
      EXPECT_LE(request.upper[j], model.upper[j]);
      EXPECT_EQ(request.lower[j], std::round(request.lower[j]));
      EXPECT_EQ(request.upper[j], std::round(request.upper[j]));
    }
    repair_result_t result;
    result.feasible   = true;
    result.assignment = population.front();
    result.objective  = model.cost(result.assignment);
    return result;
  };
  model.repair.cpufj  = repair;
  model.repair.submip = repair;
  int polls           = 0;
  run_lns(
    model,
    [&] { return population; },
    [](const auto&) {},
    [&] { return repairs >= 2 || ++polls > 1000; },
    42);
  EXPECT_EQ(repairs, 2);
  EXPECT_EQ(population.front()[1], 1.0 - 5e-5);
}

std::atomic<bool> worker_ran{false};

void failing_worker(
  const model_t& model, const snapshot_fn& snapshot, const submit_fn&, const stop_fn&, uint64_t)
{
  worker_ran = true;
  EXPECT_TRUE(omp_in_parallel());
  EXPECT_GE(omp_get_num_threads(), 9);
  EXPECT_EQ(omp_get_max_threads(), 1);
  EXPECT_EQ(model.feasibility_tolerance, 3e-7);
  EXPECT_EQ(model.relative_tolerance, 4e-8);
  EXPECT_EQ(model.integrality_tolerance, 2e-4);
  EXPECT_EQ(model.row_tolerances[0], (mip::get_cstr_tolerance<int, double>(1, 2, 3e-7, 4e-8)));
  EXPECT_FALSE(snapshot().empty());
  throw std::runtime_error("injected optional worker failure");
}

TEST(HiveLns, WorkerFailurePreservesValidatedPopulationIncumbent)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  const std::vector<double> coefficients{1, 1}, lower{0, 0}, upper{1, 1}, objective{1, 2};
  const std::vector<double> row_lower{1}, row_upper{2};
  const std::vector<int> columns{0, 1}, offsets{0, 2};
  const std::vector<opt::var_t> types(2, opt::var_t::INTEGER);
  op.set_csr_constraint_matrix(coefficients.data(), 2, columns.data(), 2, offsets.data(), 2);
  op.set_variable_lower_bounds(lower.data(), 2);
  op.set_variable_upper_bounds(upper.data(), 2);
  op.set_variable_types(types.data(), 2);
  op.set_objective_coefficients(objective.data(), 2);
  op.set_constraint_lower_bounds(row_lower.data(), 1);
  op.set_constraint_upper_bounds(row_upper.data(), 1);
  opt::mip_solver_settings_t<int, double> settings;
  settings.tolerances.absolute_tolerance    = 3e-7;
  settings.tolerances.relative_tolerance    = 4e-8;
  settings.tolerances.integrality_tolerance = 2e-4;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();
  mip::solution_t<int, double> incumbent(problem);
  incumbent.copy_new_assignment(std::vector<double>{1, 0});
  ASSERT_TRUE(incumbent.compute_feasibility());
  dm.population.add_solution(std::move(incumbent));
  ASSERT_TRUE(dm.population.is_feasible());
  const auto before = dm.population.best_feasible().get_host_assignment();
  for (int team_size : {2, 8, 9, 10}) {
    worker_ran = false;
#pragma omp parallel num_threads(team_size)
    {
#pragma omp masked
      {
        mip::hive_lns_bridge_t<int, double> worker(
          context, dm.population, cuopt::timer_t(10), failing_worker);
        EXPECT_NO_THROW(worker.finish());
      }
    }
    EXPECT_EQ(worker_ran.load(), team_size >= 9);
    EXPECT_FALSE(dm.population.lns_observer);
  }
  EXPECT_TRUE(dm.population.best_feasible().compute_feasibility());
  EXPECT_EQ(dm.population.best_feasible().get_host_assignment(), before);
  EXPECT_FALSE(context.preempt_heuristic_solver_.load());
  EXPECT_FALSE(dm.population.lns_observer);
}

TEST(HiveLns, ThreadBudgetLeavesCapacityForExistingSolver)
{
  for (int size = 2; size <= 48; ++size) {
    const int workers = mip::lns_worker_count(size, false);
    EXPECT_EQ(workers, size <= 8 ? 0 : (size == 9 ? 1 : 2));
    if (workers) EXPECT_GE(size - workers, 8);
    EXPECT_EQ(mip::lns_worker_count(size, true), 0);
  }
}

TEST(HiveLns, CpufjLnsRevalidatesWithSolverTolerances)
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

std::atomic<bool> early_hive_completed{false};

void polling_then_failing_early_hive(const model_t& model,
                                     const snapshot_fn& snapshot,
                                     const submit_fn& submit,
                                     const stop_fn&,
                                     uint64_t)
{
  EXPECT_TRUE(omp_in_parallel());
  EXPECT_EQ(omp_get_num_threads(), 7);
  EXPECT_EQ(omp_get_max_threads(), 1);
  EXPECT_EQ(model.feasibility_tolerance, 3e-7);
  EXPECT_EQ(model.relative_tolerance, 4e-8);
  EXPECT_EQ(model.integrality_tolerance, 2e-4);
  EXPECT_EQ(model.row_tolerances[0], (mip::get_cstr_tolerance<int, double>(0, 2, 3e-7, 4e-8)));
  auto best = snapshot();
  EXPECT_EQ(best, (population_t{{1.0, 0.0}}));
  submit({.75, 0});
  // The test's producer supplies a newer best through the shared CPUFJ store.
  best = snapshot();
  EXPECT_EQ(best, (population_t{{.5, 0.0}}));
  submit({.25, 0});
  submit({-1, 0});  // Invalid candidate must not reach the callback or shared best.
  early_hive_completed = true;
  throw std::runtime_error("injected presolve LNS failure");
}

TEST(HiveLns, PresolveWorkersPollAndPublishSharedBestAndContainFailure)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op);
  opt::mip_solver_settings_t<int, double> settings;
  settings.tolerances.absolute_tolerance    = 3e-7;
  settings.tolerances.relative_tolerance    = 4e-8;
  settings.tolerances.integrality_tolerance = 2e-4;
  std::atomic<bool> preemption{false};
  auto anchor =
    mip::init_fj_cpu_from_optimization_problem(op, settings.get_tolerances(), preemption);
  auto shared = std::make_shared<mip::fj_cpu_shared_incumbent_t<int, double>>();
  shared->publish(1, 1, {1, 0});
  int reports = 0;
  std::atomic<bool> source_ready{false};
  early_hive_completed = false;
#pragma omp parallel num_threads(7)
  {
#pragma omp masked
    {
      mip::early_lns_t<int, double> workers(
        *anchor,
        shared,
        preemption,
        [&](double objective, const auto& x, const char* origin) {
          ++reports;
          EXPECT_STREQ(origin, "Hive LNS");
          EXPECT_EQ(x[0], objective);
          if (reports == 1) source_ready = true;
        },
        42,
        polling_then_failing_early_hive);
      workers.set_source([&](auto& x) {
        if (!source_ready.exchange(false)) return false;
        x = {.5, 0};
        return true;
      });
      workers.start();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!early_hive_completed.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      EXPECT_NO_THROW(workers.finish());
      EXPECT_NO_THROW(workers.finish());
    }
  }
  EXPECT_TRUE(early_hive_completed.load());
  EXPECT_EQ(reports, 2);
  EXPECT_EQ(shared->objective.load(), .25);
  EXPECT_EQ(shared->assignment, (std::vector<double>{.25, 0}));
  EXPECT_FALSE(preemption.load());
}

TEST(HiveLns, PresolveSearchImprovesAFeasibleCpuIncumbent)
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
#pragma omp parallel num_threads(7)
  {
#pragma omp masked
    {
      mip::early_lns_t<int, double> workers(
        *anchor,
        shared,
        preemption,
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
}

TEST(HiveLns, PresolvePortfolioBudgetAndLifetime)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op, true);
  opt::mip_solver_settings_t<int, double> settings;
  for (int team_size : {2, 6, 7, 10}) {
#pragma omp parallel num_threads(team_size)
    {
#pragma omp masked
      {
        mip::early_cpufj_t<int, double> portfolio(op, settings.get_tolerances(), {}, 42);
        for (int restart = 0; restart < 2; ++restart) {
          // Even an oversized request must honor the presolve reservation.
          portfolio.start(team_size);
          EXPECT_EQ(portfolio.lane_count(), std::max(1, team_size - 4));
          EXPECT_EQ(portfolio.improvement_lane_count(), team_size >= 7 ? 2 : 0);
          EXPECT_NO_THROW(portfolio.stop(/*keep_lns=*/true));
          EXPECT_EQ(portfolio.lane_count(), team_size >= 7 ? 2 : 0);
          if (team_size >= 7) {
            std::atomic<int> polls{0};
            portfolio.set_incumbent_callback({});
            portfolio.set_lns_source([&](auto&) {
              ++polls;
              return false;
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!polls.load() && std::chrono::steady_clock::now() < deadline)
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            EXPECT_GT(polls.load(), 0);
            // Stop before destroying the source callback's captured counter.
            EXPECT_NO_THROW(portfolio.stop());
          }
          EXPECT_NO_THROW(portfolio.stop());
          EXPECT_EQ(portfolio.lane_count(), 0);
          EXPECT_NO_THROW(portfolio.stop());
        }
      }
    }
  }
  // The short initialization probe precedes the OMP team and keeps its single lane.
  mip::early_cpufj_t<int, double> probe(op, settings.get_tolerances(), {}, 42);
  probe.start(20, true);
  EXPECT_EQ(probe.lane_count(), 1);
  EXPECT_EQ(probe.improvement_lane_count(), 0);
  probe.stop();
}

TEST(HiveLns, PersistentBridgeMapsAssignmentsAndRejectsRowViolations)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op);
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<double> lower{-inf, 0}, upper{inf, 1};
  const std::vector<opt::var_t> types{opt::var_t::CONTINUOUS, opt::var_t::INTEGER};
  const double row_lower = -1;
  op.set_variable_lower_bounds(lower.data(), 2);
  op.set_variable_upper_bounds(upper.data(), 2);
  op.set_variable_types(types.data(), 2);
  op.set_constraint_lower_bounds(&row_lower, 1);
  opt::mip_solver_settings_t<int, double> settings;
  settings.tolerances.absolute_tolerance = 3e-7;
  settings.tolerances.relative_tolerance = 4e-8;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();  // Splits the free variable, changing assignment size.
  ASSERT_GT(problem.n_variables, op.get_n_variables());
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();
  mip::persistent_lns_bridge_t<int, double> bridge(problem, dm.population);
  bridge.submit({-1, 1});
  // The LNS seed must be available before the GPU population drains its queue.
  std::vector<double> source;
  ASSERT_TRUE(bridge.snapshot(source));
  EXPECT_EQ(source, (std::vector<double>{-1, 1}));
  EXPECT_FALSE(bridge.snapshot(source));
  dm.population.add_external_solutions_to_population();
  ASSERT_TRUE(dm.population.is_feasible());
  EXPECT_EQ(dm.population.best_feasible().get_objective(), 1);
  EXPECT_FALSE(bridge.snapshot(source));
  bridge.submit({-2, 0});  // Bounds and integrality pass, but the row fails.
  dm.population.add_external_solutions_to_population();
  EXPECT_EQ(dm.population.best_feasible().get_objective(), 1);
  bridge.submit({-1, 0});
  dm.population.add_external_solutions_to_population();
  EXPECT_EQ(dm.population.best_feasible().get_objective(), -1);
  ASSERT_TRUE(bridge.snapshot(source));
  EXPECT_EQ(source, (std::vector<double>{-1, 0}));
}

TEST(HiveLns, PendingSeedsUseSolverTolerancesAndDoNotPoisonValidatedCache)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op);
  const double row_lower = 1;
  op.set_constraint_lower_bounds(&row_lower, 1);
  opt::mip_solver_settings_t<int, double> settings;
  settings.tolerances.absolute_tolerance = 3e-7;
  settings.tolerances.relative_tolerance = 4e-8;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();
  mip::persistent_lns_bridge_t<int, double> bridge(problem, dm.population);
  // Deliberately queue an invalid low objective and a valid incumbent. No callback
  // is registered; queued seeds must still be independently checked by the bridge.
  dm.population.add_external_solution({0, 0}, 0, mip::solution_origin_t::CPUFJ);
  dm.population.add_external_solution({-0.1, 1.1}, -1, mip::solution_origin_t::CPUFJ);
  dm.population.add_external_solution({1 + 1e-6, 0}, 1 + 1e-6, mip::solution_origin_t::CPUFJ);
  std::vector<double> source;
  ASSERT_TRUE(bridge.snapshot(source));
  EXPECT_EQ(source, (std::vector<double>{1, 0}));
  EXPECT_FALSE(bridge.snapshot(source));
  double objective = 0;
  EXPECT_FALSE(dm.population.get_best_feasible_snapshot(source, objective));
  // A row residual within the configured tolerance is an admissible seed.
  dm.population.add_external_solution({1 - 1e-7, 0}, 1 - 1e-7, mip::solution_origin_t::CPUFJ);
  ASSERT_TRUE(bridge.snapshot(source));
  EXPECT_DOUBLE_EQ(source[0], 1 - 1e-7);
  // A larger violation is rejected, without advancing the objective cutoff.
  dm.population.add_external_solution({1 - 1e-4, 0}, 1 - 1e-4, mip::solution_origin_t::CPUFJ);
  EXPECT_FALSE(bridge.snapshot(source));
  dm.population.add_external_solution({1 - 2e-7, 0}, 1 - 2e-7, mip::solution_origin_t::CPUFJ);
  ASSERT_TRUE(bridge.snapshot(source));
  EXPECT_DOUBLE_EQ(source[0], 1 - 2e-7);
}

TEST(HiveLns, RejectedCachedSeedDoesNotHidePendingImprovement)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op);
  const std::vector<opt::var_t> types{opt::var_t::INTEGER, opt::var_t::CONTINUOUS};
  op.set_variable_types(types.data(), 2);
  const std::vector<double> objective{1, -1};
  const double lower = -std::numeric_limits<double>::infinity();
  const double upper = 1;
  op.set_objective_coefficients(objective.data(), 2);
  op.set_constraint_lower_bounds(&lower, 1);
  op.set_constraint_upper_bounds(&upper, 1);
  opt::mip_solver_settings_t<int, double> settings;
  settings.tolerances.integrality_tolerance = 1e-5;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();
  mip::persistent_lns_bridge_t<int, double> bridge(problem, dm.population);
  dm.population.add_external_solution({0, 0}, 0, mip::solution_origin_t::CPUFJ);
  std::vector<double> source;
  ASSERT_TRUE(bridge.snapshot(source));
  // This cached point passes the solver's tolerances, but its objective returns
  // to zero when the private LNS seed is clamped onto the exact domain.
  mip::solution_t<int, double> cached(problem);
  cached.copy_new_assignment(std::vector<double>{-1e-6, 0});
  ASSERT_TRUE(cached.compute_feasibility());
  ASSERT_LE(cached.compute_max_variable_violation(), settings.tolerances.integrality_tolerance);
  dm.population.add_solution(std::move(cached));
  dm.population.add_external_solution({0, 5e-7}, -5e-7, mip::solution_origin_t::CPUFJ);
  ASSERT_TRUE(bridge.snapshot(source));
  EXPECT_EQ(source, (std::vector<double>{0, 5e-7}));
  EXPECT_FALSE(bridge.snapshot(source));
  // Consuming a cache entry for LNS does not alter the population's incumbent.
  double cached_objective;
  ASSERT_TRUE(dm.population.get_best_feasible_snapshot(source, cached_objective));
  EXPECT_EQ(source, (std::vector<double>{-1e-6, 0}));
  EXPECT_DOUBLE_EQ(cached_objective, -1e-6);
}

TEST(HiveLns, PendingSeedQueueRetainsBestUnderWeakerProducerTraffic)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op);
  opt::mip_solver_settings_t<int, double> settings;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();
  mip::persistent_lns_bridge_t<int, double> bridge(problem, dm.population);
  dm.population.add_external_solution({0.1, 0}, 0.1, mip::solution_origin_t::CPUFJ);
  for (int i = 0; i < 10; ++i) {
    const double x = 0.2 + 0.05 * i;
    dm.population.add_external_solution({x, 0}, x, mip::solution_origin_t::CPUFJ);
  }
  std::vector<double> source;
  ASSERT_TRUE(bridge.snapshot(source));
  EXPECT_EQ(source, (std::vector<double>{0.1, 0}));
  EXPECT_FALSE(bridge.snapshot(source));
}

TEST(HiveLns, ExternalQueueKeepsGlobalBestFiftyAcrossOrigins)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op);
  // Population hashing needs an integer column; keep x continuous for objective markers.
  const std::vector<opt::var_t> types{opt::var_t::CONTINUOUS, opt::var_t::INTEGER};
  op.set_variable_types(types.data(), 2);
  opt::mip_solver_settings_t<int, double> settings;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();

  const std::vector<mip::solution_origin_t> origins{mip::solution_origin_t::CPUFJ,
                                                    mip::solution_origin_t::BRANCH_AND_BOUND,
                                                    mip::solution_origin_t::EXTERNAL};
  // Reuse the queue after each drain, with ascending, descending, and permuted arrivals.
  for (int order = 0; order < 3; ++order) {
    for (int i = 0; i < 100; ++i) {
      const int index    = order == 0 ? i : (order == 1 ? 99 - i : (i * 37) % 100);
      const double value = static_cast<double>(index) / 128;
      dm.population.add_external_solution({value, 0}, value, origins[index % origins.size()]);
      EXPECT_EQ(dm.population.get_external_solution_size(), std::min(i + 1, 50));
    }
    EXPECT_TRUE(dm.population.solutions_in_external_queue_.load());
    auto candidates = dm.population.get_external_solutions();
    ASSERT_EQ(candidates.size(), 50);
    // The retained set includes 17 CPUFJ entries, with no separate per-origin cap.
    for (size_t i = 0; i < candidates.size(); ++i) {
      EXPECT_TRUE(candidates[i].get_feasible());
      EXPECT_EQ(candidates[i].get_host_assignment(),
                (std::vector<double>{static_cast<double>(i) / 128, 0}));
    }
    EXPECT_EQ(dm.population.get_external_solution_size(), 0);
    EXPECT_FALSE(dm.population.solutions_in_external_queue_.load());
    EXPECT_TRUE(dm.population.get_external_solutions().empty());
  }
}

TEST(HiveLns, ExternalQueueRejectsNonfiniteObjectivesAndNonimprovingOverflow)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op);
  // Population hashing needs an integer column; keep x continuous for objective markers.
  const std::vector<opt::var_t> types{opt::var_t::CONTINUOUS, opt::var_t::INTEGER};
  op.set_variable_types(types.data(), 2);
  opt::mip_solver_settings_t<int, double> settings;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();

  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<double> nonfinite{std::numeric_limits<double>::quiet_NaN(), inf, -inf};
  for (double objective : nonfinite) {
    dm.population.add_external_solution({0.75, 0}, objective, mip::solution_origin_t::CPUFJ);
    EXPECT_EQ(dm.population.get_external_solution_size(), 0);
  }
  EXPECT_FALSE(dm.population.solutions_in_external_queue_.load());
  for (int i = 0; i < 50; ++i) {
    const double value = static_cast<double>(i) / 128;
    dm.population.add_external_solution({value, 0}, value, mip::solution_origin_t::EXTERNAL);
  }
  for (int i = 0; i < 100; ++i) {
    for (double objective : nonfinite) {
      dm.population.add_external_solution(
        {0.75, 0}, objective, mip::solution_origin_t::BRANCH_AND_BOUND);
    }
    // Distinct assignments expose accidental replacement on a tied objective.
    dm.population.add_external_solution({0.75, 0}, 49.0 / 128, mip::solution_origin_t::EXTERNAL);
    dm.population.add_external_solution({0.75, 0}, 0.75, mip::solution_origin_t::EXTERNAL);
    EXPECT_EQ(dm.population.get_external_solution_size(), 50);
  }
  auto candidates = dm.population.get_external_solutions();
  ASSERT_EQ(candidates.size(), 50);
  for (size_t i = 0; i < candidates.size(); ++i) {
    EXPECT_TRUE(candidates[i].get_feasible());
    EXPECT_EQ(candidates[i].get_host_assignment(),
              (std::vector<double>{static_cast<double>(i) / 128, 0}));
  }
  EXPECT_EQ(dm.population.get_external_solution_size(), 0);
}

TEST(HiveLns, ExternalQueueValidatesWithSolverTolerancesBeforeReplacingIncumbent)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op);
  // Population hashing needs an integer column; keep x continuous for objective markers.
  const std::vector<opt::var_t> types{opt::var_t::CONTINUOUS, opt::var_t::INTEGER};
  op.set_variable_types(types.data(), 2);
  const double row_lower = 1;
  op.set_constraint_lower_bounds(&row_lower, 1);
  opt::mip_solver_settings_t<int, double> settings;
  settings.tolerances.absolute_tolerance = 3e-7;
  settings.tolerances.relative_tolerance = 4e-8;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();

  dm.population.add_external_solution({1, 1}, 3, mip::solution_origin_t::BRANCH_AND_BOUND);
  dm.population.add_external_solutions_to_population();
  ASSERT_TRUE(dm.population.is_feasible());
  ASSERT_EQ(dm.population.best_feasible().get_objective(), 3);

  // Ranking uses reported objectives, but those reports cannot replace a validated incumbent.
  for (int i = 0; i < 50; ++i) {
    dm.population.add_external_solution({0, 0}, -1e9, mip::solution_origin_t::BRANCH_AND_BOUND);
  }
  dm.population.add_external_solution({1, 0}, 1, mip::solution_origin_t::BRANCH_AND_BOUND);
  dm.population.add_external_solutions_to_population();
  EXPECT_EQ(dm.population.get_external_solution_size(), 0);
  EXPECT_TRUE(dm.population.best_feasible().compute_feasibility());
  EXPECT_EQ(dm.population.best_feasible().get_objective(), 3);

  const double tolerance = mip::get_cstr_tolerance<int, double>(
    row_lower, 2, settings.tolerances.absolute_tolerance, settings.tolerances.relative_tolerance);
  const double outside = 1 - 2 * tolerance;
  const double inside  = 1 - 0.5 * tolerance;
  dm.population.add_external_solution({outside, 0}, outside, mip::solution_origin_t::EXTERNAL);
  dm.population.add_external_solution({inside, 0}, inside, mip::solution_origin_t::EXTERNAL);
  {
    auto candidates = dm.population.get_external_solutions();
    ASSERT_EQ(candidates.size(), 2);
    EXPECT_FALSE(candidates[0].get_feasible());
    EXPECT_TRUE(candidates[1].get_feasible());
  }
  dm.population.add_external_solution({inside, 0}, inside, mip::solution_origin_t::EXTERNAL);
  dm.population.add_external_solutions_to_population();
  EXPECT_EQ(dm.population.get_external_solution_size(), 0);
  EXPECT_TRUE(dm.population.best_feasible().compute_feasibility());
  EXPECT_EQ(dm.population.best_feasible().get_objective(), inside);
  EXPECT_EQ(dm.population.best_feasible().get_host_assignment(), (std::vector<double>{inside, 0}));
}

TEST(HiveLns, ExternalQueueDrainAllowsReentrantProducerAndLeavesNewHeapPending)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_early_lns_test_problem(op);
  // Population hashing needs an integer column; keep x continuous for objective markers.
  const std::vector<opt::var_t> types{opt::var_t::CONTINUOUS, opt::var_t::INTEGER};
  op.set_variable_types(types.data(), 2);
  opt::mip_solver_settings_t<int, double> settings;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();

  bool produced                     = false;
  problem.branch_and_bound_callback = [&](const auto&, auto) {
    if (produced) return true;
    produced = true;
    EXPECT_EQ(dm.population.get_external_solution_size(), 0);
    for (int i = 60; i > 0; --i) {
      const double value = static_cast<double>(i) / 128;
      dm.population.add_external_solution({value, 0}, value, mip::solution_origin_t::EXTERNAL);
    }
    return true;
  };
  dm.population.add_external_solution({0.75, 0}, 0.75, mip::solution_origin_t::EXTERNAL);
  dm.population.add_external_solutions_to_population();
  ASSERT_TRUE(produced);
  // The swapped heap is drained without the producer lock; callback traffic waits for next time.
  EXPECT_EQ(dm.population.best_feasible().get_objective(), 0.75);
  EXPECT_EQ(dm.population.get_external_solution_size(), 50);
  EXPECT_TRUE(dm.population.solutions_in_external_queue_.load());

  problem.branch_and_bound_callback = {};
  dm.population.add_external_solutions_to_population();
  EXPECT_EQ(dm.population.get_external_solution_size(), 0);
  EXPECT_FALSE(dm.population.solutions_in_external_queue_.load());
  EXPECT_EQ(dm.population.best_feasible().get_objective(), 1.0 / 128);
}

TEST(HiveLns, BenchmarkApiErrorsAreDistinctFromNoSolution)
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
}  // namespace cuopt::hive_lns::test
