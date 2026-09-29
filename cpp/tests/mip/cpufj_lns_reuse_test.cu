/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mip_heuristics/feasibility_jump/cpu/climber.hpp>
#include <mip_heuristics/local_search/cpufj_lns.cuh>

#include <gtest/gtest.h>

#include <atomic>
#include <limits>

namespace cuopt::mathematical_optimization::mip::test {
namespace {
std::unique_ptr<fj_cpu_climber_t<int, double>> equality_climber(std::atomic<bool>& stop)
{
  mip_solver_settings_t<int, double>::tolerances_t tolerances;
  fj_settings_t settings;
  settings.seed = 42;
  // The equality creates two search rows from one model row, exposing a stale
  // model-row recomputation when the scalar engine restarts.
  auto climber = init_fj_cpu_from_host_model<int, double>(2,
                                                          1,
                                                          2,
                                                          false,
                                                          1.0,
                                                          0.0,
                                                          {1.0, 1.0},
                                                          {0, 1},
                                                          {0, 2},
                                                          {2.0, 1.0},
                                                          {0.0, 0.0},
                                                          {2.0, 2.0},
                                                          {2.0},
                                                          {2.0},
                                                          {},
                                                          {},
                                                          {var_t::INTEGER, var_t::CONTINUOUS},
                                                          tolerances,
                                                          stop,
                                                          settings);
  // Continuous variables can be eliminated by the binary adapter. Select the
  // scalar engine explicitly so this fixture exercises repeated row conversion.
  climber->low_latency = true;
  return climber;
}
}  // namespace

TEST(CpuFjLnsWrapper, RepeatedBurstsRestoreModelRowContractAndKeepIncumbent)
{
  std::atomic<bool> stop{false};
  auto climber                      = equality_climber(stop);
  climber->settings.iteration_limit = 0;
  climber->h_best_assignment        = std::vector<double>{0.0, 2.0};
  climber->h_best_objective         = 2.0;
  climber->feasible_found           = true;
  // A feasible but worse perturbation must not replace the known incumbent.
  climber->h_assignment = std::vector<double>{2.0, 0.0};
  EXPECT_FALSE(repair_cpufj_lns_neighborhood(climber.get(), 1.0));
  ASSERT_EQ(climber->n_rows, 2);
  ASSERT_TRUE(climber->h_initial_left_weights.empty());
  ASSERT_TRUE(climber->h_initial_right_weights.empty());
  EXPECT_EQ(climber->h_best_objective, 2.0);
  EXPECT_EQ(climber->h_best_assignment.underlying(), (std::vector<double>{0.0, 2.0}));

  // This burst enters with two search rows and released setup weights, but must
  // rebuild from one model row with the LNS climber's initial unit weights.
  // A ruined infeasible seed must not overwrite the archived feasible point.
  climber->h_assignment = std::vector<double>{0.0, 0.0};
  EXPECT_FALSE(repair_cpufj_lns_neighborhood(climber.get(), 1.0));
  EXPECT_EQ(climber->violated_constraints.max_size(), 2);
  EXPECT_EQ(climber->violated_constraints.size(), 1);
  EXPECT_EQ(climber->row_state()[0].slack, -2.0);
  EXPECT_EQ(climber->row_state()[1].slack, 2.0);
  EXPECT_EQ(climber->row_state()[0].weight, 1.0);
  EXPECT_EQ(climber->row_state()[1].weight, 1.0);
  EXPECT_TRUE(climber->h_initial_left_weights.empty());
  EXPECT_TRUE(climber->h_initial_right_weights.empty());
  EXPECT_EQ(climber->h_incumbent_objective, 0.0);

  climber->h_assignment = std::vector<double>{1.0, 1.0};
  EXPECT_FALSE(repair_cpufj_lns_neighborhood(climber.get(), 1.0));
  EXPECT_TRUE(climber->violated_constraints.empty());
  EXPECT_EQ(climber->h_incumbent_objective, 3.0);
  EXPECT_EQ(climber->h_best_objective, 2.0);

  // A fresh, valid better burst must still replace the archive after repeated resets.
  climber->h_best_assignment = std::vector<double>{2.0, 0.0};
  climber->h_best_objective  = 4.0;
  climber->h_assignment      = std::vector<double>{0.0, 2.0};
  EXPECT_TRUE(repair_cpufj_lns_neighborhood(climber.get(), 1.0));
  EXPECT_EQ(climber->n_rows, 2);
  EXPECT_EQ(climber->h_best_objective, 2.0);
  EXPECT_EQ(climber->h_best_assignment.underlying(), (std::vector<double>{0.0, 2.0}));
}

TEST(CpuFjLnsWrapper, RuinRepairImprovesAndSurvivesSeveralEqualityBursts)
{
  std::atomic<bool> stop{false};
  auto climber = equality_climber(stop);
  std::vector<double> incumbent{2.0, 0.0};
  double incumbent_objective    = 4.0;
  int reports                   = 0;
  int polls                     = 0;
  climber->improvement_callback = [&](double, const auto& assignment, double) {
    ASSERT_TRUE(
      verify_cpufj_lns_feasible(*climber->problem, climber->h_var_bounds.underlying(), assignment));
    const double objective = 2.0 * assignment[0] + assignment[1];
    if (objective < incumbent_objective) {
      incumbent           = assignment;
      incumbent_objective = objective;
      ++reports;
    }
  };
  run_cpufj_lns_ruin_repair<int, double>(climber.get(), [&](auto& assignment, auto& objective) {
    if (++polls > 3) {
      stop = true;
      return false;
    }
    assignment = incumbent;
    objective  = incumbent_objective;
    return true;
  });
  EXPECT_EQ(polls, 4);
  EXPECT_EQ(climber->n_rows, 2);
  EXPECT_GT(reports, 0);
  EXPECT_EQ(incumbent_objective, 2.0);
  EXPECT_EQ(incumbent, (std::vector<double>{0.0, 2.0}));
}
TEST(CpuFjLnsWrapper, BinaryBurstsKeepBetterIncumbentsAndAcceptImprovements)
{
  // Cover both native binary variables and general integers encoded as bits.
  for (const double upper : {1.0, 2.0}) {
    SCOPED_TRACE(upper);
    std::atomic<bool> stop{false};
    mip_solver_settings_t<int, double>::tolerances_t tolerances;
    fj_settings_t settings;
    settings.seed = 42;
    auto climber =
      init_fj_cpu_from_host_model<int, double>(2,
                                               1,
                                               2,
                                               false,
                                               1.0,
                                               0.0,
                                               {1.0, 1.0},
                                               {0, 1},
                                               {0, 2},
                                               {2.0, 1.0},
                                               {0.0, 0.0},
                                               {upper, upper},
                                               {upper},
                                               {std::numeric_limits<double>::infinity()},
                                               {},
                                               {},
                                               {var_t::INTEGER, var_t::INTEGER},
                                               tolerances,
                                               stop,
                                               settings);
    climber->settings.iteration_limit  = 0;
    climber->h_best_assignment         = std::vector<double>{0.0, upper};
    climber->h_best_objective          = upper;
    climber->h_last_reported_objective = std::numeric_limits<double>::max();
    climber->feasible_found            = true;
    int reports                        = 0;
    double accepted_objective          = upper;
    climber->improvement_callback      = [&](double, const auto& assignment, double) {
      EXPECT_TRUE(verify_cpufj_lns_feasible(
        *climber->problem, climber->h_var_bounds.underlying(), assignment));
      // The unchanged CPUFJ engine may report a worse-than-archive point during
      // a burst. Only the consumer's validated improvements count as accepted.
      const double objective = 2.0 * assignment[0] + assignment[1];
      if (objective < accepted_objective) {
        accepted_objective = objective;
        ++reports;
      }
    };

    for (const auto& ruined : {std::vector<double>{upper, upper}, std::vector<double>{0.0, 0.0}}) {
      climber->h_assignment = ruined;
      EXPECT_FALSE(repair_cpufj_lns_neighborhood(climber.get(), 1.0));
      EXPECT_EQ(climber->n_rows, 0);  // The scalar engine must not handle this fixture.
      EXPECT_TRUE(climber->h_initial_left_weights.empty());
      EXPECT_TRUE(climber->h_initial_right_weights.empty());
      EXPECT_EQ(climber->h_best_objective, upper);
      EXPECT_EQ(climber->h_best_assignment.underlying(), (std::vector<double>{0.0, upper}));
      EXPECT_EQ(reports, 0);
    }

    // A genuinely improving feasible start still updates the host incumbent and callback.
    climber->h_best_assignment         = std::vector<double>{upper, upper};
    climber->h_best_objective          = 3.0 * upper;
    climber->h_last_reported_objective = 3.0 * upper;
    climber->h_assignment              = std::vector<double>{upper, 0.0};
    accepted_objective                 = 3.0 * upper;
    EXPECT_TRUE(repair_cpufj_lns_neighborhood(climber.get(), 1.0));
    EXPECT_EQ(climber->h_best_objective, 2.0 * upper);
    EXPECT_EQ(climber->h_best_assignment.underlying(), (std::vector<double>{upper, 0.0}));
    EXPECT_EQ(reports, 1);
    EXPECT_EQ(accepted_objective, 2.0 * upper);
  }
}
}  // namespace cuopt::mathematical_optimization::mip::test
