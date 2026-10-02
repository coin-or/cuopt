/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mip_heuristics/feasibility_jump/cpu/climber.hpp>
#include <mip_heuristics/lns/cpufj.cuh>
#include <mip_heuristics/lns/cpufj_geometry.cuh>

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

TEST(CpuFjLnsWrapper, LocalCrossingResetKeepsValidatedArchive)
{
  std::atomic<bool> stop{false};
  auto climber                       = equality_climber(stop);
  climber->settings.iteration_limit  = 8;
  climber->h_best_assignment         = std::vector<double>{0.0, 2.0};
  climber->h_best_objective          = 2.0;
  climber->h_last_reported_objective = std::numeric_limits<double>::max();
  climber->feasible_found            = true;
  climber->h_assignment              = std::vector<double>{0.0, 0.0};
  int local_crossings                = 0;
  int accepted_improvements          = 0;
  climber->improvement_callback      = [&](double, const auto& assignment, double) {
    EXPECT_TRUE(
      verify_cpufj_lns_feasible(*climber->problem, climber->h_var_bounds.underlying(), assignment));
    ++local_crossings;
    const double objective = 2.0 * assignment[0] + assignment[1];
    if (objective < 2.0) ++accepted_improvements;
  };
  // The new solve must recognize its first feasible crossing even though the
  // external archive is already optimal. Its consumer still accepts only gains.
  EXPECT_FALSE(repair_cpufj_lns_neighborhood(climber.get(), 1.0, true));
  EXPECT_GT(local_crossings, 0);
  EXPECT_EQ(accepted_improvements, 0);
  EXPECT_TRUE(climber->feasible_found);
  EXPECT_EQ(climber->h_best_objective, 2.0);
  EXPECT_EQ(climber->h_best_assignment.underlying(), (std::vector<double>{0.0, 2.0}));

  // A worse solve-local feasible start must also leave the archive intact.
  climber->settings.iteration_limit = climber->iterations;
  climber->h_assignment             = std::vector<double>{2.0, 0.0};
  EXPECT_FALSE(repair_cpufj_lns_neighborhood(climber.get(), 1.0, true));
  EXPECT_TRUE(climber->feasible_found);
  EXPECT_EQ(climber->h_best_objective, 2.0);
  EXPECT_EQ(climber->h_best_assignment.underlying(), (std::vector<double>{0.0, 2.0}));
}

TEST(CpuFjLnsWrapper, PreemptedLocalCrossingResetRetainsArchive)
{
  for (const bool preempt : {false, true}) {
    SCOPED_TRACE(preempt);
    std::atomic<bool> stop{preempt};
    auto climber                  = equality_climber(stop);
    climber->halted               = !preempt;
    climber->h_best_assignment    = std::vector<double>{0.0, 2.0};
    climber->h_best_objective     = 2.0;
    climber->feasible_found       = true;
    climber->h_assignment         = std::vector<double>{0.0, 0.0};
    int reports                   = 0;
    climber->improvement_callback = [&](double, const auto&, double) { ++reports; };
    EXPECT_FALSE(repair_cpufj_lns_neighborhood(climber.get(), 1.0, true));
    EXPECT_EQ(reports, 0);
    EXPECT_EQ(stop.load(), preempt);
    EXPECT_EQ(climber->halted.load(), !preempt);
    EXPECT_TRUE(climber->feasible_found);
    EXPECT_EQ(climber->h_best_objective, 2.0);
    EXPECT_EQ(climber->h_best_assignment.underlying(), (std::vector<double>{0.0, 2.0}));
  }
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

namespace {
struct lns_geometry_test_options_t {
  double big_m{2};
  double row_scale{1};
  double inactive_row_shift{0};
  bool upper_rows{false};
  bool unbounded{false};
  bool integer_objective{false};
  int groups{8};
  int scope_size{4};
  mip_solver_settings_t<int, double>::tolerances_t tolerances;
};

std::unique_ptr<fj_cpu_climber_t<int, double>> geometry_climber(
  std::atomic<bool>& stop, const lns_geometry_test_options_t& options = {})
{
  const int variables = 4 + 2 * options.groups;
  std::vector<double> coefficients, row_lower, row_upper;
  std::vector<int> columns, offsets{0};
  for (int group = 0; group < options.groups; ++group) {
    for (int selector = 0; selector < 2; ++selector) {
      const double direction = (selector == 0 ? 1.0 : -1.0) * options.row_scale;
      const double sense     = options.upper_rows ? -1.0 : 1.0;
      for (int pair = 0; pair < 2; ++pair) {
        columns.insert(columns.end(),
                       {2 * pair,
                        pair == 1 && options.scope_size == 3 ? 0 : 2 * pair + 1,
                        4 + 2 * group + selector});
        coefficients.insert(
          coefficients.end(),
          {sense * direction, -sense * direction, -sense * options.big_m * options.row_scale});
        const double rhs = (1.0 - options.big_m) * options.row_scale + options.inactive_row_shift;
        row_lower.push_back(options.upper_rows ? -std::numeric_limits<double>::infinity() : rhs);
        row_upper.push_back(options.upper_rows ? -rhs : std::numeric_limits<double>::infinity());
        offsets.push_back(columns.size());
      }
    }
    columns.insert(columns.end(), {4 + 2 * group, 5 + 2 * group});
    coefficients.insert(coefficients.end(), {1.0, 1.0});
    row_lower.push_back(1.0);
    row_upper.push_back(1.0);
    offsets.push_back(columns.size());
  }
  std::vector<double> objective(variables, 0.0), lower(variables, 0.0), upper(variables, 1.0);
  std::vector<var_t> types(variables, var_t::INTEGER);
  for (int variable = 0; variable < 4; ++variable) {
    objective[variable] = 1.0;
    types[variable]     = var_t::CONTINUOUS;
  }
  if (options.integer_objective) objective[4] = 1.0;
  if (options.unbounded) upper[1] = std::numeric_limits<double>::infinity();
  fj_settings_t settings;
  settings.seed = 42;
  auto climber  = init_fj_cpu_from_host_model<int, double>(variables,
                                                          row_lower.size(),
                                                          coefficients.size(),
                                                          false,
                                                          1.0,
                                                          0.0,
                                                          coefficients,
                                                          columns,
                                                          offsets,
                                                          objective,
                                                          lower,
                                                          upper,
                                                          row_lower,
                                                          row_upper,
                                                           {},
                                                           {},
                                                          types,
                                                          options.tolerances,
                                                          stop,
                                                          settings);
  EXPECT_EQ(climber->n_binary_vars, 2 * options.groups);
  EXPECT_EQ(climber->n_integer_vars, 0);
  return climber;
}
}  // namespace

TEST(CpuFjLnsGeometry, RecognizesOriginalStrengthenedAndScaledDisjunctions)
{
  for (const double big_m : {2000.0, 2.0}) {
    for (const double row_scale : {0.001, 1.0, 1000.0}) {
      for (const bool upper_rows : {false, true}) {
        SCOPED_TRACE(big_m);
        SCOPED_TRACE(row_scale);
        SCOPED_TRACE(upper_rows);
        std::atomic<bool> stop{false};
        lns_geometry_test_options_t options;
        options.big_m               = big_m;
        options.row_scale           = row_scale;
        options.upper_rows          = upper_rows;
        auto climber                = geometry_climber(stop, options);
        const auto assignment       = climber->h_assignment.underlying();
        const auto best             = climber->h_best_assignment.underlying();
        const auto shared_problem   = climber->problem;
        const auto perturb_interval = climber->perturb_interval;
        EXPECT_TRUE(configure_cpufj_lns_geometry(*climber));
        EXPECT_EQ(climber->perturb_interval, perturb_interval);
        EXPECT_EQ(climber->problem, shared_problem);
        EXPECT_EQ(climber->h_assignment.underlying(), assignment);
        EXPECT_EQ(climber->h_best_assignment.underlying(), best);
        EXPECT_TRUE(climber->low_latency);
        EXPECT_TRUE(climber->objective_directed_perturb);
        EXPECT_DOUBLE_EQ(climber->continuous_perturb_fraction, 0.2);
        EXPECT_DOUBLE_EQ(climber->objective_weight_floor, 1.0);
        EXPECT_TRUE(climber->use_cardinality_exchange);
        EXPECT_FALSE(climber->use_fixed_charge_network_start);
        EXPECT_FALSE(climber->use_lp_start);
        EXPECT_FALSE(climber->use_deep_lp_pump);
        EXPECT_FALSE(climber->use_lp_polish);
        EXPECT_FALSE(climber->use_precedence_start);
        EXPECT_FALSE(climber->use_affine_equality_start);
        EXPECT_FALSE(climber->use_unit_commitment_start);
        EXPECT_FALSE(climber->use_pmedian_start);
        EXPECT_FALSE(climber->use_equality_substitution);
        EXPECT_FALSE(climber->use_bound_prop);
        EXPECT_FALSE(climber->use_integer_bit_encoding);
      }
    }
  }
}

TEST(CpuFjLnsGeometry, PreservesPrivateRepairCadence)
{
  std::atomic<bool> stop{false};
  auto climber              = geometry_climber(stop);
  climber->perturb_interval = 37;
  const auto seed           = climber->settings.seed;
  EXPECT_TRUE(configure_cpufj_lns_geometry(*climber));
  EXPECT_EQ(climber->perturb_interval, 37);
  EXPECT_EQ(climber->settings.seed, seed);
}

TEST(CpuFjLnsGeometry, InactiveRowProofUsesConfiguredTolerances)
{
  for (const bool upper_rows : {false, true}) {
    std::atomic<bool> stop{false};
    lns_geometry_test_options_t options;
    options.upper_rows                    = upper_rows;
    options.row_scale                     = 1e6;
    options.tolerances.absolute_tolerance = 1e-6;
    options.tolerances.relative_tolerance = 1e-8;
    // The row tolerance is approximately .01: the proof must retain the same
    // tolerance after presolve rather than require exact cancellation of bounds.
    options.inactive_row_shift = 0.005;
    auto within_tolerance      = geometry_climber(stop, options);
    EXPECT_TRUE(has_cpufj_lns_geometry(*within_tolerance));
    options.inactive_row_shift = 0.03;
    auto outside_tolerance     = geometry_climber(stop, options);
    EXPECT_FALSE(configure_cpufj_lns_geometry(*outside_tolerance));
    EXPECT_FALSE(outside_tolerance->low_latency);
    EXPECT_DOUBLE_EQ(outside_tolerance->continuous_perturb_fraction, 0.0);
  }
}

TEST(CpuFjLnsGeometry, RejectsIncompleteStructureAndUnprovedInactiveRows)
{
  std::atomic<bool> stop{false};
  for (int rejected_case = 0; rejected_case < 4; ++rejected_case) {
    SCOPED_TRACE(rejected_case);
    lns_geometry_test_options_t options;
    if (rejected_case == 0) options.groups = 7;
    if (rejected_case == 1) options.scope_size = 3;
    if (rejected_case == 2) options.integer_objective = true;
    if (rejected_case == 3) options.unbounded = true;
    auto climber = geometry_climber(stop, options);
    EXPECT_FALSE(configure_cpufj_lns_geometry(*climber));
    EXPECT_DOUBLE_EQ(climber->continuous_perturb_fraction, 0.0);
  }
  // Unbounded continuous domains remain eligible through the original large-M
  // certificate; only the new inactive-bound proof requires finite endpoints.
  lns_geometry_test_options_t original;
  original.big_m     = 2000;
  original.unbounded = true;
  auto climber       = geometry_climber(stop, original);
  EXPECT_TRUE(has_cpufj_lns_geometry(*climber));
}
}  // namespace cuopt::mathematical_optimization::mip::test
