/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/feasibility_jump/cpu/climber.hpp>
#include <mip_heuristics/utils.cuh>
#include <mip_heuristics/utils.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Bound strengthening can make a valid disjunction's M coefficient small. Certify
// that its inactive (binary-zero) row is already implied by the continuous bounds.
template <typename i_t, typename f_t>
bool cpufj_lns_inactive_row_redundant(const fj_cpu_climber_t<i_t, f_t>& climber,
                                      i_t row,
                                      i_t gate,
                                      bool lower_row,
                                      std::vector<f_t>& coefficients,
                                      std::vector<f_t>& endpoints)
{
  const auto& problem = *climber.problem;
  coefficients.clear();
  endpoints.clear();
  for (i_t q = problem.offsets[row]; q < problem.offsets[row + 1]; ++q) {
    const i_t variable    = problem.variables[q];
    const f_t coefficient = problem.coefficients[q];
    if (variable == gate || coefficient == f_t{0}) continue;
    const auto bounds = climber.h_var_bounds[variable];
    const f_t endpoint =
      lower_row == (coefficient > f_t{0}) ? get_lower(bounds) : get_upper(bounds);
    if (!std::isfinite(endpoint) || !std::isfinite(coefficient * endpoint)) return false;
    coefficients.push_back(coefficient);
    endpoints.push_back(endpoint);
  }
  const f_t activity = compensated_dot2(coefficients.data(), endpoints.data(), coefficients.size());
  const f_t lower = problem.cstr_lb[row], upper = problem.cstr_ub[row];
  const f_t tolerance = get_cstr_tolerance<i_t, f_t>(
    lower, upper, problem.tolerances.absolute_tolerance, problem.tolerances.relative_tolerance);
  return std::isfinite(activity) &&
         (lower_row ? activity >= lower - tolerance : activity <= upper + tolerance);
}

// The CPUFJ geometric portfolio certificate, extended only for this private LNS
// worker to accept bound-strengthened inactive rows. The shared model is read-only.
template <typename i_t, typename f_t>
bool has_cpufj_lns_geometry(const fj_cpu_climber_t<i_t, f_t>& climber)
{
  const auto& problem = *climber.problem;
  const i_t groups    = static_cast<i_t>(problem.card_cardinalities.size());
  if (climber.n_binary_vars <= 0 || climber.n_integer_vars != 0 || groups < 8 ||
      problem.h_objective_vars.empty())
    return false;
  i_t covered = 0;
  for (i_t group : problem.card_group_of_variable)
    covered += group >= 0;
  if (covered < static_cast<i_t>(0.9 * climber.n_binary_vars)) return false;
  for (i_t variable : problem.h_objective_vars)
    if (problem.h_var_types[variable] != var_t::CONTINUOUS) return false;
  for (i_t cardinality : problem.card_cardinalities)
    if (cardinality != 1) return false;
  for (i_t variable : climber.h_binary_indices)
    if (problem.card_group_of_variable[variable] < 0) return false;

  std::vector<std::array<i_t, 4>> scopes(groups);
  std::vector<i_t> scope_sizes(groups, 0);
  std::vector<f_t> coefficients, endpoints;
  i_t gated = 0, equalities = 0;
  for (i_t row = 0; row < problem.n_constraints; ++row) {
    const f_t lower = problem.cstr_lb[row], upper = problem.cstr_ub[row];
    if (lower == upper) {
      ++equalities;
      continue;
    }
    i_t binary           = -1;
    f_t gate_coefficient = 0, continuous_max = 0;
    for (i_t q = problem.offsets[row]; q < problem.offsets[row + 1]; ++q) {
      const i_t variable = problem.variables[q];
      if (climber.h_is_binary_variable[variable]) {
        if (binary >= 0) return false;
        binary           = variable;
        gate_coefficient = problem.coefficients[q];
      } else {
        continuous_max = std::max(continuous_max, std::abs(problem.coefficients[q]));
      }
    }
    if (binary < 0) continue;
    const i_t group      = problem.card_group_of_variable[binary];
    const bool lower_row = gate_coefficient < 0 && std::isfinite(lower) && !std::isfinite(upper);
    const bool upper_row = gate_coefficient > 0 && std::isfinite(upper) && !std::isfinite(lower);
    if (group < 0 || (!lower_row && !upper_row) || continuous_max <= 0) return false;
    if (std::abs(gate_coefficient) < f_t{1000} * continuous_max &&
        !cpufj_lns_inactive_row_redundant(climber, row, binary, lower_row, coefficients, endpoints))
      return false;
    ++gated;
    auto& scope = scopes[group];
    auto& size  = scope_sizes[group];
    for (i_t q = problem.offsets[row]; q < problem.offsets[row + 1]; ++q) {
      const i_t variable = problem.variables[q];
      if (variable == binary ||
          std::find(scope.begin(), scope.begin() + size, variable) != scope.begin() + size)
        continue;
      if (size == 4) return false;
      scope[size++] = variable;
    }
  }
  return equalities == groups && gated >= 0.8 * problem.n_constraints &&
         std::all_of(scope_sizes.begin(), scope_sizes.end(), [](i_t size) { return size == 4; });
}

template <typename i_t, typename f_t>
bool configure_cpufj_lns_geometry(fj_cpu_climber_t<i_t, f_t>& climber)
{
  if (!has_cpufj_lns_geometry(climber)) return false;
  // Lane seven selects existing scalar/cardinality repair without constructive
  // starts. Keep the incumbent as the start and use local continuous perturbations.
  const i_t perturb_interval = climber.perturb_interval;
  climber.low_latency        = true;
  apply_lane_diversification(climber, 7, climber.settings.seed);
  // Short LNS bursts restart the stall counter. Keep their existing repair
  // cadence instead of the long-running portfolio lane's randomized interval.
  climber.perturb_interval               = perturb_interval;
  climber.use_fixed_charge_network_start = false;
  climber.continuous_perturb_fraction    = f_t{0.2};
  climber.objective_directed_perturb     = true;
  return true;
}

}  // namespace cuopt::mathematical_optimization::mip
