/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Recognize disjoint exact-one regions with four continuous coordinates per group.
// Callers select eligible models and may certify an activating row with a small M.
template <typename i_t, typename f_t, typename certify_small_gate_t>
bool has_cpufj_disjunctive_geometry(const fj_cpu_climber_t<i_t, f_t>& climber,
                                    certify_small_gate_t certify_small_gate)
{
  const auto& problem = *climber.problem;
  const i_t groups    = static_cast<i_t>(problem.card_cardinalities.size());
  for (i_t cardinality : problem.card_cardinalities)
    if (cardinality != 1) return false;
  for (i_t variable : climber.h_binary_indices)
    if (problem.card_group_of_variable[variable] < 0) return false;

  std::vector<std::array<i_t, 4>> scopes(groups);
  std::vector<i_t> scope_sizes(groups, 0);
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
    if (group < 0 || (!lower_row && !upper_row) || !(continuous_max > 0)) return false;
    if (!(std::abs(gate_coefficient) >= f_t{1000} * continuous_max) &&
        !certify_small_gate(row, binary, lower_row))
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
bool has_cpufj_disjunctive_geometry(const fj_cpu_climber_t<i_t, f_t>& climber)
{
  return has_cpufj_disjunctive_geometry(climber, [](i_t, i_t, bool) { return false; });
}

}  // namespace cuopt::mathematical_optimization::mip
