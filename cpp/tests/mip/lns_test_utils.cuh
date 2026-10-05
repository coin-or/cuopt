/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/feasibility_jump/cpu/climber.hpp>

namespace cuopt::mathematical_optimization::mip::test {

struct host_model_t {
  std::vector<double> coefficients, lower, upper, objective, row_lower, row_upper;
  std::vector<int> columns, offsets;
  std::vector<var_t> types;
};

inline std::unique_ptr<fj_cpu_climber_t<int, double>> make_anchor(
  const host_model_t& m,
  std::atomic<bool>& preemption,
  mip_solver_settings_t<int, double>::tolerances_t tolerances,
  fj_settings_t settings = {})
{
  return init_fj_cpu_from_host_model<int, double>((int)m.lower.size(),
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
                                                  settings);
}

}  // namespace cuopt::mathematical_optimization::mip::test
