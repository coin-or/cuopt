/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Callbacks run on the producing thread while population locks are held. They must be short
// and must not call back into the population.
template <typename i_t, typename f_t>
class population_observer_t {
 public:
  virtual ~population_observer_t() = default;
  // Every finite candidate passed to add_external_solution, before any validation.
  virtual void on_external_solution(const std::vector<f_t>& /*assignment*/, f_t /*objective*/) {}
  // Every feasible solution stored in the population. is_best marks the incumbent slot.
  virtual void on_feasible_solution(const std::vector<f_t>& /*assignment*/,
                                    f_t /*objective*/,
                                    bool /*is_best*/)
  {
  }
};

}  // namespace cuopt::mathematical_optimization::mip
