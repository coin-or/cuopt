/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/diversity/population.cuh>

#include <deque>
#include <mutex>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Host-side LNS seed source. CPU workers read it without touching device memory or the
// population locks.
template <typename i_t, typename f_t>
class lns_population_feed_t {
 public:
  explicit lns_population_feed_t(population_t<i_t, f_t>& population) : population_(population)
  {
    population_.set_feasible_solution_callback(
      [this](const std::vector<f_t>& assignment) { add_feasible_solution(assignment); });
  }
  ~lns_population_feed_t() { population_.clear_feasible_solution_callback(); }
  lns_population_feed_t(const lns_population_feed_t&)            = delete;
  lns_population_feed_t& operator=(const lns_population_feed_t&) = delete;

  // Recently accepted population members, unvalidated in the consumer's representation.
  void recent_feasible(std::vector<std::vector<f_t>>& out)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    out.insert(out.end(), recent_.begin(), recent_.end());
  }

 private:
  static constexpr size_t max_recent = 8;

  void add_feasible_solution(const std::vector<f_t>& assignment)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (recent_.size() == max_recent) recent_.pop_front();
    recent_.push_back(assignment);
  }

  population_t<i_t, f_t>& population_;
  std::mutex mutex_;
  std::deque<std::vector<f_t>> recent_;
};

}  // namespace cuopt::mathematical_optimization::mip
