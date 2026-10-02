/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/diversity/population.cuh>
#include <mip_heuristics/diversity/population_observer.hpp>

#include <deque>
#include <mutex>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Host-side LNS seed source. CPU workers read it without touching device memory or the
// population locks.
template <typename i_t, typename f_t>
class lns_population_feed_t final : public population_observer_t<i_t, f_t> {
 public:
  explicit lns_population_feed_t(population_t<i_t, f_t>& population) : population_(population)
  {
    population_.add_observer(this);
  }
  ~lns_population_feed_t() override { population_.remove_observer(this); }
  lns_population_feed_t(const lns_population_feed_t&)            = delete;
  lns_population_feed_t& operator=(const lns_population_feed_t&) = delete;

  void on_feasible_solution(const std::vector<f_t>& assignment, f_t, bool) override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (recent_.size() == max_recent) recent_.pop_front();
    recent_.push_back(assignment);
  }

  // Recently accepted population members, unvalidated in the consumer's representation.
  void recent_feasible(std::vector<std::vector<f_t>>& out)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    out.insert(out.end(), recent_.begin(), recent_.end());
  }

 private:
  static constexpr size_t max_recent = 8;

  population_t<i_t, f_t>& population_;
  std::mutex mutex_;
  std::deque<std::vector<f_t>> recent_;
};

}  // namespace cuopt::mathematical_optimization::mip
