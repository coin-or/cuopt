/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/diversity/population.cuh>
#include <mip_heuristics/feasibility_jump/cpu/state.hpp>

#include <deque>
#include <limits>
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
      [this](const std::vector<f_t>& assignment, f_t objective, f_t user_objective, bool is_best) {
        add_feasible_solution(assignment, objective, user_objective, is_best);
      });
  }
  ~lns_population_feed_t() { population_.clear_feasible_solution_callback(); }
  lns_population_feed_t(const lns_population_feed_t&)            = delete;
  lns_population_feed_t& operator=(const lns_population_feed_t&) = delete;

  bool best_feasible(std::vector<f_t>& assignment)
  {
    return best_.adopt(std::numeric_limits<f_t>::infinity(), assignment);
  }

  // Recently accepted population members, unvalidated in the consumer's representation.
  void recent_feasible(std::vector<std::vector<f_t>>& out)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    out.insert(out.end(), recent_.begin(), recent_.end());
  }

 private:
  static constexpr size_t max_recent = 8;

  void add_feasible_solution(const std::vector<f_t>& assignment,
                             f_t objective,
                             f_t user_objective,
                             bool is_best)
  {
    // Only the population's best slot feeds CPUFJ LNS; accepted members still
    // enter the recent seed queue even when their objective is not an improvement.
    if (is_best) best_.publish(objective, user_objective, assignment);
    std::lock_guard<std::mutex> lock(mutex_);
    if (recent_.size() == max_recent) recent_.pop_front();
    recent_.push_back(assignment);
  }

  population_t<i_t, f_t>& population_;
  fj_cpu_shared_incumbent_t<i_t, f_t> best_;
  std::mutex mutex_;
  std::deque<std::vector<f_t>> recent_;
};

}  // namespace cuopt::mathematical_optimization::mip
