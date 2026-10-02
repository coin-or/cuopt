/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/diversity/population.cuh>
#include <mip_heuristics/diversity/population_observer.hpp>

#include <algorithm>
#include <deque>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Host-side LNS seed source. CPU workers read it without touching device memory or the
// population locks. Pending candidates come from the external queue and are not validated:
// the consumer must check feasibility.
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

  bool take_seed_candidate(std::vector<f_t>& out_assignment,
                           f_t& out_objective,
                           f_t objective_cutoff)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.erase(std::remove_if(pending_.begin(),
                                  pending_.end(),
                                  [objective_cutoff](const auto& seed) {
                                    return seed.first >= objective_cutoff;
                                  }),
                   pending_.end());
    const bool cached = !best_assignment_.empty() && best_objective_ < objective_cutoff &&
                        best_objective_ < last_cached_objective_;
    auto best = std::min_element(pending_.begin(),
                                 pending_.end(),
                                 [](const auto& a, const auto& b) { return a.first < b.first; });
    if (best != pending_.end() && (!cached || best->first < best_objective_)) {
      out_objective  = best->first;
      out_assignment = std::move(best->second);
      pending_.erase(best);
      return true;
    }
    if (!cached) return false;
    out_objective  = best_objective_;
    out_assignment = best_assignment_;
    // A cached point can cease to improve after private normalization. Deliver it
    // once so rejection cannot repeatedly hide usable pending incumbents.
    last_cached_objective_ = best_objective_;
    return true;
  }

  void on_external_solution(const std::vector<f_t>& assignment, f_t objective) override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.emplace_back(objective, assignment);
    if (pending_.size() > max_pending) {
      auto worst = std::max_element(pending_.begin(),
                                    pending_.end(),
                                    [](const auto& a, const auto& b) { return a.first < b.first; });
      pending_.erase(worst);
    }
  }

  void on_feasible_solution(const std::vector<f_t>& assignment,
                            f_t objective,
                            bool is_best) override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (recent_.size() == max_recent) recent_.pop_front();
    recent_.push_back(assignment);
    if (!is_best) return;
    best_assignment_ = assignment;
    best_objective_  = objective;
  }

  // Recently accepted population members, unvalidated in the consumer's representation.
  void recent_feasible(std::vector<std::vector<f_t>>& out)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    out.insert(out.end(), recent_.begin(), recent_.end());
  }

 private:
  static constexpr size_t max_pending = 10;
  static constexpr size_t max_recent  = 8;

  population_t<i_t, f_t>& population_;
  std::mutex mutex_;
  std::vector<f_t> best_assignment_;
  f_t best_objective_{std::numeric_limits<f_t>::max()};
  f_t last_cached_objective_{std::numeric_limits<f_t>::infinity()};
  std::vector<std::pair<f_t, std::vector<f_t>>> pending_;
  std::deque<std::vector<f_t>> recent_;
};

}  // namespace cuopt::mathematical_optimization::mip
