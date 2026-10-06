/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/lns/cpufj_validation.cuh>
#include <mip_heuristics/utils.hpp>
#include <utilities/scope_guard.hpp>
#include <utilities/splitmix64.hpp>
#include <utilities/timer.hpp>

#include <array>
#include <atomic>
#include <memory>
#include <mutex>

namespace cuopt::mathematical_optimization::mip {

// Feasibility search occupies the two reserved LNS tasks until their first seed.
// Prepare both clones before launching either task: the production LNS anchor can
// start changing as soon as the first task finishes its feasibility search.
template <typename i_t, typename f_t>
class lns_feasibility_bootstrap_t {
 public:
  lns_feasibility_bootstrap_t(const fj_cpu_climber_t<i_t, f_t>& anchor,
                              int first_lane,
                              int64_t base_seed)
    : problem_(anchor.problem),
      bounds_(anchor.h_var_bounds.underlying()),
      first_lane_(first_lane),
      base_seed_(base_seed)
  {
    cuopt::splitmix64_t seeds(base_seed);
    for (int slot = 0; slot < 2; ++slot) {
      auto settings   = anchor.settings;
      settings.seed   = seeds.next_i32();
      climbers_[slot] = init_fj_cpu_clone(anchor, anchor.preemption_flag, settings);
      climbers_[slot]->shared_incumbent.reset();
      climbers_[slot]->low_latency = anchor.low_latency;
      climbers_[slot]->log_prefix  = "[LNS feasibility " + std::to_string(slot) + "] ";
      active_[slot]                = climbers_[slot].get();
    }
  }

  // Seed handoff cancels only the temporary searches, never the production LNS
  // workers or the enclosing solver. Global preemption remains directly visible.
  void request_stop()
  {
    if (cancel_.exchange(true)) return;
    std::lock_guard<std::mutex> lock(active_mutex_);
    for (auto* climber : active_)
      if (climber) climber->halted.store(true);
  }

  void notify_seed(const std::vector<f_t>& assignment)
  {
    if (cancel_.load()) return;
    auto x = assignment;
    if (clamp_and_validate_cpufj_lns_seed(*problem_, bounds_, x)) request_stop();
  }

  template <typename Snapshot, typename Submit>
  void run(int slot,
           const Snapshot& snapshot,
           const Submit& submit,
           timer_t timer = timer_t(std::numeric_limits<double>::infinity()))
  {
    auto climber = std::move(climbers_[slot]);
    // Clear only this slot, under the notifier's lock, before destroying its clone.
    cuopt::scope_guard clear_active([this, slot] {
      std::lock_guard<std::mutex> lock(active_mutex_);
      active_[slot] = nullptr;
    });
    std::vector<f_t> seed(problem_->n_variables);
    if (snapshot(seed)) notify_seed(seed);
    if (cancel_.load() || climber->preemption_flag.load() || timer.check_time_limit()) return;

    climber->improvement_callback = [this, &submit](
                                      f_t, const std::vector<f_t>& candidate, double) {
      if (cancel_.load()) return;
      auto x = candidate;
      if (!clamp_and_validate_cpufj_lns_seed(*problem_, bounds_, x)) return;
      const f_t objective = compensated_dot2_indexed(problem_->h_obj_coeffs.data(),
                                                     x.data(),
                                                     problem_->h_objective_vars.data(),
                                                     problem_->h_objective_vars.size());
      if (!std::isfinite(objective)) return;
      submit(x, objective);
      request_stop();
    };
    apply_lane_diversification(*climber, first_lane_ + slot, base_seed_);
    if (cancel_.load() || climber->preemption_flag.load() || timer.check_time_limit()) return;
    // One continuous solve preserves CPUFJ's search state while awaiting a seed.
    CUOPT_LOG_DEBUG("LNS feasibility slot %d begins", slot);
    cpufj_solve(climber.get(), timer.remaining_time());
    CUOPT_LOG_DEBUG("LNS feasibility slot %d ends; feasible %d", slot, climber->feasible_found);
  }

 private:
  std::shared_ptr<const fj_cpu_problem_t<i_t, f_t>> problem_;
  const std::vector<typename type_2<f_t>::type> bounds_;
  const int first_lane_;
  const int64_t base_seed_;
  std::array<std::unique_ptr<fj_cpu_climber_t<i_t, f_t>>, 2> climbers_;
  std::array<fj_cpu_climber_t<i_t, f_t>*, 2> active_{};
  std::atomic<bool> cancel_{false};
  std::mutex active_mutex_;
};

}  // namespace cuopt::mathematical_optimization::mip
