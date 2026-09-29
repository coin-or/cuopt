/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/early_heuristic.cuh>
#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class early_lns_t;

template <typename i_t, typename f_t>
class early_cpufj_t : public early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>> {
 public:
  early_cpufj_t(const optimization_problem_t<i_t, f_t>& op_problem,
                const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                early_incumbent_callback_t<f_t> incumbent_callback,
                uint64_t seed);

  ~early_cpufj_t();

  static constexpr const char* name() { return "CPUFJ"; }

  // n_lanes is the total persistent-worker budget, including the two optional LNS
  // tasks. Callers sharing the team with other work reserve their capacity first.
  void start(int n_lanes, bool low_latency = false);
  void stop(bool keep_lns = false);
  void set_incumbent_callback(early_incumbent_callback_t<f_t> callback, bool replay_best = false);
  void set_lns_source(std::function<bool(std::vector<f_t>&)> source);

  int lane_count() const { return (int)climbers_.size() + improvement_lane_count(); }
  int improvement_lane_count() const { return lns_ ? 2 : 0; }

 private:
  friend class early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>>;

  std::vector<f_t> to_user_assignment(const std::vector<f_t>& assignment);

  const optimization_problem_t<i_t, f_t>* problem_ptr_{nullptr};
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances_;
  std::vector<std::unique_ptr<fj_cpu_climber_t<i_t, f_t>>> climbers_;
  std::unique_ptr<early_lns_t<i_t, f_t>> lns_;
  std::thread worker_;
  std::atomic<bool> preemption_flag_{false};
  std::atomic<bool> lns_preemption_flag_{false};
  // Explicit seed for this climber's FJ RNG, resolved once from the solve's base seed (see
  // mip_solver_context_t::base_seed) since this heuristic runs before that context exists.
  uint64_t seed_;
  // try_update_best and the incumbent callback behind it are not thread-safe, and every lane
  // reports into them from its own task.
  std::mutex incumbent_mutex_;
};

}  // namespace cuopt::mathematical_optimization::mip
