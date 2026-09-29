/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "early_cpufj.cuh"
#include "early_lns.cuh"

#include <mip_heuristics/lns_thread_budget.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/splitmix64.hpp>

#include <omp.h>

#include <algorithm>
#include <string>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
early_cpufj_t<i_t, f_t>::early_cpufj_t(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  early_incumbent_callback_t<f_t> incumbent_callback,
  uint64_t seed)
  : early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>>(op_problem, std::move(incumbent_callback)),
    problem_ptr_(&op_problem),
    tolerances_(tolerances),
    seed_(seed)
{
}

template <typename i_t, typename f_t>
early_cpufj_t<i_t, f_t>::~early_cpufj_t()
{
  stop();
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::start(int n_lanes, bool low_latency)
{
  const bool threaded = !omp_in_parallel();
  // 1: presolve, 1: early GPU FJ, 1: early CPU FJ
  if (lns_ || !climbers_.empty() ||
      (!threaded && omp_get_num_threads() < CUOPT_MIP_EARLY_CPUFJ_REQUIRED_THREAD_COUNT)) {
    return;
  }

  this->preemption_flag_.store(false);
  this->start_time_ = std::chrono::steady_clock::now();

  // Tasks are not preempted, so a lane posted beyond the team size would sit in the queue for the
  // whole of presolve without running an iteration.
  const int worker_budget =
    threaded
      ? 1
      : std::clamp(
          n_lanes, 1, std::max(1, omp_get_num_threads() - CUOPT_MIP_EARLY_CPUFJ_RESERVED_THREADS));
  const int improvement_lanes = threaded ? 0 : presolve_lns_worker_count(worker_budget);
  n_lanes                     = worker_budget - improvement_lanes;
  cuopt::splitmix64_t seed_rng(seed_);
  const int64_t base_seed = seed_rng.next_i64();
  climbers_.resize(n_lanes);

  auto report_incumbent = [this](f_t solver_obj, const std::vector<f_t>& assignment, double) {
    std::lock_guard<std::mutex> guard(incumbent_mutex_);
    this->try_update_best(solver_obj, assignment);
  };

  // Lane 0 builds the host problem representation and every other lane copies it. All of it
  // finishes before the first task is posted, so no lane reads a template another lane is running
  // on.
  for (int k = 0; k < n_lanes; ++k) {
    fj_settings_t settings;
    settings.seed = seed_rng.next_i32();
    if (k == 0) {
      climbers_[0] = init_fj_cpu_from_optimization_problem(
        *this->problem_ptr_, tolerances_, preemption_flag_, settings);
    } else {
      climbers_[k] = init_fj_cpu_clone(*climbers_[0], preemption_flag_, settings);
    }
    climbers_[k]->low_latency = low_latency;
    apply_lane_diversification<i_t, f_t>(*climbers_[k], k, base_seed);
    climbers_[k]->log_prefix           = "[Early CPUFJ " + std::to_string(k) + "] ";
    climbers_[k]->improvement_callback = report_incumbent;
  }

  auto shared = std::make_shared<fj_cpu_shared_incumbent_t<i_t, f_t>>();
  for (int k = 0; k < n_lanes; ++k)
    climbers_[k]->shared_incumbent = shared;

  // Construct both private search states before any lane starts mutating the anchor.
  if (improvement_lanes) {
    try {
      lns_ = std::make_unique<early_lns_t<i_t, f_t>>(
        *climbers_[0],
        shared,
        lns_preemption_flag_,
        [this](f_t objective, const std::vector<f_t>& x, const char* origin) {
          std::lock_guard<std::mutex> guard(incumbent_mutex_);
          this->try_update_best(objective, x, origin);
        },
        seed_);
    } catch (const std::exception& e) {
      CUOPT_LOG_WARN("Early LNS setup failed: %s", e.what());
    } catch (...) {
      CUOPT_LOG_WARN("Early LNS setup failed with unknown error");
    }
  }
  if (!threaded)
    CUOPT_LOG_INFO("Early CPUFJ budget: %d feasibility + %d LNS workers within %d OpenMP threads",
                   n_lanes,
                   improvement_lane_count(),
                   omp_get_num_threads());
  CUOPT_LOG_DEBUG("Launching %d early CPUFJ %s", n_lanes, threaded ? "thread" : "tasks");
  if (threaded) {
    auto* climber = climbers_[0].get();
    worker_       = std::thread([climber] { cpufj_solve(climber); });
    return;
  }
  for (int k = 0; k < n_lanes; ++k) {
    auto* climber = climbers_[k].get();
#pragma omp task firstprivate(climber) priority(CUOPT_DEFAULT_TASK_PRIORITY) \
  depend(out : *climber) default(none)
    cpufj_solve(climber);
  }
  if (lns_) lns_->start();
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::stop(bool keep_lns)
{
  if (climbers_.empty() && !lns_) { return; }

  preemption_flag_.store(true);
  if (lns_ && !keep_lns) lns_->request_stop();

  // Every lane is told to stop before any wait, otherwise the first wait blocks on a lane that has
  // not been asked to exit yet.
  for (auto& climber : climbers_) {
    climber->halted = true;
  }
  if (worker_.joinable()) {
    worker_.join();
  } else {
    for (size_t k = 0; k < climbers_.size(); ++k) {
#pragma omp taskwait depend(in : *climbers_[k])  // Wait for each early CPUFJ task to finish
    }
  }

  if (lns_ && !keep_lns) {
    lns_->finish();
    lns_.reset();
  }

  [[maybe_unused]] i_t total_iterations = 0;
  for (const auto& climber : climbers_) {
    total_iterations += climber->iterations;
  }

  CUOPT_LOG_DEBUG("[Early CPUFJ] Stopped after %d iterations over %d climbers, solution_found=%d",
                  total_iterations,
                  (int)climbers_.size(),
                  !keep_lns && this->solution_found_);

  climbers_.clear();
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::set_incumbent_callback(early_incumbent_callback_t<f_t> callback,
                                                     bool replay_best)
{
  std::lock_guard<std::mutex> guard(incumbent_mutex_);
  this->incumbent_callback_ = std::move(callback);
  if (replay_best && this->solution_found_ && this->incumbent_callback_) {
    this->incumbent_callback_(this->best_objective_,
                              this->get_best_user_objective(),
                              this->best_assignment_,
                              "Persistent LNS");
  }
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::set_lns_source(std::function<bool(std::vector<f_t>&)> source)
{
  if (lns_) lns_->set_source(std::move(source));
}

template <typename i_t, typename f_t>
std::vector<f_t> early_cpufj_t<i_t, f_t>::to_user_assignment(const std::vector<f_t>& assignment)
{
  return assignment;
}

#if MIP_INSTANTIATE_FLOAT
template class early_cpufj_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class early_cpufj_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
