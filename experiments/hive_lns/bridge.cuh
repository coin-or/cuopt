/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
// Frozen integration. The evolvable header receives only immutable owning CPU data.
#include <pthread.h>
#include <atomic>
#include <deque>
#include <exception>
#include <mip_heuristics/diversity/population.cuh>
#include <mip_heuristics/lns_thread_budget.hpp>
#include <mip_heuristics/solver_context.cuh>
#include <mip_heuristics/utils.cuh>
#include <mutex>
#include <stdexcept>
#include <utilities/copy_helpers.hpp>
#include "../../cpp/src/mip_heuristics/lns_improvement.hpp"
#include "repair_tools.cuh"

namespace cuopt::mathematical_optimization::mip {
template <typename i_t, typename f_t>
class hive_lns_bridge_t {
 public:
  hive_lns_bridge_t(mip_solver_context_t<i_t, f_t>& context,
                    population_t<i_t, f_t>& population,
                    cuopt::timer_t timer,
                    cuopt::hive_lns::run_lns_fn run = cuopt::hive_lns::run_lns,
                    bool enabled                    = true)
    : context_(context), population_(population), timer_(timer)
  {
    if (!enabled) return;
    if (lns_worker_count(omp_get_num_threads(),
                         context.settings.determinism_mode == CUOPT_MODE_DETERMINISTIC) == 0)
      return;
    try {
      start(run);
    } catch (const std::exception& e) {
      CUOPT_LOG_WARN("Hive LNS setup failed: %s", e.what());
    } catch (...) {
      CUOPT_LOG_WARN("Hive LNS setup failed with unknown error");
    }
  }
  ~hive_lns_bridge_t() { finish(); }
  void request_stop() { stop_.store(true); }
  void finish()
  {
    request_stop();
    if (!worker_started_) return;
    auto* worker = this;
#pragma omp taskwait depend(in : *worker)
    population_.lns_observer = {};
    worker_started_          = false;
  }
  const std::vector<f_t>& best_assignment() const { return best_; }

 private:
  void start(cuopt::hive_lns::run_lns_fn run)
  {
    auto& pb    = *context_.problem_ptr;
    auto stream = pb.handle_ptr->get_stream();
    auto copy   = [stream](auto& dest, const auto& source) {
      auto values = cuopt::host_copy(source, stream);
      dest.assign(values.begin(), values.end());
    };
    copy(model_.offsets, pb.offsets);
    copy(model_.columns, pb.variables);
    copy(model_.coefficients, pb.coefficients);
    copy(model_.objective, pb.objective_coefficients);
    copy(model_.row_lower, pb.constraint_lower_bounds);
    copy(model_.row_upper, pb.constraint_upper_bounds);
    model_.feasibility_tolerance = pb.tolerances.absolute_tolerance;
    model_.relative_tolerance    = pb.tolerances.relative_tolerance;
    model_.integrality_tolerance = pb.tolerances.integrality_tolerance;
    for (size_t r = 0; r < model_.row_lower.size(); ++r) {
      model_.row_tolerances.push_back(
        get_cstr_tolerance<i_t, f_t>(model_.row_lower[r],
                                     model_.row_upper[r],
                                     pb.tolerances.absolute_tolerance,
                                     pb.tolerances.relative_tolerance));
    }
    auto bounds = cuopt::host_copy(pb.variable_bounds, stream);
    auto types  = cuopt::host_copy(pb.variable_types, stream);
    pb.handle_ptr->sync_stream();
    for (size_t j = 0; j < bounds.size(); ++j) {
      model_.lower.push_back(bounds[j].x);
      model_.upper.push_back(bounds[j].y);
      model_.integer.push_back(types[j] == var_t::INTEGER);
    }
    cudaGetDevice(&device_);
    {
      std::lock_guard<std::recursive_mutex> lock(population_.write_mutex);
      for (auto& member : population_.solutions) {
        if (member.first && member.second.get_feasible())
          offer(member.second.get_host_assignment());
      }
    }
    population_.lns_observer = [this](const std::vector<f_t>& x) { offer(x); };
    worker_started_          = true;
    auto* worker             = this;
#pragma omp task firstprivate(worker, run) depend(out : *worker) default(none) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY)
    worker->run_worker(run);
  }

  void run_worker(cuopt::hive_lns::run_lns_fn run)
  {
    // OMP threads are reused by other solver tasks after this worker completes.
    char previous_name[16]{};
    pthread_getname_np(pthread_self(), previous_name, sizeof(previous_name));
    const int previous_max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    try {
      if (cudaSetDevice(device_) != cudaSuccess)
        throw std::runtime_error("LNS CUDA device setup failed");
      if (pthread_setname_np(pthread_self(), "cuopt-hive-lns") != 0)
        throw std::runtime_error("LNS thread attribution setup failed");
      // This handle and both repair backends belong exclusively to this worker.
      raft::handle_t repair_handle;
      model_.repair.cpufj = [this, &repair_handle](const auto& request) {
        return cuopt::hive_lns::repair_neighborhood(
          model_,
          request,
          cuopt::hive_lns::repair_backend_t::cpufj,
          [this] { return stopped(); },
          &repair_handle,
          timer_.remaining_time(),
          context_.preempt_heuristic_solver_);
      };
      model_.repair.submip = [this, &repair_handle](const auto& request) {
        return cuopt::hive_lns::repair_neighborhood(
          model_,
          request,
          cuopt::hive_lns::repair_backend_t::submip,
          [this] { return stopped(); },
          &repair_handle,
          timer_.remaining_time(),
          context_.preempt_heuristic_solver_);
      };
      CUOPT_LOG_INFO(
        "HIVE_LNS_STARTED omp_thread=%d team_size=%d", omp_get_thread_num(), omp_get_num_threads());
      run(
        model_,
        [this] { return snapshot(); },
        [this](const auto& x) { submit(x); },
        [this] { return stopped(); },
        context_.base_seed);
    } catch (const std::exception& e) {
      stop_.store(true);
      CUOPT_LOG_WARN("LNS worker disabled after failure: %s", e.what());
    } catch (...) {
      stop_.store(true);
      CUOPT_LOG_WARN("LNS worker disabled after unknown failure");
    }
    CUOPT_LOG_INFO("HIVE_LNS_INPUTS %lu", inputs_);
    CUOPT_LOG_INFO("HIVE_LNS_FINISHED");
    omp_set_num_threads(previous_max_threads);
    if (previous_name[0]) pthread_setname_np(pthread_self(), previous_name);
  }
  bool stopped() const
  {
    return stop_.load() || timer_.check_time_limit() || context_.preempt_heuristic_solver_.load();
  }
  void offer(const std::vector<f_t>& source)
  {
    if (stopped()) return;
    std::vector<double> x(source.begin(), source.end());
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (cache_.size() == 8) cache_.pop_front();
    cache_.push_back(std::move(x));
  }
  cuopt::hive_lns::population_t snapshot()
  {
    cuopt::hive_lns::population_t result;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      result.assign(cache_.begin(), cache_.end());
    }
    result.erase(
      std::remove_if(
        result.begin(), result.end(), [this](const auto& x) { return !model_.feasible(x); }),
      result.end());
    inputs_ += result.size();
    return result;
  }
  void submit(const std::vector<double>& x)
  {
    if (stopped()) return;
    if (!model_.feasible(x)) throw std::runtime_error("Invalid LNS submission");
    const auto cost = model_.cost(x);
    if (cost >= best_cost_) return;
    if (stopped()) return;
    best_.assign(x.begin(), x.end());
    best_cost_ = cost;
    // Same lock ordering as population.add_solution: population, then publication.
    // Keep the population best stable through objective recomputation/publication.
    std::lock_guard<std::recursive_mutex> lock(population_.write_mutex);
    if (stopped() || !population_.is_feasible()) return;
    f_t population_best = std::numeric_limits<f_t>::infinity();
    for (auto& member : population_.solutions) {
      if (member.first && member.second.get_feasible())
        population_best = std::min(population_best, member.second.get_objective());
    }
    context_.solution_publication.publish_if_better(
      context_.problem_ptr, best_, f_t(cost), population_best);
  }
  mip_solver_context_t<i_t, f_t>& context_;
  population_t<i_t, f_t>& population_;
  cuopt::timer_t timer_;
  cuopt::hive_lns::model_t model_;
  std::mutex cache_mutex_;
  std::deque<std::vector<double>> cache_;
  std::atomic<bool> stop_{false};
  bool worker_started_{false};
  std::vector<f_t> best_;
  double best_cost_ = std::numeric_limits<double>::infinity();
  size_t inputs_    = 0;
  int device_       = 0;
};
}  // namespace cuopt::mathematical_optimization::mip
