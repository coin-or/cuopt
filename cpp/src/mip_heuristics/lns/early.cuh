/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/lns/cpufj.cuh>
#include <mip_heuristics/lns/improvement.hpp>
#include <mip_heuristics/lns/repair_tools.cuh>
#include <mip_heuristics/lns/task_errors.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/scope_guard.hpp>

#include <exception>
#include <memory>
#include <mutex>

namespace cuopt::mathematical_optimization::mip {

// The two improvement tasks own private search state in the same representation as
// the presolve CPUFJ portfolio. All communication uses its shared host incumbent.
template <typename i_t, typename f_t>
class early_lns_t {
 public:
  using report_fn = std::function<void(f_t, const std::vector<f_t>&, const char*)>;

  early_lns_t(const fj_cpu_climber_t<i_t, f_t>& anchor,
              std::shared_ptr<fj_cpu_shared_incumbent_t<i_t, f_t>> shared,
              std::atomic<bool>& preemption,
              cuopt::lns::task_errors_t& task_errors,
              report_fn report,
              uint64_t seed,
              cuopt::lns::run_lns_fn run_lns = cuopt::lns::run_lns)
    : shared_(std::move(shared)),
      preemption_(preemption),
      task_errors_(task_errors),
      report_(std::move(report)),
      seed_(seed),
      run_lns_(run_lns)
  {
    fj_settings_t settings;
    settings.seed = seed;
    cpufj_        = init_fj_cpu_clone(anchor, preemption_, settings);
    // Only independently validated LNS improvements enter the shared portfolio.
    cpufj_->shared_incumbent.reset();
    cpufj_->log_prefix           = "[Early CPUFJ LNS] ";
    cpufj_->improvement_callback = [this](f_t, const std::vector<f_t>& x, double) {
      submit(x, "CPUFJ LNS");
    };

    const auto& problem = *anchor.problem;
    model_.offsets.assign(problem.offsets.begin(), problem.offsets.end());
    model_.columns.assign(problem.variables.begin(), problem.variables.end());
    model_.coefficients.assign(problem.coefficients.begin(), problem.coefficients.end());
    model_.objective.assign(problem.h_obj_coeffs.begin(), problem.h_obj_coeffs.end());
    model_.row_lower.assign(problem.cstr_lb.begin(), problem.cstr_lb.end());
    model_.row_upper.assign(problem.cstr_ub.begin(), problem.cstr_ub.end());
    model_.feasibility_tolerance = problem.tolerances.absolute_tolerance;
    model_.relative_tolerance    = problem.tolerances.relative_tolerance;
    model_.integrality_tolerance = problem.tolerances.integrality_tolerance;
    for (i_t r = 0; r < problem.n_constraints; ++r) {
      model_.row_tolerances.push_back(
        get_cstr_tolerance<i_t, f_t>(problem.cstr_lb[r],
                                     problem.cstr_ub[r],
                                     problem.tolerances.absolute_tolerance,
                                     problem.tolerances.relative_tolerance));
    }
    for (i_t v = 0; v < problem.n_variables; ++v) {
      const auto bounds = anchor.h_var_bounds[v];
      model_.lower.push_back(get_lower(bounds));
      model_.upper.push_back(get_upper(bounds));
      model_.integer.push_back(problem.h_var_types[v] == var_t::INTEGER);
    }
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }

  ~early_lns_t() { finish(); }

  void start()
  {
    if (started_) return;
    started_     = true;
    auto* worker = this;
    auto* cpu    = cpufj_.get();
#pragma omp task firstprivate(worker, cpu) depend(out : *cpu) default(none) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY)
    {
      try {
        worker->run_cpufj();
      } catch (...) {
        worker->fail(std::current_exception());
      }
    }
#pragma omp task firstprivate(worker) depend(out : *worker) default(none) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY)
    {
      try {
        worker->run_repair_lns();
      } catch (...) {
        worker->fail(std::current_exception());
      }
    }
    // Both tasks have reserved capacity. Do not enter a task scheduling point
    // until they are running on other team members: a taskwait for feasibility
    // lanes could otherwise execute a queued persistent task on the solve thread,
    // preventing that thread from ever reaching the LNS stop signal.
    while (workers_started_.load() < 2)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  void set_source(std::function<bool(std::vector<f_t>&)> source)
  {
    std::lock_guard<std::mutex> lock(source_mutex_);
    source_ = std::move(source);
  }

  void request_stop()
  {
    stop_.store(true);
    cpufj_->halted = true;
  }

  void finish()
  {
    request_stop();
    if (!started_) return;
    auto* worker = this;
    auto* cpu    = cpufj_.get();
#pragma omp taskwait depend(in : *cpu, *worker)
    started_ = false;
  }

 private:
  bool stopped() const { return stop_.load() || preemption_.load(); }

  void fail(std::exception_ptr error)
  {
    request_stop();
    task_errors_.capture(std::move(error));
  }

  bool snapshot(std::vector<f_t>& assignment, f_t& objective)
  {
    std::function<bool(std::vector<f_t>&)> source;
    {
      std::lock_guard<std::mutex> lock(source_mutex_);
      source = source_;
    }
    std::vector<f_t> external;
    if (source && source(external)) {
      const std::vector<double> x(external.begin(), external.end());
      if (model_.feasible(x)) {
        const f_t cost = model_.cost(x);
        shared_->publish(cost, cpufj_->get_user_objective(cost), external);
      }
    }
    assignment.resize(model_.lower.size());
    return shared_->adopt(std::numeric_limits<f_t>::infinity(), assignment, &objective);
  }

  cuopt::lns::population_t repair_lns_snapshot()
  {
    std::vector<f_t> assignment;
    f_t objective;
    if (!snapshot(assignment, objective)) return {};
    std::vector<double> x(assignment.begin(), assignment.end());
    if (!model_.feasible(x)) return {};
    return {std::move(x)};
  }

  void submit(const std::vector<f_t>& assignment, const char* origin)
  {
    if (stopped()) return;
    const std::vector<double> x(assignment.begin(), assignment.end());
    if (!model_.feasible(x)) return;
    const f_t objective = model_.cost(x);
    // Publish before invoking callbacks, so polling only holds the short copy lock.
    shared_->publish(objective, cpufj_->get_user_objective(objective), assignment);
    report_(objective, assignment, origin);
  }

  void run_cpufj()
  {
    ++workers_started_;
    const int previous_max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    cuopt::scope_guard restore([&] { omp_set_num_threads(previous_max_threads); });
    run_cpufj_lns_ruin_repair<i_t, f_t>(
      cpufj_.get(), [this](auto& x, auto& objective) { return snapshot(x, objective); });
  }

  void run_repair_lns()
  {
    ++workers_started_;
    const int previous_max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    cuopt::scope_guard restore([&] { omp_set_num_threads(previous_max_threads); });
    RAFT_CUDA_TRY(cudaSetDevice(device_));
    raft::handle_t repair_handle;
    model_.repair.cpufj = [this, &repair_handle](const auto& request) {
      return repair(request, cuopt::lns::repair_backend_t::cpufj, repair_handle);
    };
    model_.repair.submip = [this, &repair_handle](const auto& request) {
      return repair(request, cuopt::lns::repair_backend_t::submip, repair_handle);
    };
    run_lns_(
      model_,
      [this] { return repair_lns_snapshot(); },
      [this](const auto& x) { submit(std::vector<f_t>(x.begin(), x.end()), "Repair LNS"); },
      [this] { return stopped(); },
      seed_);
  }

  cuopt::lns::repair_result_t repair(const cuopt::lns::repair_request_t& request,
                                     cuopt::lns::repair_backend_t backend,
                                     const raft::handle_t& handle)
  {
    return cuopt::lns::repair_neighborhood(
      model_,
      request,
      backend,
      [this] { return stopped(); },
      &handle,
      std::numeric_limits<double>::infinity(),
      preemption_);
  }

  std::shared_ptr<fj_cpu_shared_incumbent_t<i_t, f_t>> shared_;
  std::atomic<bool>& preemption_;
  cuopt::lns::task_errors_t& task_errors_;
  report_fn report_;
  std::mutex source_mutex_;
  std::function<bool(std::vector<f_t>&)> source_;
  uint64_t seed_;
  cuopt::lns::run_lns_fn run_lns_;
  std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> cpufj_;
  cuopt::lns::model_t model_;
  std::atomic<bool> stop_{false};
  std::atomic<int> workers_started_{0};
  bool started_{false};
  int device_{0};
};
}  // namespace cuopt::mathematical_optimization::mip
