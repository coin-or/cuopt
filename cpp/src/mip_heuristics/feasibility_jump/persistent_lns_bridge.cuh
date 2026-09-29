/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/diversity/population.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <utilities/copy_helpers.hpp>

#include <mutex>

namespace cuopt::mathematical_optimization::mip {

// The persistent workers retain the Papilo model. Main heuristics use the further
// reduced cuOpt model. Map both directions on an immutable copy and private stream.
// The owner must join the workers before destroying this bridge or the population.
template <typename i_t, typename f_t>
class persistent_lns_bridge_t {
 public:
  persistent_lns_bridge_t(const problem_t<i_t, f_t>& problem, population_t<i_t, f_t>& population)
    : population_(population)
  {
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
    problem.handle_ptr->sync_stream();
    problem_ = std::make_unique<problem_t<i_t, f_t>>(problem, &handle_);
    handle_.sync_stream();
    population_.enable_lns_seed_polling();
  }

  void submit(const std::vector<f_t>& papilo_assignment, const char* origin = "LNS")
  {
    std::vector<f_t> assignment;
    f_t objective;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      RAFT_CUDA_TRY(cudaSetDevice(device_));
      auto x = cuopt::device_copy(papilo_assignment, handle_.get_stream());
      if (!problem_->pre_process_assignment(x)) return;
      solution_t<i_t, f_t> candidate(*problem_);
      candidate.assignment = std::move(x);
      if (!candidate.compute_feasibility()) return;
      objective  = candidate.get_objective();
      assignment = candidate.get_host_assignment();
    }
    // Publication uses the main solver's common incumbent gate and configured tolerances.
    population_.add_external_solution(assignment, objective, solution_origin_t::CPUFJ);
    CUOPT_LOG_INFO("PERSISTENT_LNS_CANDIDATE source=%s objective=%.17g",
                   origin,
                   problem_->get_user_obj_from_solver_obj(objective));
  }

  bool snapshot(std::vector<f_t>& papilo_assignment)
  {
    std::vector<f_t> assignment;
    f_t objective;
    std::lock_guard<std::mutex> lock(mutex_);
    // Consume rejected candidates so an invalid low objective cannot hide a later
    // valid incumbent. The population's validated cache is never changed here.
    // At most ten queued candidates plus the validated cache. Bound retries even
    // if a malformed cached assignment fails revalidation.
    for (int attempt = 0; attempt < 11; ++attempt) {
      if (!population_.take_lns_seed_candidate(assignment, objective, last_source_objective_))
        break;
      if (assignment.size() != static_cast<size_t>(problem_->n_variables) ||
          !std::all_of(
            assignment.begin(), assignment.end(), [](f_t x) { return std::isfinite(x); }))
        continue;
      RAFT_CUDA_TRY(cudaSetDevice(device_));
      solution_t<i_t, f_t> candidate(*problem_);
      candidate.assignment      = cuopt::device_copy(assignment, handle_.get_stream());
      const f_t bound_violation = candidate.compute_max_variable_violation();
      if (bound_violation > problem_->tolerances.integrality_tolerance ||
          !candidate.compute_feasibility())
        continue;
      if (bound_violation > f_t{0}) {
        // Normalize only this private, tolerance-feasible seed. Exact neighborhood
        // domains remain strict, and moving onto a bound must preserve the rows.
        candidate.clamp_within_bounds();
        if (!candidate.compute_feasibility()) continue;
      }
      objective = candidate.get_objective();
      if (!std::isfinite(objective) || objective >= last_source_objective_) continue;
      auto x = std::move(candidate.assignment);
      problem_->post_process_assignment(x, true, handle_.get_stream());
      papilo_assignment      = cuopt::host_copy(x, handle_.get_stream());
      last_source_objective_ = objective;
      CUOPT_LOG_INFO("PERSISTENT_LNS_SEED objective=%.17g",
                     problem_->get_user_obj_from_solver_obj(objective));
      return true;
    }
    return false;
  }

 private:
  population_t<i_t, f_t>& population_;
  raft::handle_t handle_;
  std::unique_ptr<problem_t<i_t, f_t>> problem_;
  std::mutex mutex_;
  f_t last_source_objective_{std::numeric_limits<f_t>::infinity()};
  int device_{0};
};
}  // namespace cuopt::mathematical_optimization::mip
