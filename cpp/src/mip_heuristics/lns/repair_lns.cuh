/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <utilities/timer.hpp>
#include <utilities/type_2.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace raft {
class handle_t;
}

namespace cuopt::mathematical_optimization {
enum class var_t;
}

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct fj_cpu_climber_t;
template <typename i_t, typename f_t>
struct fj_cpu_problem_t;

enum class lns_repair_backend_t { cpufj, submip };

template <typename f_t>
std::vector<f_t> lns_integer_values(f_t lower, f_t upper)
{
  std::vector<f_t> values;
  for (f_t value = lower; value <= upper;) {
    values.push_back(value);
    // Original model domains can exceed the consecutive-integer range of f_t.
    // Advance to the next representable integer if adding one rounds back down.
    value = std::max(value + f_t{1}, std::nextafter(value, std::numeric_limits<f_t>::infinity()));
  }
  return values;
}

// Equal lower and upper bounds fix a variable. Bounds must lie within the model's domains.
template <typename f_t>
struct lns_repair_request_t {
  std::vector<f_t> start, lower, upper;
  f_t time_limit_seconds = 0.1;
  uint64_t seed          = 0;
  int max_iterations     = 10000;
  int max_nodes          = 500;
};

template <typename f_t>
struct lns_repair_result_t {
  bool feasible = false;
  std::vector<f_t> assignment;
  f_t objective = std::numeric_limits<f_t>::infinity();
};

// The neighborhood with fixed columns eliminated into the row bounds.
template <typename i_t, typename f_t>
struct lns_neighborhood_t {
  std::vector<i_t> offsets{0}, columns, free_columns;
  std::vector<f_t> coefficients, lower, upper, objective, row_lower, row_upper, row_tolerances;
  std::vector<var_t> types;
  std::vector<f_t> full, start;
  bool possible = true;
};

// Ruin-and-repair improvement search on the host problem of a CPUFJ climber. Every trajectory
// starts from a validated feasible seed, and only validated improvements are submitted.
template <typename i_t, typename f_t>
class repair_lns_t {
 public:
  using seed_fn   = std::function<void(std::vector<std::vector<f_t>>&)>;
  using submit_fn = std::function<void(const std::vector<f_t>&, f_t)>;

  repair_lns_t(const fj_cpu_climber_t<i_t, f_t>& anchor,
               std::atomic<bool>& preemption,
               uint64_t seed,
               cuopt::timer_t timer = cuopt::timer_t(std::numeric_limits<double>::infinity()));

  // Population seeds use the solver model's domains. CPUFJ may cap integer bounds or
  // strengthen continuous variables to integers, excluding otherwise feasible seeds.
  // Keep those domains separately while sharing the anchor's immutable matrix.
  repair_lns_t(const fj_cpu_climber_t<i_t, f_t>& anchor,
               std::vector<typename type_2<f_t>::type> bounds,
               std::vector<var_t> types,
               std::atomic<bool>& preemption,
               uint64_t seed,
               cuopt::timer_t timer = cuopt::timer_t(std::numeric_limits<double>::infinity()));

  std::atomic<bool> halted{false};

  bool feasible(const std::vector<f_t>& x) const;
  f_t cost(const std::vector<f_t>& x) const;

  lns_neighborhood_t<i_t, f_t> make_neighborhood(const lns_repair_request_t<f_t>& request) const;

  // Synchronous and single-threaded. The handle is only used by the sub-MIP backend.
  lns_repair_result_t<f_t> repair(const lns_repair_request_t<f_t>& request,
                                  lns_repair_backend_t backend,
                                  const raft::handle_t* handle);

  // Runs until halted or preempted. The calling thread must have the solver's device set.
  void run(const seed_fn& seeds, const submit_fn& submit);

 private:
  struct bandit_arm_t;
  struct bound_change_t;
  struct search_state_t;
  struct backend_config_t;
  struct branch_state_t;

  void consider_repair_candidate(const lns_repair_request_t<f_t>& request,
                                 const lns_neighborhood_t<i_t, f_t>& nb,
                                 const std::vector<f_t>& reduced,
                                 const cuopt::timer_t& timer,
                                 lns_repair_result_t<f_t>& result) const;
  size_t refresh_scores(const std::vector<f_t>& current,
                        const std::vector<i_t>& movable,
                        const std::vector<std::vector<std::pair<i_t, f_t>>>& column_rows,
                        std::vector<std::pair<double, i_t>>& variable_scores) const;
  void run_backend(search_state_t& state,
                   const std::vector<f_t>& lo_bounds,
                   const std::vector<f_t>& hi_bounds,
                   const backend_config_t& config);
  bool propagate(branch_state_t& episode) const;
  void rollback(branch_state_t& episode, size_t mark) const;
  void search(search_state_t& state, branch_state_t& episode);

  bool stopped() const { return halted.load() || preemption_.load() || timer_.check_time_limit(); }
  bool is_integer_var(i_t j) const;
  f_t lower(i_t j) const { return get_lower(bounds_[j]); }
  f_t upper(i_t j) const { return get_upper(bounds_[j]); }

  std::shared_ptr<const fj_cpu_problem_t<i_t, f_t>> problem_;
  std::vector<typename type_2<f_t>::type> bounds_;
  std::vector<var_t> types_;
  std::vector<f_t> row_tolerances_;
  std::atomic<bool>& preemption_;
  uint64_t seed_;
  cuopt::timer_t timer_;
};

}  // namespace cuopt::mathematical_optimization::mip
