/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace cuopt::hive_lns {
// Repair a private neighborhood: equal lower/upper bounds fix a variable.
// Bounds must lie within the model's domains. Start may be infeasible.
struct repair_request_t {
  std::vector<double> start, lower, upper;
  double time_limit_seconds = 0.1;
  uint64_t seed             = 0;
  int max_iterations        = 10000;
  int max_nodes             = 500;
};
struct repair_result_t {
  bool feasible = false;
  std::vector<double> assignment;  // full model coordinates; empty on failure/timeout
  double objective       = std::numeric_limits<double>::infinity();
  double elapsed_seconds = 0;
};
using repair_fn = std::function<repair_result_t(const repair_request_t&)>;
struct repair_tools_t {
  // Synchronous, single-threaded backends on the LNS worker. They never publish
  // incumbents or mutate population. Time includes setup and is capped by the
  // remaining solve time. An empty result is normal if no repair is found.
  // Every returned assignment passes the fixed model and neighborhood checks.
  repair_fn cpufj;
  repair_fn submip;
};

// Frozen, owning CPU view. All objective coefficients use minimization convention.
struct model_t {
  std::vector<int> offsets, columns;
  std::vector<double> coefficients, lower, upper, row_lower, row_upper, objective;
  // Computed by the bridge with the solver's get_cstr_tolerance helper.
  std::vector<double> row_tolerances;
  std::vector<bool> integer;
  repair_tools_t repair;
  double feasibility_tolerance = 1e-6;
  double relative_tolerance    = 1e-12;
  double integrality_tolerance = 1e-5;

  double cost(const std::vector<double>& x) const
  {
    double value = 0, correction = 0;
    for (size_t j = 0; j < x.size(); ++j) {
      double term = objective[j] * x[j] - correction;
      double next = value + term;
      correction  = (next - value) - term;
      value       = next;
    }
    return value;
  }
  bool feasible(const std::vector<double>& x) const
  {
    if (x.size() != lower.size() || row_tolerances.size() != row_lower.size()) return false;
    for (size_t j = 0; j < x.size(); ++j) {
      if (!std::isfinite(x[j]) || x[j] < lower[j] - integrality_tolerance ||
          x[j] > upper[j] + integrality_tolerance ||
          (integer[j] && std::abs(x[j] - std::round(x[j])) > integrality_tolerance))
        return false;
    }
    for (size_t r = 0; r < row_lower.size(); ++r) {
      double value = 0, correction = 0;
      for (int p = offsets[r]; p < offsets[r + 1]; ++p) {
        double term = coefficients[p] * x[columns[p]] - correction;
        double next = value + term;
        correction  = (next - value) - term;
        value       = next;
      }
      if (!std::isfinite(value) || value < row_lower[r] - row_tolerances[r] ||
          value > row_upper[r] + row_tolerances[r])
        return false;
    }
    return std::isfinite(cost(x));
  }

  // Only normalize a private seed, never a population member. A tolerance-feasible
  // assignment need not be a legal exact fixing; rounding can also invalidate rows.
  bool normalize_seed(std::vector<double>& x) const
  {
    if (!feasible(x)) return false;
    for (size_t j = 0; j < x.size(); ++j) {
      const double lo = integer[j] ? std::ceil(lower[j]) : lower[j];
      const double hi = integer[j] ? std::floor(upper[j]) : upper[j];
      if (lo > hi) return false;
      x[j] = std::clamp(integer[j] ? std::round(x[j]) : x[j], lo, hi);
    }
    return feasible(x);
  }
};
using population_t = std::vector<std::vector<double>>;
using snapshot_fn  = std::function<population_t()>;
using submit_fn    = std::function<void(const std::vector<double>&)>;
using stop_fn      = std::function<bool()>;
using run_lns_fn =
  void (*)(const model_t&, const snapshot_fn&, const submit_fn&, const stop_fn&, uint64_t);
}  // namespace cuopt::hive_lns
