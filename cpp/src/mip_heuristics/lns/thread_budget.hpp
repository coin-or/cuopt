/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/mip_constants.hpp>

#include <algorithm>

namespace cuopt::mathematical_optimization::mip {
inline constexpr bool presolve_cpufj_lns_enabled    = true;
inline constexpr bool presolve_repair_lns_enabled   = false;
inline constexpr bool persistent_cpufj_lns_enabled  = false;
inline constexpr bool persistent_repair_lns_enabled = false;

// Benchmark one LNS worker while returning disabled workers to the main solve.
inline int lns_worker_count(int team_size, bool deterministic)
{
  return deterministic || team_size < CUOPT_MIP_EARLY_CPUFJ_RESERVED_THREADS + 3
           ? 0
           : int(persistent_cpufj_lns_enabled) + int(persistent_repair_lns_enabled);
}

// This is a split of the existing early CPUFJ budget, not extra pool capacity.
// Preserve at least one feasibility lane and the caller's presolve reservation.
inline int presolve_lns_worker_count(int early_worker_budget)
{
  return early_worker_budget >= 3
           ? int(presolve_cpufj_lns_enabled) + int(presolve_repair_lns_enabled)
           : 0;
}

// Fill the capacity remaining after presolve and the enabled auxiliary workers.
// Papilo's reservation includes its calling OpenMP worker; cuOpt presolve instead
// reserves capacity for its caller and probing tasks.
inline int presolve_early_worker_budget(int team_size,
                                        int presolve_workers,
                                        int gpufj_workers,
                                        int structural_workers)
{
  // Preserve the original small-team eligibility for the presolve LNS workers.
  const int minimum = team_size >= CUOPT_MIP_EARLY_CPUFJ_RESERVED_THREADS + 3 ? 3 : 1;
  return std::max(minimum, team_size - presolve_workers - gpufj_workers - structural_workers);
}

inline bool early_structural_has_capacity(int team_size, int cpufj_workers, int gpufj_workers)
{
  return team_size > cpufj_workers + gpufj_workers + 1;
}

inline int papilo_thread_budget(int team_size,
                                int cpufj_workers,
                                int gpufj_workers,
                                int structural_workers)
{
  return std::clamp(team_size - cpufj_workers - gpufj_workers - structural_workers,
                    1,
                    CUOPT_MIP_PAPILO_THREAD_LIMIT);
}

inline int probing_thread_budget(int team_size, int cpufj_workers, int structural_workers)
{
  return std::max(1, team_size - 1 - cpufj_workers - structural_workers);
}

}  // namespace cuopt::mathematical_optimization::mip
