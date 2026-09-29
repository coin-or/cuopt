/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/mip_constants.hpp>

namespace cuopt::mathematical_optimization::mip {
// Persistent LNS tasks share the solve's OpenMP team. Leave the existing eight-thread
// feasibility portfolio available, then enable Hive LNS and CPUFJ LNS in that order.
inline int lns_worker_count(int team_size, bool deterministic)
{
  if (deterministic || team_size <= CUOPT_MIP_FJ_REQUIRED_THREAD_COUNT) return 0;
  return team_size == CUOPT_MIP_FJ_REQUIRED_THREAD_COUNT + 1 ? 1 : 2;
}

// This is a split of the existing early CPUFJ budget, not extra pool capacity.
// Preserve at least one feasibility lane and the caller's presolve reservation.
inline int presolve_lns_worker_count(int early_worker_budget)
{
  return early_worker_budget >= 3 ? 2 : 0;
}

}  // namespace cuopt::mathematical_optimization::mip
