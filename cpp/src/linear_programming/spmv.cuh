/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

// THIS IS LIKELY THE INNER-MOST INCLUDE
// FOR COMPILE TIME, WE SHOULD KEEP THE INCLUDES ON THIS HEADER MINIMAL

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>

#include <utilities/macros.cuh>

#include <mip/logger.hpp>
#include <mip/problem/problem.cuh>

#include <mip/presolve/load_balanced_partition_helpers.cuh>
#include <raft/core/nvtx.hpp>
#include <raft/random/rng_device.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include "managed_stream_pool.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class spmv_t {
 public:
  struct spmv_view_t {
    raft::device_span<const i_t> reorg_ids;
    raft::device_span<const i_t> offsets;
    raft::device_span<const i_t> elem;
    raft::device_span<const f_t> coeff;
    i_t nnz;
  };

  spmv_t(problem_t<i_t, f_t>& problem,
         //raft::device_span<f_t> ax_input_,
         //raft::device_span<f_t> ax_output_,
         //raft::device_span<f_t> aty_input_,
         //raft::device_span<f_t> aty_output_,
         //raft::device_span<f_t> aty_next_input_,
         //raft::device_span<f_t> aty_next_output_,
         bool debug = false);
  ~spmv_t();
  spmv_t() = delete;
  void setup_lb_problem(problem_t<i_t, f_t>& problem, bool debug = false);
  void setup_lb_meta();
  spmv_view_t get_A_view();
  spmv_view_t get_AT_view();
  void call_Ax_graph(raft::device_span<f_t> input,
                     raft::device_span<f_t> output,
                     bool dry_run = false);
  void call_ATy_graph(raft::device_span<f_t> input,
                      raft::device_span<f_t> output,
                      bool dry_run = false);

  void Ax(const raft::handle_t* h, raft::device_span<f_t> input, raft::device_span<f_t> output);
  void ATy(const raft::handle_t* h, raft::device_span<f_t> input, raft::device_span<f_t> output);

  managed_stream_pool streams;

  static constexpr i_t heavy_degree_cutoff = 16 * 1024;
  problem_t<i_t, f_t>* pb;
  const raft::handle_t* handle_ptr;

  i_t n_variables;
  i_t n_constraints;
  i_t nnz;

  // csr - cnst
  rmm::device_uvector<i_t> cnst_reorg_ids;
  rmm::device_uvector<f_t> coefficients;
  rmm::device_uvector<i_t> variables;
  rmm::device_uvector<i_t> offsets;

  // csc - vars
  rmm::device_uvector<i_t> vars_reorg_ids;
  rmm::device_uvector<f_t> reverse_coefficients;
  rmm::device_uvector<i_t> reverse_constraints;
  rmm::device_uvector<i_t> reverse_offsets;

  // lb members
  rmm::device_uvector<i_t> tmp_cnst_ids;
  rmm::device_uvector<i_t> tmp_vars_ids;

  // Number of blocks for heavy ids
  rmm::device_uvector<i_t> heavy_cnst_block_segments;
  rmm::device_uvector<i_t> heavy_vars_block_segments;
  rmm::device_uvector<i_t> heavy_cnst_vertex_ids;
  rmm::device_uvector<i_t> heavy_vars_vertex_ids;
  rmm::device_uvector<i_t> heavy_cnst_pseudo_block_ids;
  rmm::device_uvector<i_t> heavy_vars_pseudo_block_ids;

  i_t num_blocks_heavy_cnst;
  i_t num_blocks_heavy_vars;

  rmm::device_uvector<f_t> tmp_ax;
  rmm::device_uvector<f_t> tmp_aty;

  // lb sub-warp opt members
  i_t cnst_heavy_beg_id;
  bool is_cnst_sub_warp_single_bin;
  i_t cnst_sub_warp_count;
  i_t cnst_med_block_count;
  rmm::device_uvector<i_t> warp_cnst_offsets;
  rmm::device_uvector<i_t> warp_cnst_id_offsets;
  rmm::device_uvector<i_t> block_cnst_offsets;
  rmm::device_uvector<i_t> block_cnst_id_offsets;

  bool is_vars_sub_warp_single_bin;
  i_t vars_sub_warp_count;
  rmm::device_uvector<i_t> warp_vars_offsets;
  rmm::device_uvector<i_t> warp_vars_id_offsets;

  // binning
  std::vector<i_t> cnst_bin_offsets;
  std::vector<i_t> vars_bin_offsets;

  vertex_bin_t<i_t> cnst_binner;
  vertex_bin_t<i_t> vars_binner;

  raft::device_span<f_t> ax_input;
  raft::device_span<f_t> ax_output;

  raft::device_span<f_t> aty_input;
  raft::device_span<f_t> aty_output;

  raft::device_span<f_t> aty_next_input;
  raft::device_span<f_t> aty_next_output;

  // spmv graphs
  bool ax_graph_created;
  bool aty_graph_created;
  bool aty_next_graph_created;

  cudaGraphExec_t ax_exec;
  cudaGraph_t ax_graph;

  cudaGraphExec_t aty_exec;
  cudaGraph_t aty_graph;

  cudaGraphExec_t aty_next_exec;
  cudaGraph_t aty_next_graph;
};

}  // namespace cuopt::linear_programming::detail
