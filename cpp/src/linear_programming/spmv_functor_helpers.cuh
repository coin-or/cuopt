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

#include <mip/presolve/load_balanced_partition_helpers.cuh>
#include <mip/problem/problem.cuh>
#include "spmv_functor_kernels.cuh"

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t>
i_t get_id_offset(const std::vector<i_t>& bin_offsets, i_t degree_cutoff)
{
  return bin_offsets[ceil_log_2(degree_cutoff)];
}

template <typename i_t>
std::pair<i_t, i_t> get_id_range(const std::vector<i_t>& bin_offsets,
                                 i_t degree_beg,
                                 i_t degree_end)
{
  return std::make_pair(bin_offsets[ceil_log_2(degree_beg)],
                        bin_offsets[ceil_log_2(degree_end) + 1]);
}

template <typename i_t, typename f_t, i_t block_dim, typename OutputTransformer, typename view_t>
void spmv_heavy(managed_stream_pool& streams,
                view_t view,
                raft::device_span<f_t> input,
                OutputTransformer t,
                raft::device_span<f_t> tmp_out,
                const rmm::device_uvector<i_t>& heavy_items_vertex_ids,
                const rmm::device_uvector<i_t>& heavy_items_pseudo_block_ids,
                const rmm::device_uvector<i_t>& heavy_items_block_segments,
                const std::vector<i_t>& bin_offsets,
                i_t heavy_degree_cutoff,
                i_t num_blocks_heavy_items,
                bool dry_run = false)
{
  if (num_blocks_heavy_items != 0) {
    auto heavy_stream = streams.get_stream();
    // TODO : Check heavy_vars_block_segments size for profiling
    if (!dry_run) {
      auto heavy_items_beg_id = get_id_offset(bin_offsets, heavy_degree_cutoff);
      lb_spmv_heavy_kernel<i_t, f_t, block_dim>
        <<<num_blocks_heavy_items, block_dim, 0, heavy_stream>>>(
          heavy_items_beg_id,
          make_span(heavy_items_vertex_ids),
          make_span(heavy_items_pseudo_block_ids),
          heavy_degree_cutoff,
          view,
          input,
          tmp_out);
      auto num_heavy_items = bin_offsets.back() - heavy_items_beg_id;
      finalize_spmv_kernel<i_t, f_t><<<num_heavy_items, 32, 0, heavy_stream>>>(
        heavy_items_beg_id, make_span(heavy_items_block_segments), tmp_out, view, t);
    }
  }
}

template <typename i_t, typename f_t, i_t block_dim, typename OutputTransformer, typename view_t>
void spmv_per_block(managed_stream_pool& streams,
                    view_t view,
                    raft::device_span<f_t> input,
                    OutputTransformer t,
                    const std::vector<i_t>& bin_offsets,
                    i_t degree_beg,
                    i_t degree_end,
                    bool dry_run)
{
  static_assert(block_dim <= 1024, "Cannot launch kernel with more than 1024 threads");

  auto [item_id_beg, item_id_end] = get_id_range(bin_offsets, degree_beg, degree_end);

  auto block_count = item_id_end - item_id_beg;
  if (block_count > 0) {
    // std::cout<<"spmv_per_block "<<item_id_beg<<" "<<item_id_end<<"\n";
    auto block_stream = streams.get_stream();
    if (!dry_run) {
      lb_spmv_block_kernel<i_t, f_t, block_dim>
        <<<block_count, block_dim, 0, block_stream>>>(item_id_beg, view, input, t);
    }
  }
}

template <typename i_t, typename f_t, typename OutputTransformer, typename view_t>
void spmv_per_block(managed_stream_pool& streams,
                    view_t view,
                    raft::device_span<f_t> input,
                    OutputTransformer t,
                    const std::vector<i_t>& bin_offsets,
                    i_t heavy_degree_cutoff,
                    bool dry_run = false)
{
  if (view.nnz < 10000) {
    spmv_per_block<i_t, f_t, 32>(streams, view, input, t, bin_offsets, 32, 32, dry_run);
    spmv_per_block<i_t, f_t, 64>(streams, view, input, t, bin_offsets, 64, 64, dry_run);
    spmv_per_block<i_t, f_t, 128>(streams, view, input, t, bin_offsets, 128, 128, dry_run);
    spmv_per_block<i_t, f_t, 256>(streams, view, input, t, bin_offsets, 256, 256, dry_run);
  } else {
    //[1024, heavy_degree_cutoff/2] -> 1024 block size
    spmv_per_block<i_t, f_t, 1024>(
      streams, view, input, t, bin_offsets, 1024, heavy_degree_cutoff / 2, dry_run);
    //[512, 512] -> 128 block size
    spmv_per_block<i_t, f_t, 128>(streams, view, input, t, bin_offsets, 128, 512, dry_run);
  }
}

template <typename i_t,
          typename f_t,
          i_t threads_per_constraint,
          typename OutputTransformer,
          typename view_t>
void spmv_sub_warp(managed_stream_pool& streams,
                   view_t view,
                   raft::device_span<f_t> input,
                   OutputTransformer t,
                   i_t degree_beg,
                   i_t degree_end,
                   const std::vector<i_t>& bin_offsets,
                   bool dry_run)
{
  constexpr i_t block_dim         = 32;
  auto items_per_block            = block_dim / threads_per_constraint;
  auto [item_id_beg, item_id_end] = get_id_range(bin_offsets, degree_beg, degree_end);

  auto block_count = raft::ceildiv<i_t>(item_id_end - item_id_beg, items_per_block);
  if (block_count != 0) {
    // std::cout<<"spmv_sub_warp "<<item_id_beg<<" "<<item_id_end<<"\n";
    auto sub_warp_thread = streams.get_stream();
    if (!dry_run) {
      lb_spmv_sub_warp_kernel<i_t, f_t, block_dim, threads_per_constraint>
        <<<block_count, block_dim, 0, sub_warp_thread>>>(item_id_beg, item_id_end, view, input, t);
    }
  }
}

template <typename i_t,
          typename f_t,
          i_t threads_per_constraint,
          typename OutputTransformer,
          typename view_t>
void spmv_sub_warp(managed_stream_pool& streams,
                   view_t view,
                   raft::device_span<f_t> input,
                   OutputTransformer t,
                   i_t degree,
                   const std::vector<i_t>& bin_offsets,
                   bool dry_run)
{
  spmv_sub_warp<i_t, f_t, threads_per_constraint>(
    streams, view, input, t, degree, degree, bin_offsets, dry_run);
}

template <typename i_t, typename f_t, typename OutputTransformer, typename view_t>
void spmv_sub_warp(managed_stream_pool& streams,
                   view_t view,
                   raft::device_span<f_t> input,
                   OutputTransformer t,
                   i_t item_sub_warp_count,
                   rmm::device_uvector<i_t>& warp_item_offsets,
                   rmm::device_uvector<i_t>& warp_item_id_offsets,
                   bool dry_run)
{
  constexpr i_t block_dim = 256;

  auto block_count = raft::ceildiv<i_t>(item_sub_warp_count * 32, block_dim);
  if (block_count != 0) {
    auto sub_warp_stream = streams.get_stream();
    if (!dry_run) {
      lb_spmv_sub_warp_kernel<i_t, f_t, block_dim><<<block_count, block_dim, 0, sub_warp_stream>>>(
        view, input, t, make_span(warp_item_offsets), make_span(warp_item_id_offsets));
    }
  }
}

template <typename i_t, typename f_t, typename OutputTransformer, typename view_t>
void spmv_sub_warp(managed_stream_pool& streams,
                   view_t view,
                   raft::device_span<f_t> input,
                   OutputTransformer t,
                   bool is_sub_warp_single_bin,
                   i_t item_sub_warp_count,
                   rmm::device_uvector<i_t>& warp_item_offsets,
                   rmm::device_uvector<i_t>& warp_item_id_offsets,
                   const std::vector<i_t>& bin_offsets,
                   bool dry_run = false)
{
  if (view.nnz < 10000) {
    // std::cout<<"spmv_sub_warp < 10k\n";
    spmv_sub_warp<i_t, f_t, 16>(streams, view, input, t, 16, bin_offsets, dry_run);
    spmv_sub_warp<i_t, f_t, 8>(streams, view, input, t, 8, bin_offsets, dry_run);
    spmv_sub_warp<i_t, f_t, 4>(streams, view, input, t, 4, bin_offsets, dry_run);
    spmv_sub_warp<i_t, f_t, 2>(streams, view, input, t, 2, bin_offsets, dry_run);
    spmv_sub_warp<i_t, f_t, 1>(streams, view, input, t, 1, bin_offsets, dry_run);
  } else {
    if (is_sub_warp_single_bin) {
      // std::cout<<"spmv_sub_warp single_bin\n";
      spmv_sub_warp<i_t, f_t, 16>(streams, view, input, t, 64, bin_offsets, dry_run);
      spmv_sub_warp<i_t, f_t, 8>(streams, view, input, t, 32, bin_offsets, dry_run);
      spmv_sub_warp<i_t, f_t, 4>(streams, view, input, t, 16, bin_offsets, dry_run);
      spmv_sub_warp<i_t, f_t, 2>(streams, view, input, t, 8, bin_offsets, dry_run);
      spmv_sub_warp<i_t, f_t, 1>(streams, view, input, t, 1, 4, bin_offsets, dry_run);
    } else {
      // std::cout<<"spmv_sub_warp multi_bin\n";
      spmv_sub_warp<i_t, f_t>(streams,
                              view,
                              input,
                              t,
                              item_sub_warp_count,
                              warp_item_offsets,
                              warp_item_id_offsets,
                              dry_run);
    }
  }
}
}  // namespace cuopt::linear_programming::detail
