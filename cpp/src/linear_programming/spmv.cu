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

#include <mip/mip_constants.hpp>
#include <linear_programming/pdlp.cuh>
#include "spmv.cuh"
#include "spmv_helpers.cuh"
#include "spmv_setup_helpers.cuh"

#include <nvtx3/nvtx3.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
spmv_t<i_t, f_t>::spmv_t(problem_t<i_t, f_t>& problem_,
                         //raft::device_span<f_t> ax_input_,
                         //raft::device_span<f_t> ax_output_,
                         //raft::device_span<f_t> aty_input_,
                         //raft::device_span<f_t> aty_output_,
                         //raft::device_span<f_t> aty_next_input_,
                         //raft::device_span<f_t> aty_next_output_,
                         bool debug)
  : streams(16),
    pb(&problem_),
    handle_ptr(pb->handle_ptr),
    n_constraints(pb->n_constraints),
    n_variables(pb->n_variables),
    nnz(pb->nnz),
    cnst_reorg_ids(n_constraints, handle_ptr->get_stream()),
    coefficients(nnz, handle_ptr->get_stream()),
    variables(nnz, handle_ptr->get_stream()),
    offsets(n_constraints + 1, handle_ptr->get_stream()),
    vars_reorg_ids(n_variables, handle_ptr->get_stream()),
    reverse_coefficients(nnz, handle_ptr->get_stream()),
    reverse_constraints(nnz, handle_ptr->get_stream()),
    reverse_offsets(n_variables + 1, handle_ptr->get_stream()),
    tmp_cnst_ids(n_constraints, handle_ptr->get_stream()),
    tmp_vars_ids(n_variables, handle_ptr->get_stream()),
    heavy_cnst_block_segments(0, problem_.handle_ptr->get_stream()),
    heavy_vars_block_segments(0, problem_.handle_ptr->get_stream()),
    heavy_cnst_vertex_ids(0, problem_.handle_ptr->get_stream()),
    heavy_vars_vertex_ids(0, problem_.handle_ptr->get_stream()),
    heavy_cnst_pseudo_block_ids(0, problem_.handle_ptr->get_stream()),
    heavy_vars_pseudo_block_ids(0, problem_.handle_ptr->get_stream()),
    num_blocks_heavy_cnst(0),
    num_blocks_heavy_vars(0),
    tmp_ax(0, problem_.handle_ptr->get_stream()),
    tmp_aty(0, problem_.handle_ptr->get_stream()),
    warp_cnst_offsets(0, problem_.handle_ptr->get_stream()),
    warp_cnst_id_offsets(0, problem_.handle_ptr->get_stream()),
    block_cnst_offsets(0, problem_.handle_ptr->get_stream()),
    block_cnst_id_offsets(0, problem_.handle_ptr->get_stream()),
    warp_vars_offsets(0, problem_.handle_ptr->get_stream()),
    warp_vars_id_offsets(0, problem_.handle_ptr->get_stream()),
    cnst_binner(handle_ptr),
    vars_binner(handle_ptr),
    //ax_input(ax_input_),
    //ax_output(ax_output_),
    //aty_input(aty_input_),
    //aty_output(aty_output_),
    //aty_next_input(aty_next_input_),
    //aty_next_output(aty_next_output_),
    ax_graph_created(false),
    aty_graph_created(false),
    aty_next_graph_created(false),
    ax_exec(nullptr),
    aty_exec(nullptr),
    aty_next_exec(nullptr)
{
  setup_lb_problem(problem_, debug);
  setup_lb_meta();
}
template <typename i_t, typename f_t>
spmv_t<i_t, f_t>::~spmv_t()
{
  if (ax_graph_created) { cudaGraphExecDestroy(ax_exec); }
  if (aty_graph_created) { cudaGraphExecDestroy(aty_exec); }
  if (aty_next_graph_created) { cudaGraphExecDestroy(aty_next_exec); }
}

template <typename i_t>
i_t lb(i_t i) {
  if (i < 2) {
    return 0;
  } else {
    return (1<<(i-2));
  }
}

template <typename i_t>
i_t ub(i_t i) {
  if (i == 0) {
    return 0;
  } else {
    return (1<<(i-1));
  }
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::setup_lb_problem(problem_t<i_t, f_t>& problem_, bool debug)
{
  pb            = &problem_;
  handle_ptr    = pb->handle_ptr;
  n_constraints = pb->n_constraints;
  n_variables   = pb->n_variables;
  nnz           = pb->nnz;
  cnst_reorg_ids.resize(n_constraints, handle_ptr->get_stream());
  coefficients.resize(nnz, handle_ptr->get_stream());
  variables.resize(nnz, handle_ptr->get_stream());
  offsets.resize(n_constraints + 1, handle_ptr->get_stream());
  vars_reorg_ids.resize(n_variables, handle_ptr->get_stream());
  reverse_coefficients.resize(nnz, handle_ptr->get_stream());
  reverse_constraints.resize(nnz, handle_ptr->get_stream());
  reverse_offsets.resize(n_variables + 1, handle_ptr->get_stream());
  tmp_cnst_ids.resize(n_constraints, handle_ptr->get_stream());
  tmp_vars_ids.resize(n_variables, handle_ptr->get_stream());

  // RAFT_CHECK_CUDA(stream.synchronize());
  // std::cerr<<"pt 0\n";

  cnst_binner.setup(pb->offsets.data(), nullptr, 0, n_constraints);
  auto dist_cnst = cnst_binner.run(tmp_cnst_ids, handle_ptr);
  vars_binner.setup(pb->reverse_offsets.data(), nullptr, 0, n_variables);
  auto dist_vars = vars_binner.run(tmp_vars_ids, handle_ptr);

  // RAFT_CHECK_CUDA(stream.synchronize());
  // std::cerr<<"pt 1\n";

  auto cnst_bucket = dist_cnst.degree_range();
  auto vars_bucket = dist_vars.degree_range();

  cnst_reorg_ids.resize(cnst_bucket.vertex_ids.size(), handle_ptr->get_stream());
  vars_reorg_ids.resize(vars_bucket.vertex_ids.size(), handle_ptr->get_stream());

  // RAFT_CHECK_CUDA(stream.synchronize());
  // std::cerr<<"pt 2\n";

  raft::copy(cnst_reorg_ids.data(),
             cnst_bucket.vertex_ids.data(),
             cnst_bucket.vertex_ids.size(),
             handle_ptr->get_stream());
  raft::copy(vars_reorg_ids.data(),
             vars_bucket.vertex_ids.data(),
             vars_bucket.vertex_ids.size(),
             handle_ptr->get_stream());

  // RAFT_CHECK_CUDA(stream.synchronize());
  // std::cerr<<"pt 3\n";

  create_graph<i_t, f_t>(handle_ptr,
                         cnst_reorg_ids,
                         offsets,
                         coefficients,
                         variables,
                         problem_.offsets,
                         problem_.coefficients,
                         problem_.variables,
                         debug);

  // RAFT_CHECK_CUDA(stream.synchronize());
  // std::cerr<<"pt 4\n";

  create_graph<i_t, f_t>(handle_ptr,
                         vars_reorg_ids,
                         reverse_offsets,
                         reverse_coefficients,
                         reverse_constraints,
                         problem_.reverse_offsets,
                         problem_.reverse_coefficients,
                         problem_.reverse_constraints,
                         debug);

  // RAFT_CHECK_CUDA(stream.synchronize());
  // std::cerr<<"pt 5\n";

  cnst_bin_offsets = dist_cnst.bin_offsets_;
  vars_bin_offsets = dist_vars.bin_offsets_;
  //if (nnz < 10000) {
  //  compact_bins(cnst_bin_offsets, n_constraints);
  //  compact_bins(vars_bin_offsets, n_variables);
  //}

  if (true) {
    std::cout<<"cnst_bin_offsets\n";
    std::cout<<"bin\t(lb\tub]\tbeg_off\tcount\n";
    for (int i = 0; i < static_cast<i_t>(cnst_bin_offsets.size()); ++i) {
      if (i > 0) {
        std::cout<<i<<"\t"<<lb(i)<<"\t"<<ub(i)<<"\t"<<cnst_bin_offsets[i]<<"\t"<<(cnst_bin_offsets[i] - cnst_bin_offsets[i-1])<<"\n";
      } else {
        std::cout<<i<<"\t"<<lb(i)<<"\t"<<ub(i)<<"\t"<<cnst_bin_offsets[i]<<"\tn/a\n";
      }
      if (cnst_bin_offsets[i] == cnst_bin_offsets.back()) {
        std::cout<<"\n";
        break;
      }
    }
  }
  if (false) {
    std::cout<<"vars_bin_offsets\n";
    for (int i = 0; i < static_cast<i_t>(vars_bin_offsets.size()); ++i) {
      std::cout<<i<<" "<<vars_bin_offsets[i]<<"\n";
    }
  }

  // RAFT_CHECK_CUDA(stream.synchronize());
  // std::cerr<<"pt 6\n";

  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::setup_lb_meta()
{
  auto stream = handle_ptr->get_stream();
  stream.synchronize();
  ax_graph_created  = false;
  aty_graph_created = false;

  num_blocks_heavy_cnst = create_heavy_item_block_segments(stream,
                                                           heavy_cnst_vertex_ids,
                                                           heavy_cnst_pseudo_block_ids,
                                                           heavy_cnst_block_segments,
                                                           heavy_degree_cutoff,
                                                           cnst_bin_offsets,
                                                           offsets);

  num_blocks_heavy_vars = create_heavy_item_block_segments(stream,
                                                           heavy_vars_vertex_ids,
                                                           heavy_vars_pseudo_block_ids,
                                                           heavy_vars_block_segments,
                                                           heavy_degree_cutoff,
                                                           vars_bin_offsets,
                                                           reverse_offsets);

  tmp_ax.resize(num_blocks_heavy_cnst, stream);
  tmp_aty.resize(num_blocks_heavy_vars, stream);

  //std::tie(is_cnst_sub_warp_single_bin, cnst_sub_warp_count) =
  //  sub_warp_meta(stream, warp_cnst_offsets, warp_cnst_id_offsets, cnst_bin_offsets, 4, true);

  //std::tie(is_vars_sub_warp_single_bin, vars_sub_warp_count) =
  //  sub_warp_meta(stream, warp_vars_offsets, warp_vars_id_offsets, vars_bin_offsets, 4);

  //i_t w_t_r = 1;
  //std::cout<<"w_t_r "<<w_t_r<<"\n";
  //std::tie(num_sub_warp_blocks, num_medium_blocks) = block_meta(cnst_bin_offsets, w_t_r, 16*1024, true);
  i_t w_t_r = 4;
  //std::cout<<"w_t_r "<<w_t_r<<"\n";
  std::tie(cnst_sub_warp_count, cnst_med_block_count, cnst_heavy_beg_id) = block_meta(
      stream,
      warp_cnst_offsets, warp_cnst_id_offsets, 
      block_cnst_offsets, block_cnst_id_offsets, 
      cnst_bin_offsets, w_t_r, 16*1024, true);
  std::cout<<"num_blocks_heavy_cnst "<<num_blocks_heavy_cnst<<"\n";

  RAFT_CHECK_CUDA(stream.synchronize());
  stream.synchronize();
  streams.sync_all_issued();

  //if (!ax_graph_created) {
  //  ax_graph_created = build_graph(
  //    streams,
  //    handle_ptr,
  //    ax_graph,
  //    ax_exec,
  //    [&]() { this->call_Ax_graph(ax_input, ax_output, true); },
  //    [&]() { this->call_Ax_graph(ax_input, ax_output); });
  //}

  //streams.sync_all_issued();
  //if (!aty_graph_created) {
  //  aty_graph_created = build_graph(
  //    streams,
  //    handle_ptr,
  //    aty_graph,
  //    aty_exec,
  //    [this]() { this->call_ATy_graph(aty_input, aty_output, true); },
  //    [this]() { this->call_ATy_graph(aty_input, aty_output); });
  //}

  //streams.sync_all_issued();
  //if (!aty_next_graph_created) {
  //  aty_next_graph_created = build_graph(
  //    streams,
  //    handle_ptr,
  //    aty_next_graph,
  //    aty_next_exec,
  //    [this]() { this->call_ATy_graph(aty_next_input, aty_next_output, true); },
  //    [this]() { this->call_ATy_graph(aty_next_input, aty_next_output); });
  //}
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::Ax(const raft::handle_t* h, raft::device_span<f_t> input, raft::device_span<f_t> output)
{
  //raft::common::nvtx::range scope("ax");
  //cudaGraphLaunch(ax_exec, h->get_stream());
  spmv_call(h->get_stream(), get_A_view(),
      input, output,
      make_span(tmp_ax),
      cnst_sub_warp_count,
      cnst_med_block_count,
      num_blocks_heavy_cnst,
      cnst_heavy_beg_id,
      pb->n_constraints - cnst_heavy_beg_id,
      warp_cnst_offsets,
      warp_cnst_id_offsets,
      block_cnst_offsets,
      block_cnst_id_offsets, 
      heavy_cnst_vertex_ids,
      heavy_cnst_pseudo_block_ids,
      heavy_cnst_block_segments);
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::ATy(const raft::handle_t* h,
                           raft::device_span<f_t> input,
                           raft::device_span<f_t> output)
{
  //raft::common::nvtx::range scope("ay");
  //if (input.data() == aty_input.data() && output.data() == aty_output.data()) {
  //  cudaGraphLaunch(aty_exec, h->get_stream());
  //} else if (input.data() == aty_next_input.data() && output.data() == aty_next_output.data()) {
  //  cudaGraphLaunch(aty_next_exec, h->get_stream());
  //} else {
  //  std::cerr << "ATy unexpected call\n";
  //}
}

template <typename i_t, typename f_t>
typename spmv_t<i_t, f_t>::spmv_view_t spmv_t<i_t, f_t>::get_A_view()
{
  spmv_t::spmv_view_t v;
  v.reorg_ids = make_span(cnst_reorg_ids);
  v.coeff     = make_span(coefficients);
  v.elem      = make_span(variables);
  v.offsets   = make_span(offsets);
  v.nnz       = nnz;
  return v;
}

template <typename i_t, typename f_t>
typename spmv_t<i_t, f_t>::spmv_view_t spmv_t<i_t, f_t>::get_AT_view()
{
  spmv_t::spmv_view_t v;
  v.reorg_ids = make_span(vars_reorg_ids);
  v.coeff     = make_span(reverse_coefficients);
  v.elem      = make_span(reverse_constraints);
  v.offsets   = make_span(reverse_offsets);
  v.nnz       = nnz;
  return v;
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::call_Ax_graph(raft::device_span<f_t> input,
                                     raft::device_span<f_t> output,
                                     bool dry_run)
{
  // streams.sync_all_issued();
  //auto view = get_A_view();

  //spmv_heavy<i_t, f_t, 640>(streams,
  //                          view,
  //                          input,
  //                          output,
  //                          make_span(tmp_ax),
  //                          heavy_cnst_vertex_ids,
  //                          heavy_cnst_pseudo_block_ids,
  //                          heavy_cnst_block_segments,
  //                          cnst_bin_offsets,
  //                          heavy_degree_cutoff,
  //                          num_blocks_heavy_cnst,
  //                          dry_run);

  //spmv_per_block<i_t, f_t>(
  //  streams, view, input, output, cnst_bin_offsets, heavy_degree_cutoff, dry_run);
  //spmv_sub_warp<i_t, f_t>(streams,
  //                        view,
  //                        input,
  //                        output,
  //                        is_cnst_sub_warp_single_bin,
  //                        cnst_sub_warp_count,
  //                        warp_cnst_offsets,
  //                        warp_cnst_id_offsets,
  //                        cnst_bin_offsets,
  //                        dry_run);
  // streams.sync_all_issued();
}

template <typename i_t, typename f_t>
void spmv_t<i_t, f_t>::call_ATy_graph(raft::device_span<f_t> input,
                                      raft::device_span<f_t> output,
                                      bool dry_run)
{
  // auto r_id = host_copy(vars_reorg_ids);
  // auto r_off = host_copy(reverse_offsets);
  // for (int i = 0; i < r_id.size(); ++i) {
  //   if (r_id[i] < 5) {
  //     std::cout<<"r_off "<<i<<" r_id "<<r_id[i]<<" "<<r_off[i+1] - r_off[i]<<"\n";
  //   }
  // }
  // streams.sync_all_issued();
  //auto view = get_AT_view();

  //spmv_heavy<i_t, f_t, 640>(streams,
  //                          view,
  //                          input,
  //                          output,
  //                          make_span(tmp_aty),
  //                          heavy_vars_vertex_ids,
  //                          heavy_vars_pseudo_block_ids,
  //                          heavy_vars_block_segments,
  //                          vars_bin_offsets,
  //                          heavy_degree_cutoff,
  //                          num_blocks_heavy_vars,
  //                          dry_run);

  //spmv_per_block<i_t, f_t>(
  //  streams, view, input, output, vars_bin_offsets, heavy_degree_cutoff, dry_run);
  //spmv_sub_warp<i_t, f_t>(streams,
  //                        view,
  //                        input,
  //                        output,
  //                        is_vars_sub_warp_single_bin,
  //                        vars_sub_warp_count,
  //                        warp_vars_offsets,
  //                        warp_vars_id_offsets,
  //                        vars_bin_offsets,
  //                        dry_run);
  // streams.sync_all_issued();
}

#if MIP_INSTANTIATE_FLOAT
template class spmv_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class spmv_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
