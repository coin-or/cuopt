/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/io/data_model_view.hpp>
#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

#include <string>

namespace cuopt::mathematical_optimization::io {

/**
 * @brief Writes an optimization problem to an LP formatted file
 *
 * Writes the problem in the LP format understood by read_lp(). The emitted
 * dialect is a superset of what a typical solver expects and round-trips
 * through read_lp():
 *   - Minimize / Maximize objective (with optional quadratic '[ ... ] / 2' term)
 *   - Subject To constraints (linear, plus optional quadratic '[ ... ]' term)
 *   - Bounds
 *   - Generals / Binaries / Semi-Continuous variable sections
 *
 * Notes / limitations that mirror the LP format itself:
 *   - Range rows (a linear row with two distinct finite bounds) are emitted as
 *     two constraints ('_lo' with '>=' and '_up' with '<=') because the LP
 *     format has no single-line ranged-row syntax.
 *   - The objective scaling factor is not representable in LP format and is
 *     therefore not written (identical to the MPS writer).
 *
 * @tparam i_t  data type of the indices
 * @tparam f_t  data type of the weights and variables
 *
 * @param[in] problem Data model view to write
 * @param[in] lp_file_path Path to the LP file to write
 */
template <typename i_t, typename f_t>
void write_lp(const data_model_view_t<i_t, f_t>& problem, const std::string& lp_file_path);

/**
 * @brief Writes an optimization problem to an LP formatted file
 *
 * Same as the data_model_view_t overload.
 *
 * @param[in] problem Data model to write
 * @param[in] lp_file_path Path to the LP file to write
 */
template <typename i_t, typename f_t>
void write_lp(const mps_data_model_t<i_t, f_t>& problem, const std::string& lp_file_path);

}  // namespace cuopt::mathematical_optimization::io
