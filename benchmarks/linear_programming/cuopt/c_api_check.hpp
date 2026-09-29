/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuopt/mathematical_optimization/cuopt_c.h>

#include <stdexcept>
#include <string>

namespace cuopt_bench {
class c_api_error_t : public std::runtime_error {
 public:
  c_api_error_t(cuopt_int_t code, const char* operation)
    : std::runtime_error(std::string(operation) + " failed with code " + std::to_string(code)),
      code(code),
      operation(operation)
  {
  }
  const cuopt_int_t code;
  const char* const operation;
};

inline void check_c_api(cuopt_int_t code, const char* operation)
{
  if (code != CUOPT_SUCCESS) { throw c_api_error_t(code, operation); }
}
}  // namespace cuopt_bench
