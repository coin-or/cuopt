// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// libcuopt-client only: host-side parsing and problem representation, no GPU needed.
#include <cuopt/mathematical_optimization/io/parser.hpp>

#include <gtest/gtest.h>

namespace io = cuopt::mathematical_optimization::io;

static constexpr const char* kTinyLp = R"(NAME          TINY
ROWS
 N  obj
 L  c1
COLUMNS
    x         obj       -1.0   c1        1.0
    y         obj       -1.0   c1        1.0
RHS
    rhs       c1        4.0
BOUNDS
 UP bnd       x         3.0
 UP bnd       y         3.0
ENDATA
)";

TEST(IsolatedClient, ParsesMpsFromString)
{
  auto model = io::read_mps_from_string<int, double>(kTinyLp);
  EXPECT_EQ(model.get_n_variables(), 2);
  EXPECT_EQ(model.get_n_constraints(), 1);
  EXPECT_EQ(model.get_nnz(), 2);
}

TEST(IsolatedClient, MalformedMpsThrows)
{
  EXPECT_ANY_THROW((io::read_mps_from_string<int, double>("not an mps file")));
}
