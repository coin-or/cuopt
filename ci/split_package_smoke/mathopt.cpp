// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuopt/mathematical_optimization/cuopt_c.h>

#include <cmath>
#include <cstdio>

// minimize -0.2*x1 + 0.1*x2  s.t.  3*x1 + 4*x2 <= 5.4,  2.7*x1 + 10.1*x2 <= 4.9,  x >= 0
int main()
{
  cuopt_int_t row_offsets[]    = {0, 2, 4};
  cuopt_int_t column_indices[] = {0, 1, 0, 1};
  cuopt_float_t values[]       = {3.0, 4.0, 2.7, 10.1};
  cuopt_float_t objective[]    = {-0.2, 0.1};
  cuopt_float_t con_lower[]    = {-CUOPT_INFINITY, -CUOPT_INFINITY};
  cuopt_float_t con_upper[]    = {5.4, 4.9};
  cuopt_float_t var_lower[]    = {0.0, 0.0};
  cuopt_float_t var_upper[]    = {CUOPT_INFINITY, CUOPT_INFINITY};
  char var_types[]             = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  cuOptOptimizationProblem problem = nullptr;
  cuOptSolverSettings settings     = nullptr;
  cuOptSolution solution           = nullptr;
  cuopt_int_t termination          = -1;
  cuopt_float_t objective_value    = 0.0;

  cuopt_int_t status = cuOptCreateRangedProblem(2,
                                                2,
                                                CUOPT_MINIMIZE,
                                                0.0,
                                                objective,
                                                row_offsets,
                                                column_indices,
                                                values,
                                                con_lower,
                                                con_upper,
                                                var_lower,
                                                var_upper,
                                                var_types,
                                                &problem);
  if (status == CUOPT_SUCCESS) status = cuOptCreateSolverSettings(&settings);
  if (status == CUOPT_SUCCESS) status = cuOptSolve(problem, settings, &solution);
  if (status == CUOPT_SUCCESS) status = cuOptGetTerminationStatus(solution, &termination);
  if (status == CUOPT_SUCCESS) status = cuOptGetObjectiveValue(solution, &objective_value);

  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  std::printf("status=%d termination=%d objective=%f\n", status, termination, objective_value);
  return status == CUOPT_SUCCESS && termination == CUOPT_TERMINATION_STATUS_OPTIMAL &&
             std::fabs(objective_value + 0.36) < 1e-3
           ? 0
           : 1;
}
