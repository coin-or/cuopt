/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <exception>
#include <mutex>

namespace cuopt::lns {

// Exceptions cannot leave an OpenMP task. LNS tasks capture the first one here and the owner
// of the OpenMP team rethrows it after the team has joined.
class task_errors_t {
 public:
  void capture(std::exception_ptr error)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_) error_ = std::move(error);
  }

  void rethrow_if_error() const
  {
    if (error_) std::rethrow_exception(error_);
  }

 private:
  std::mutex mutex_;
  std::exception_ptr error_;
};

}  // namespace cuopt::lns
