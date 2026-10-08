# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

function(find_and_configure_cccl)
        include(${rapids-cmake-dir}/cpm/cccl.cmake)
        # Build export set only: the installed config finds CCCL itself, and only when a GPU
        # component is installed (see the FINAL_CODE_BLOCK in cpp/CMakeLists.txt). The conda
        # client package does not ship CCCL's CMake files, so requiring it would break client-only.
        rapids_cpm_cccl(BUILD_EXPORT_SET cuopt-exports)
endfunction()

find_and_configure_cccl()
