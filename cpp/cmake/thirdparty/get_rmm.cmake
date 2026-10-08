# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

function(find_and_configure_rmm)
    include(${rapids-cmake-dir}/cpm/rmm.cmake)
    # Build export set only: the installed config finds rmm itself, and only when a GPU
    # component is installed (see the FINAL_CODE_BLOCK in cpp/CMakeLists.txt).
    rapids_cpm_rmm(BUILD_EXPORT_SET cuopt-exports)
endfunction()

find_and_configure_rmm()
