# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sample gtest that builds against ONE installed cuOpt component only.
#   cmake -S tests -B build -DCOMPONENT=<client|mathopt|routing> -DCMAKE_PREFIX_PATH=<...>
cmake_minimum_required(VERSION 3.30)

set(COMPONENT "" CACHE STRING "cuOpt component under test: client, mathopt or routing")
if(NOT COMPONENT MATCHES "^(client|mathopt|routing)$")
  message(FATAL_ERROR "COMPONENT must be client, mathopt or routing (got '${COMPONENT}')")
endif()

if(COMPONENT STREQUAL "client")
  project(isolated_${COMPONENT} LANGUAGES CXX)
else()
  project(isolated_${COMPONENT} LANGUAGES CXX CUDA)
endif()

find_package(cuopt CONFIG REQUIRED)
find_package(GTest REQUIRED)

if(NOT TARGET cuopt::${COMPONENT})
  message(FATAL_ERROR "Target cuopt::${COMPONENT} not provided by the installed package")
endif()

if(COMPONENT STREQUAL "routing")
  set(_src ${COMPONENT}/${COMPONENT}_test.cu)
else()
  set(_src ${COMPONENT}/${COMPONENT}_test.cpp)
endif()

add_executable(isolated_${COMPONENT}_test ${_src})
target_compile_features(isolated_${COMPONENT}_test PRIVATE cxx_std_20)
# Only the component under test (plus whatever it pulls in transitively) is linked.
target_link_libraries(isolated_${COMPONENT}_test PRIVATE cuopt::${COMPONENT}
                                                         GTest::gtest GTest::gtest_main)
