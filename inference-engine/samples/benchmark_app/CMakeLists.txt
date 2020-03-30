# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set(TARGET_NAME "benchmark_app")

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_sample(NAME benchmark_app
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES format_reader
              OPENCV_DEPENDENCIES imgcodecs)

find_package(ngraph REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE ${NGRAPH_LIBRARIES} onnx_importer)