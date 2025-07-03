cmake_minimum_required(VERSION 3.27)
project(kd_tree_cuda CUDA)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD  23)

# Automatically find all .cu files in src/
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

add_executable(kd_tree_cuda
        main.cu
        ${CUDA_SOURCES}
)

target_include_directories(kd_tree_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(kd_tree_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

add_executable(test_correctness
        test_correctness.cu
        ${CUDA_SOURCES}
)

target_include_directories(test_correctness PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(test_correctness PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

add_executable(test_performance
        test_performance.cu
        ${CUDA_SOURCES}
)

target_include_directories(test_performance PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(test_performance PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)