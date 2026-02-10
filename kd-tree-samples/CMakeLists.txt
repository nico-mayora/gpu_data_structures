cmake_minimum_required(VERSION 3.27.8)
project(kd-tree_samples CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
include(cmake/embed_ptx.cmake)

set(glfw_dir ${PROJECT_SOURCE_DIR}/common/externals/glfw)
set(OptiX_ROOT_DIR $ENV{OptiX_INSTALL_DIR})
set(CMAKE_PREFIX_PATH $ENV{OptiX_INSTALL_DIR})

# Path Tracer

embed_ptx(
        OUTPUT_TARGET
            pathTracer_ptx
        PTX_LINK_LIBRARIES
            owl::owl
        SOURCES
            ray-tracer/cuda/pathTracer.cu
)

set(pathTracer_sources
        ray-tracer/main.cu
        common/data/loader/mitsuba3.cu
        common/data/pt-math.cu
        ray-tracer/viewer.cu
        common/data/photon/photon-file-manager.cu
)

add_executable(pathTracer ${pathTracer_sources})

# Photon Emitter

embed_ptx(
        OUTPUT_TARGET
            photonEmitter_ptx
        PTX_LINK_LIBRARIES
            owl::owl
        SOURCES
            photon-mapper/cuda/photonEmitter.cu
)

set(photonEmitter_sources
        photon-mapper/main.cu
        common/data/loader/mitsuba3.cu
        common/data/pt-math.cu
        common/data/photon/photon-file-manager.cu
)

add_executable(photonEmitter ${photonEmitter_sources})

set(owl_dir ${PROJECT_SOURCE_DIR}/common/externals/owl)
include_directories(
        ${glfw_dir}/include
        ${owl_dir}/samples/common # For OWLViewer
)
add_subdirectory(${glfw_dir})

add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

set(tinyxml2_dir ${PROJECT_SOURCE_DIR}/common/externals/tinyxml2)
add_subdirectory(${tinyxml2_dir} EXCLUDE_FROM_ALL)

target_link_libraries(pathTracer PRIVATE pathTracer_ptx owl::owl owl_viewer tinyxml2::tinyxml2)
target_link_libraries(photonEmitter PRIVATE photonEmitter_ptx owl::owl tinyxml2::tinyxml2)

add_executable(kdtree_benchmark
        benchmark/main.cu
        common/data/photon/photon-file-manager.cu
)
target_include_directories(kdtree_benchmark PRIVATE
        ${PROJECT_SOURCE_DIR}/common/externals/owl
        ${PROJECT_SOURCE_DIR}/common/externals/owl/owl/include
)

# FIXME: If owl is not found, copy the .dll/.so into the executable folder.