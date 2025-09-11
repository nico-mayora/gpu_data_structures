cmake_minimum_required(VERSION 3.27.8)
project(rtrt_photon-mapping CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
include(cmake/embed_ptx.cmake)

set(glfw_dir ${PROJECT_SOURCE_DIR}/externals/glfw)
set(OptiX_ROOT_DIR $ENV{OptiX_INSTALL_DIR})
set(CMAKE_PREFIX_PATH $ENV{OptiX_INSTALL_DIR})

embed_ptx(
    OUTPUT_TARGET
        pathTracer_ptx
    PTX_LINK_LIBRARIES
        owl::owl
    SOURCES
        src/cuda/pathTracer.cu
)

set(sources
        src/main.cu
        src/loader/mitsuba3.cu
        src/pt-math.cu
        src/viewer.cu
)

add_executable(pathTracer ${sources})

set(owl_dir ${PROJECT_SOURCE_DIR}/externals/owl)
include_directories(
        ${glfw_dir}/include
        ${owl_dir}/samples/common # For OWLViewer
)
add_subdirectory(${glfw_dir})

add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)
add_subdirectory(${owl_dir}/samples/common/owlViewer)

set(tinyxml2_dir ${PROJECT_SOURCE_DIR}/externals/tinyxml2)
add_subdirectory(${tinyxml2_dir} EXCLUDE_FROM_ALL)

target_link_libraries(pathTracer PRIVATE pathTracer_ptx owl::owl owl_viewer tinyxml2::tinyxml2)

# FIXME: If owl is not found, copy the .dll/.so into the executable folder.