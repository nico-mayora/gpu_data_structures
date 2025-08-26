cmake_minimum_required(VERSION 3.27)
project(ray-tracer CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
include(cmake/embed_ptx.cmake)

set(glfw_dir ${PROJECT_SOURCE_DIR}/externals/glfw)
set(OptiX_ROOT_DIR $ENV{OptiX_INSTALL_DIR})
set(CMAKE_PREFIX_PATH $ENV{OptiX_INSTALL_DIR})

embed_ptx(
    OUTPUT_TARGET
        photonEmitter_ptx
    PTX_LINK_LIBRARIES
        owl::owl
    SOURCES
        src/cuda/photonEmitter.cu
)

set(sources
        src/main.cpp
        src/loader/mitsuba3.cpp
        src/pt-math.cpp
        src/photon-emitter.cpp
        src/photon-file-manager.cpp
)

set(owl_dir ${PROJECT_SOURCE_DIR}/externals/owl)

add_executable(photonEmitter ${sources})

include_directories(${glfw_dir}/include)

add_subdirectory(${glfw_dir})
add_subdirectory(${owl_dir}/samples/common)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

set(tinyxml2_dir ${PROJECT_SOURCE_DIR}/externals/tinyxml2)
add_subdirectory(${tinyxml2_dir} EXCLUDE_FROM_ALL)

target_link_libraries(photonEmitter PRIVATE photonEmitter_ptx owl::owl tinyxml2::tinyxml2 owl_viewer)