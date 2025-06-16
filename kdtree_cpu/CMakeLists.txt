cmake_minimum_required(VERSION 3.31)
project(kdtree_cpu)

set(CMAKE_CXX_STANDARD 23)

add_executable(kdtree_cpu main.cpp
        src/point.cpp
        src/point.h)
