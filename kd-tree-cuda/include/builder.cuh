#pragma once

#include <bit>
#include <thrust/sort.h>
#include <thrust/tuple.h>

constexpr int DIM = 2;

/*
 * TODO: Generalise using templates and concepts:
 * * Point struct data fields.
 * * Point dimension
*/
struct Point {
    char payload;
    float coords[DIM];
};

struct custom_less {
    int level;

    __host__ __device__ bool operator()(
        const thrust::tuple<int, Point>& a,
        const thrust::tuple<int, Point>& b) const
    {
        // Sort by tags (int) first
        if (thrust::get<0>(a) != thrust::get<0>(b)) {
            return thrust::get<0>(a) < thrust::get<0>(b);
        }

        // If tags are equal, sort by data
        const int dim = level % DIM;
        return thrust::get<1>(a).coords[dim] < thrust::get<1>(b).coords[dim];
    }
};

void __host__ buildKDTree(Point *points, const size_t N);