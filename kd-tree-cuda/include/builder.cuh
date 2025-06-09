#pragma once

#include <bit>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include "data.cuh"

/* Functor that implements custom tag-major, coordinate-minor order. */
struct ZipCompare {
    int dim = 0; // by default, we split along x.

    __device__ bool operator()(
        const thrust::tuple<int, Point>& a,
        const thrust::tuple<int, Point>& b) const
    {
        // Sort by tags (int) first
        if (thrust::get<0>(a) != thrust::get<0>(b)) {
            return thrust::get<0>(a) < thrust::get<0>(b);
        }

        // If tags are equal, sort by data
        return thrust::get<1>(a).coords[dim] < thrust::get<1>(b).coords[dim];
    }
};

/*
 * Takes a device buffer and creates a kd-tree inplace.
 */
__host__ void build_kd_tree(Point *d_points, size_t N);