#pragma once
#include "../../common/kdtree/queries.cuh"
#include "../../common/kdtree/data.cuh"
#include "validation.cuh"

// Indices/distances live in shared memory, partitioned per-thread within each block.
template<int K_VAL>
__global__ void knn_query_shared(
    const Point<3> *tree,
    const size_t    num_points,
    const float    *query_positions,
    const int       num_queries,
    uint32_t       *validation)
{
    extern __shared__ char smem[];

    size_t *block_indices   = reinterpret_cast<size_t *>(smem);
    float  *block_distances = reinterpret_cast<float *>(block_indices + blockDim.x * K_VAL);

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_queries) return;

    const float qp[3] = {
        query_positions[tid * 3 + 0],
        query_positions[tid * 3 + 1],
        query_positions[tid * 3 + 2]
    };

    size_t *my_indices   = block_indices   + threadIdx.x * K_VAL;
    float  *my_distances = block_distances + threadIdx.x * K_VAL;

    for (int k = 0; k < K_VAL; k++) {
        my_indices[k]   = 0;
        my_distances[k] = INFTY;
    }

    HeapQueryResult<K_VAL> result{my_indices, my_distances};
    knn<K_VAL, Point<3>, HeapQueryResult<K_VAL>>(qp, tree, num_points, &result);

    validate_knn<K_VAL>(tree, num_points, qp, my_indices, validation, tid);
}
