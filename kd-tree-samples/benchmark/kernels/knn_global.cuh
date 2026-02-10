#pragma once
#include "../../common/kdtree/queries.cuh"
#include "../../common/kdtree/data.cuh"
#include "validation.cuh"

// Indices/distances are pre-allocated in global memory (cudaMalloc).
// Each thread indexes into its own K-sized slice by global thread ID.
// This is the strategy used by the ray tracer.
template<int K_VAL>
__global__ void knn_query_global(
    const Point<3> *tree,
    const size_t    num_points,
    const float    *query_positions,
    size_t         *all_indices,
    float          *all_distances,
    const int       num_queries,
    uint32_t       *validation)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_queries) return;

    const float qp[3] = {
        query_positions[tid * 3 + 0],
        query_positions[tid * 3 + 1],
        query_positions[tid * 3 + 2]
    };

    size_t *my_indices   = all_indices   + tid * K_VAL;
    float  *my_distances = all_distances + tid * K_VAL;

    for (int k = 0; k < K_VAL; k++) {
        my_indices[k]   = 0;
        my_distances[k] = INFTY;
    }

    HeapQueryResult<K_VAL> result{my_indices, my_distances};
    knn<K_VAL, Point<3>, HeapQueryResult<K_VAL>>(qp, tree, num_points, &result);

    validate_knn<K_VAL>(tree, num_points, qp, my_indices, validation, tid);
}
