#pragma once
#include "../../common/kdtree/queries.cuh"
#include "../../common/kdtree/data.cuh"
#include "validation.cuh"

// Packed data lives in shared memory, partitioned per-thread within each block.
template<int K_VAL>
__global__ void knn_query_shared(
    const Point<3> *tree,
    const size_t    num_points,
    const float    *query_positions,
    const int       num_queries,
    uint32_t       *validation)
{
    extern __shared__ char smem[];

    uint64_t *block_data = reinterpret_cast<uint64_t *>(smem);

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_queries) return;

    const float qp[3] = {
        query_positions[tid * 3 + 0],
        query_positions[tid * 3 + 1],
        query_positions[tid * 3 + 2]
    };

    uint64_t *my_data = block_data + threadIdx.x * K_VAL;
    HeapQueryResult<K_VAL> result{};
    result.initialize(my_data);
    knn<K_VAL, Point<3>, HeapQueryResult<K_VAL>>(qp, tree, num_points, &result);

    size_t my_indices[K_VAL];
    for (int k = 0; k < K_VAL; k++)
        my_indices[k] = result.getIndex(k);
    validate_knn<K_VAL>(tree, num_points, qp, my_indices, validation, tid);
}
