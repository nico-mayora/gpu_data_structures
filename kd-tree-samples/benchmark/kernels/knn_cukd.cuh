#pragma once
#include "cukd/knn.h"

// KNN query using Ingo Wald's cudaKDTree library (stack-based traversal).
// Uses FlexHeapCandidateList with runtime K
template<int K_VAL>
__global__ void knn_query_cukd(
    const float3   *tree,
    const int       num_points,
    const float    *query_positions,
    const int       num_queries)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_queries) return;

    float3 qp = make_float3(
        query_positions[tid * 3 + 0],
        query_positions[tid * 3 + 1],
        query_positions[tid * 3 + 2]
    );

    uint64_t storage[K_VAL];
    cukd::FlexHeapCandidateList closest(storage, K_VAL, 1e30f);
    cukd::stackFree::knn(closest, qp, tree, num_points);
}
