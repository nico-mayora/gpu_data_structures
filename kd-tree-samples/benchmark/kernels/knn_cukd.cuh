#pragma once
#include "cukd/knn.h"

// KNN query using Ingo Wald's cudaKDTree library (stack-based traversal).
// Uses FlexHeapCandidateList (runtime k, external memory) to avoid
// NVCC C++20 two-phase lookup issues with the templated HeapCandidateList.
__global__ void knn_query_cukd(
    const float3   *tree,
    const int       num_points,
    const float    *query_positions,
    const int       num_queries,
    uint64_t       *candidate_mem,
    const int       k)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_queries) return;

    float3 qp = make_float3(
        query_positions[tid * 3 + 0],
        query_positions[tid * 3 + 1],
        query_positions[tid * 3 + 2]
    );

    uint64_t *my_mem = candidate_mem + (size_t)tid * k;
    cukd::FlexHeapCandidateList closest(my_mem, k, 1e30f);
    cukd::stackBased::knn(closest, qp, tree, num_points);
}
