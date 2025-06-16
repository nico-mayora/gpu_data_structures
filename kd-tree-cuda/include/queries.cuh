#pragma once
#include "data.cuh"

constexpr float REALLY_FAR = 1000000.f;

struct QueryResult {
    size_t K;
    size_t *pointIndices;
    float *pointDistances;
    size_t foundPoints = 0; /* <= K */

    __device__ float addNode(float dist, size_t node_id);
};

__device__ QueryResult *alloc_query_result(size_t K);

/* Return an index buffer with (at most) K values that index into the initial point buffer
* QueryResult lifetime and allocation is managed by the caller.
* */
__device__ void get_closest_k_points_in_range(const float *query_pos, const Point *tree_buf, const size_t N, const size_t K,
                                      const float query_range, QueryResult *result);

__device__ __inline__ void points_in_range(const float *query_pos, const Point *tree_buf, const size_t N,
                                           const float query_range, QueryResult *result) {
    get_closest_k_points_in_range(query_pos, tree_buf, N, N, query_range, result);
}

__device__ __inline__ void fcp(const float *query_pos, const Point *tree_buf, const size_t N, QueryResult *result) {
    const int query_range = REALLY_FAR; // TODO: Compute and use tree bounds.
    get_closest_k_points_in_range(query_pos, tree_buf, N, 1, query_range, result);
}

__device__ __inline__ void knn(const float *query_pos, const Point *tree_buf, const size_t N, const size_t K,
                               QueryResult *result) {
    const int query_range = REALLY_FAR; // TODO: Compute and use tree bounds.
    get_closest_k_points_in_range(query_pos, tree_buf, N, K, query_range, result);
}
