#pragma once
#include "data.cuh"

constexpr float INFTY = __int_as_float(0x7f800000);

template<int K>
struct QueryResult {
    size_t *pointIndices;
    float *pointDistances;
    size_t foundPoints = 0; /* <= K */

    __device__ float addNode(float dist, size_t node_id) {
        foundPoints = (foundPoints < K) ? foundPoints + 1 : K;
        for (int i = 0; i < K; ++i) {
            if (dist < this->pointDistances[i]) {
                const auto old_dist = pointDistances[i];
                const auto old_idx = pointIndices[i];
                this->pointDistances[i] = dist;
                this->pointIndices[i] = node_id;
                dist = old_dist;
                node_id = old_idx;
            }
        }

        return pointDistances[K - 1];
    }
};

using FcpResult = QueryResult<1>;

template<int K>
__device__ QueryResult<K> *alloc_query_result() {
    QueryResult<K> *result_ptr = static_cast<QueryResult<K> *>(malloc(sizeof(QueryResult<K>)));
    auto idx_buf = static_cast<size_t *>(malloc(K * sizeof(size_t)));
    auto *dist_buf = static_cast<float *>(malloc(K * sizeof(float)));

    memset(idx_buf, NULL, K * sizeof(size_t));
#pragma unroll
    for (int i = 0; i < K; ++i) {
        dist_buf[i] = INFTY;
    }

    result_ptr->pointDistances = dist_buf;
    result_ptr->pointIndices = idx_buf;

    return result_ptr;
}

/* Return an index buffer with (at most) K values that index into the initial point buffer
* QueryResult lifetime and allocation is managed by the caller.
* */
template<int K>
__device__ void get_closest_k_points_in_range(const float *query_pos, const Point *tree_buf, const size_t N,
                                              const float query_range, QueryResult<K> *result) {
    int curr = 0;
    int prev = -1;
    float max_search_radius = query_range;

    for (;;) {
        const int parent = parent_node(curr);

        if (curr >= N) {
            prev = curr;
            curr = parent;
            continue;
        }

        const bool from_parent = prev < curr;
        if (from_parent) {
            const float dist = norm2(query_pos, tree_buf[curr].coords);
            max_search_radius = result->addNode(dist, curr);
        }

        const int split_dim = curr % DIM;
        const float split_pos = tree_buf[curr].coords[split_dim];
        const float signed_dist = query_pos[split_dim] - split_pos;
        const int close_side = signed_dist > 0.f;
        const int close_child = 2 * curr + 1 + close_side;
        const int far_child = 2 * curr + 2 - close_side;
        const bool far_in_range = fabsf(signed_dist) <= max_search_radius;

        int next = parent;
        if (from_parent)
            next = close_child;
        else if (prev == close_child)
            next = far_in_range ? far_child : parent;

        if (next == -1) // We're done
            return;

        prev = curr;
        curr = next;
    }
}

template<int K>
__device__ __inline__ void points_in_range(const float *query_pos, const Point *tree_buf, const size_t N,
                                           const float query_range, QueryResult<K> *result) {
    get_closest_k_points_in_range(query_pos, tree_buf, N, query_range, result);
}

template<int K>
__device__ __inline__ void fcp(const float *query_pos, const Point *tree_buf, const size_t N, QueryResult<K> *result) {
    const float query_range = INFTY;
    get_closest_k_points_in_range(query_pos, tree_buf, N, query_range, result);
}

template<int K>
__device__ __inline__ void knn(const float *query_pos, const Point *tree_buf, const size_t N, QueryResult<K> *result) {
    const float query_range = INFTY;
    get_closest_k_points_in_range(query_pos, tree_buf, N, query_range, result);
}
