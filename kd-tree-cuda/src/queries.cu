#include "queries.cuh"

#include <cstdio>

__device__ float QueryResult::addNode(float dist, size_t node_id) {
    foundPoints = (foundPoints < K) ? foundPoints + 1 : K;
    for (int i=0; i < this->K; ++i) {
        if (dist < this->pointDistances[i]) {
            const auto old_dist = pointDistances[i];
            const auto old_idx = pointIndices[i];
            this->pointDistances[i] = dist;
            this->pointIndices[i] = node_id;
            dist = old_dist;
            node_id = old_idx;
        }
    }

    return pointDistances[K-1];
}

__device__ QueryResult *alloc_query_result(const size_t K) {
    QueryResult *result_ptr = (QueryResult *)malloc(sizeof(QueryResult));
    size_t *idx_buf = (size_t *)malloc(K * sizeof(size_t));
    float *dist_buf = (float *)malloc(K * sizeof(dist_buf));

    memset(idx_buf, NULL,K * sizeof(size_t));
    //dist_buf[0] = __int_as_float(0x7f800000);

    result_ptr->K = K;
    result_ptr->pointDistances = dist_buf;
    result_ptr->pointIndices = idx_buf;

    return result_ptr;
}

static __device__ __inline__ int parent_of(const int p) {
    return (p + 1) / 2 - 1;
}

__device__ void get_closest_k_points_in_range(const float *query_pos, const Point *tree_buf, const size_t N,
                                              const size_t K,
                                              const float query_range,
                                              QueryResult *result) {
    result->K = K;
    int curr = 0;
    int prev = -1;
    float max_search_radius = __int_as_float(0x7f800000);

    for (;;) {
        const int parent = parent_of(curr);

        if (curr >= N) {
            prev = curr;
            curr = parent;
            continue;
        }

        const bool from_parent = prev < curr;
        if (from_parent) {
            printf("CURR %d \n", curr);
            const float dist = norm2(query_pos, tree_buf[curr].coords);
            result->addNode(dist, curr);
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
