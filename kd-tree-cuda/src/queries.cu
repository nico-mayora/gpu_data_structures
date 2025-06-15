#include "queries.cuh"

#include <cstdio>

__device__ __inline__ int parent_of(const int p) {
    return (p + 1) / 2 - 1;
}

__device__ void get_closest_k_points_in_range(const float *query_pos, const Point *tree_buf, const size_t N,
                                              const size_t K,
                                              const float query_range,
                                              QueryResult *result) {
    result->K = K;
    int curr = 0;
    int prev = -1;
    // TODO: We should probably calculate the smallest AABB that encapsulates all points and use that.
    float max_search_radius = query_range;

    for (;;) {
        const int parent = parent_of(curr);

        if (curr >= N) {
            prev = curr;
            curr = parent;
            continue;
        }

        const bool from_parent = prev < curr;
        if (from_parent) {
            processNode(curr);
            // TODO: Shrink max_search_radius
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

        if (next == -1 || result->foundPoints == K) // We're done
            return;

        prev = curr;
        curr = next;
    }
}
