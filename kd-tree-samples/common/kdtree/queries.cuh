#pragma once
#include "constants.cuh"

template<int K>
struct BaseQueryResult {
    // First 32 bits => photon distance squared
    // Last 32 bits => photon idx
    // Distance in the high bits so that uint64_t comparison preserves distance ordering.
    uint64_t *photonData;

    explicit __device__ BaseQueryResult(uint64_t *photonBuffer) {
        const uint64_t seed = packData(0xFFFFFFFF, INFTY);
#pragma unroll
        for (int i = 0; i < K; ++i) {
            photonBuffer[i] = seed;
        }
        photonData = photonBuffer;
    }

    __device__ float getDistance(const uint32_t pos) const {
        return __uint_as_float(static_cast<uint32_t>(photonData[pos] >> 32));
    }

    __device__ uint32_t getIndex(const uint32_t pos) const {
        return static_cast<uint32_t>(photonData[pos]);
    }

    static __inline__ __device__ uint64_t packData(const uint32_t idx, const float dist) {
        return (static_cast<uint64_t>(__float_as_uint(dist)) << 32) | idx;
    }
};

// Implementation with a max heap. Furthest point is easily accessible; if we attempt to add a point
// to an already full heap, we remove heap[0], insert the new candidate, and let it percolate down.
// Trade-off: Results distance to query point not in ascending order.
template<int K>
struct HeapQueryResult : BaseQueryResult<K>{
    using BaseQueryResult<K>::BaseQueryResult;

    __device__ float getQueryRadiusSqr() const {
        return this->getDistance(0);
    }

    __device__ bool isFull() const {
        return static_cast<uint32_t>(this->photonData[0]) != 0xFFFFFFFFu;
    }

    // Returns max distance for points to be considered.
    __device__ float addNode(const float dist, const uint32_t node_id) {
        const uint64_t node = this->packData(node_id, dist);
        // don't add if furthest point is closer
        if (node >= this->photonData[0]) return getQueryRadiusSqr();

        // Replace root and percolate down
        int idx = 0;
        for (;;) {
            const int left = 2 * idx + 1;
            if (left >= K)
                break;
            uint64_t child = this->photonData[left];
            int larger = left;
            if (left + 1 < K) {
                if (const uint64_t right = this->photonData[left + 1]; right > child) {
                    child = right;
                    larger = left + 1;
                }
            }
            if (child <= node) break;
            this->photonData[idx] = child;
            idx = larger;
        }
        this->photonData[idx] = node;

        return getQueryRadiusSqr();
    }
};

template<int K>
struct FixedQueryResult : BaseQueryResult<K> {
    using BaseQueryResult<K>::BaseQueryResult;

    __device__ float getQueryRadiusSqr() const {
        return this->getDistance(K-1);
    }

    __device__ bool isFull() const {
        return static_cast<uint32_t>(this->photonData[K-1]) != 0xFFFFFFFFu;
    }

    // Returns max distance for points to be considered.
    __device__ float addNode(float dist, uint32_t node_id) {
        if (dist > getQueryRadiusSqr()) return getQueryRadiusSqr();

        uint64_t moving_data = this->packData(node_id, dist);
#pragma unroll
        for (int i = 0; i < K; ++i) {
            if (const uint64_t curr_data = this->photonData[i]; moving_data < curr_data) {
                this->photonData[i] = moving_data;
                moving_data = curr_data;
            }
        }

        return getQueryRadiusSqr();
    }
};

typedef FixedQueryResult<1> FcpResult;

static __device__ __inline__ int parent_node(const int p) {
    return (p + 1) / 2 - 1;
}

/* Return an index buffer with (at most) K values that index into the initial point buffer
* QueryResult lifetime and allocation is managed by the caller.
* */
template<int K, typename P, typename ResultType>
__device__ void get_closest_k_points_in_range(const float *query_pos, const P *tree_buf, const size_t N,
                                              const float query_range, ResultType *result) {
    if (N == 0) {
#ifdef DEBUG_QUERIES
        // Conditionally compiled; sponza's caustic maps are empty.
        printf("WARNING! No photons in tree buffer. Aborting query...\n");
#endif
        return;
    }

    int curr = 0;
    int prev = -1;
    float max_search_radius = (query_range == INFTY) ? INFTY : query_range * query_range;

    for (;;) {
        const int parent = parent_node(curr);

        if (curr >= N) {
            prev = curr;
            curr = parent;
            continue;
        }

        const bool from_parent = prev < curr;
        if (from_parent) {
            const float dist2 = tree_buf[curr].dist2(query_pos);
            // Only offer points within the current pruning radius, so a finite query_range
            // also bounds the result (not just the traversal).
            if (dist2 < max_search_radius)
                max_search_radius = fminf(max_search_radius, result->addNode(dist2, curr));
        }

        const int level = 31 - __clz(curr + 1);
        const int split_dim = level % P::dimension;
        const float split_pos = tree_buf[curr].coords[split_dim];
        const float signed_dist = query_pos[split_dim] - split_pos;
        const int close_side = signed_dist > 0.f;
        const int close_child = 2 * curr + 1 + close_side;
        const int far_child = 2 * curr + 2 - close_side;
        const bool far_in_range = signed_dist * signed_dist <= max_search_radius;

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

template<typename P>
__device__ __inline__ void fcp(const float *query_pos, const P *tree_buf, const size_t N, FcpResult *result) {
    const float query_range = INFTY;
    get_closest_k_points_in_range<1, P, FcpResult>(query_pos, tree_buf, N, query_range, result);
}

template<int K, typename P, typename ResultType>
__device__ __inline__ void knn(const float *query_pos, const P *tree_buf, const size_t N, ResultType *result) {
    const float query_range = INFTY;
    get_closest_k_points_in_range<K,P,ResultType>(query_pos, tree_buf, N, query_range, result);
}
