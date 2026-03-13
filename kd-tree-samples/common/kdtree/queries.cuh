#pragma once
#include "constants.cuh"

template<int K>
struct BaseQueryResult {
    // First 32 bits => photon idx
    // Last 32 bits => photon distance squared
    uint64_t *photonData;
    size_t foundPoints = 0; /* <= K */

    __device__ void initialize(uint64_t *dataBufAddr) {
        this->photonData = dataBufAddr;
        const uint64_t defaultValue = packData(0, INFTY);
#pragma unroll
        for (int i = 0; i < K; i++) {
            photonData[i] = defaultValue;
        }
    }

    __device__ float getDistance(const size_t pos) const {
        const uint32_t lo = static_cast<uint32_t>(photonData[pos]);
        return __uint_as_float(lo);
    }

    __device__ float getQueryRadiusSqr() const {
        return getDistance(0);
    }

    __device__ size_t getIndex(const size_t pos) const {
        return photonData[pos] >> 32;
    }

    __device__ bool isFull() const {
        return this->foundPoints == K;
    }

    static __device__ uint64_t packData(const uint32_t idx, const float dist) {
        return (static_cast<uint64_t>(idx) << 32) | __float_as_uint(dist);
    }
};

template<int K>
struct FixedQueryResult : BaseQueryResult<K> {
    // Returns max distance for points to be considered.
    __device__ float addNode(float dist, size_t node_id) {
        this->foundPoints = (this->foundPoints < K) ? this->foundPoints + 1 : K;
        uint64_t packed_data = this->packData(node_id, dist);
#pragma unroll
        for (int i = 0; i < K; ++i) {
            const float current_dist = __uint_as_float(static_cast<uint32_t>(packed_data));
            if (current_dist < this->getDistance(i)) {
                const uint64_t tmp_data = this->photonData[i];
                this->photonData[i] = packed_data;
                packed_data = tmp_data;
            }
        }

        return (this->foundPoints < K) ? 0.f : this->getDistance(K - 1);
    }
};

// Implementation with a max heap. Furthest point is easily accessible; if we attempt toadd a point
// to an already full heap, we remove heap[0], insert the new candidate, and let it percolate down.
// Trade-off: Results distance to query point not in ascending order.
template<int K>
struct HeapQueryResult : BaseQueryResult<K>{
    __device__ void percolateUp(size_t idx) const {
        while (idx > 0) {
            const size_t parent = (idx - 1) / 2;
            if (this->getDistance(idx) <= this->getDistance(parent)) break; // Node at correct position.

            // Swap idx with parent
            const uint64_t tmpData = this->photonData[idx];
            this->photonData[idx] = this->photonData[parent];
            this->photonData[parent] = tmpData;

            idx = parent;
        }
    }

    __device__ void percolateDown(size_t idx) const {
        for (;;) {
            size_t largest = idx;
            const size_t left = 2 * idx + 1;
            const size_t right = 2 * idx + 2;

            if (left < this->foundPoints && this->getDistance(left) > this->getDistance(largest)) {
                largest = left;
            }
            if (right < this->foundPoints && this->getDistance(right) > this->getDistance(largest)) {
                largest = right;
            }

            if (largest == idx) break;

            // Swap idx with the largest child
            const uint64_t tmpData = this->photonData[idx];
            this->photonData[idx] = this->photonData[largest];
            this->photonData[largest] = tmpData;

            idx = largest;
        }
    }

    // Returns max distance for points to be considered.
    __device__ float addNode(const float dist, const size_t node_id) {
        // If heap is full and furthest point is still closer, discard it.
        if (this->isFull() && this->getQueryRadiusSqr() < dist) return this->getQueryRadiusSqr();
        if (this->isFull()) {
            // Replace root and percolate down
            this->photonData[0] = this->packData(node_id, dist);
            percolateDown(0);
        } else {
            // Add to the end and percolate up
            this->photonData[this->foundPoints] = this->packData(node_id, dist);
            percolateUp(this->foundPoints);
            ++this->foundPoints;
        }

        return this->getQueryRadiusSqr();
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
        printf("WARNING! No photons in tree buffer. Aborting query...\n");
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
            max_search_radius = result->addNode(dist2, curr);
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
