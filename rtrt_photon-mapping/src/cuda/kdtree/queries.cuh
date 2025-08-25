#pragma once

template<int K>
struct FixedQueryResult {
    size_t *pointIndices;
    float *pointDistances;
    size_t foundPoints = 0; /* <= K */

    // Returns max distance for points to be considered.
    __device__ float addNode(float dist, size_t node_id) {
        foundPoints = (foundPoints < K) ? foundPoints + 1 : K;
#pragma unroll
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

        return (foundPoints < K) ? 0.f : pointDistances[K - 1];
    }
};

// Implementation with a max heap. Furthest point is easily accessible; if we attempt toadd a point
// to an already full heap, we remove heap[0], insert the new candidate, and let it percolate down.
// Trade-off: Results distance to query point not in ascending order.
template<int K>
struct HeapQueryResult {
    size_t *pointIndices;
    float *pointDistances;
    size_t foundPoints = 0; /* <= K */

    __device__ void percolateUp(size_t idx) {
        while (idx > 0) {
            const size_t parent = (idx - 1) / 2;
            if (pointDistances[idx] <= pointDistances[parent]) break; // Node at correct position.

            // Swap idx with parent
            const float tmpIdx = pointIndices[idx];
            const float tmpDist = pointDistances[idx];
            pointIndices[idx] = pointIndices[parent];
            pointDistances[idx] = pointIndices[parent];
            pointIndices[parent] = tmpIdx;
            pointDistances[parent] = tmpDist;

            idx = parent;
        }
    }

    __device__ void percolateDown(size_t idx) {
        for (;;) {
            size_t largest = idx;
            const size_t left = 2 * idx + 1;
            const size_t right = 2 * idx + 2;

            if (left < foundPoints && pointDistances[left] > pointDistances[largest]) {
                largest = left;
            }
            if (right < foundPoints && pointDistances[right] > pointDistances[right]) {
                largest = right;
            }

            if (largest == idx) break;

            // Swap idx with the largest child
            const float tmpIdx = pointIndices[idx];
            const float tmpDist = pointDistances[idx];
            pointIndices[idx] = pointIndices[largest];
            pointDistances[idx] = pointIndices[largest];
            pointIndices[largest] = tmpIdx;
            pointDistances[largest] = tmpDist;

            idx = largest;
        }
    }

    __device__ bool isFull() const {
        return foundPoints == K;
    }

    // Returns max distance for points to be considered.
    __device__ float addNode(float dist, size_t node_id) {
        // If heap is full and furthest point is still closer, discard it.
        if (isFull() && pointDistances[0] < dist) return pointDistances[0];

        if (isFull()) {
            // Replace root and percolate down
            pointDistances[0] = dist;
            pointIndices[0] = node_id;
            percolateDown(0);
        } else {
            // Add to the end and percolate up
            pointDistances[foundPoints] = dist;
            pointIndices[foundPoints] = node_id;
            percolateUp(foundPoints);
            foundPoints++;
        }

        return pointDistances[0];
    }
};

struct FcpResult {
    int32_t closestPointIndex = -1;
    float distance = INFTY;

    __device__ float addNode(float dist, size_t node_id) {
        if (distance > dist) {
            distance = dist;
            closestPointIndex = node_id;
        }

        return distance;

    }
};

static __device__ __inline__ int parent_node(const int p) {
    return (p + 1) / 2 - 1;
}

/* Return an index buffer with (at most) K values that index into the initial point buffer
* QueryResult lifetime and allocation is managed by the caller.
* */
template<int K, typename P, typename ResultType>
__device__ void get_closest_k_points_in_range(const float *query_pos, const P *tree_buf, const size_t N,
                                              const float query_range, ResultType *result) {
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
            const float dist2 = tree_buf[curr].dist2(query_pos);
            max_search_radius = result->addNode(dist2, curr);
        }

        const int split_dim = curr % P::dimension;
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

template<typename P>
__device__ __inline__ void fcp(const float *query_pos, const P *tree_buf, const size_t N, FcpResult *result) {
    const float query_range = INFTY;
    get_closest_k_points_in_range(query_pos, tree_buf, N, query_range, result);
}

template<int K, typename P, typename ResultType>
__device__ __inline__ void knn(const float *query_pos, const P *tree_buf, const size_t N, ResultType *result) {
    const float query_range = INFTY;
    get_closest_k_points_in_range(query_pos, tree_buf, N, query_range, result);
}
