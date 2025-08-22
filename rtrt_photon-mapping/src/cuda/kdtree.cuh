#pragma once

#include <bit>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <limits>

struct Photon {
    static constexpr uint32_t DIM = 3;
    //Required member
    float coords[DIM]; //xyz

    float colour[3];
    float power[3];
    float dir[DIM];

    /* Required method for performing queries.
     * Returns distance between this and a point buffer x.
     * We assume x's dimension is DIM.
     */
    __device__ __inline__ float dist2(const float *x) const {
        float acum = 0.;
#pragma unroll
        for (int i = 0; i < DIM; ++i) {
            const float diff = coords[i] - x[i];
            acum += diff * diff;
        }
        return acum;
    }
};

/* Functor that implements custom tag-major, coordinate-minor order. */
template <typename P>
struct ZipCompare {
    int dim = 0; // by default, we split along x.

    __device__ bool operator()(
        const thrust::tuple<int, P>& a,
        const thrust::tuple<int, P>& b) const
    {
        // Sort by tags (int) first
        if (thrust::get<0>(a) != thrust::get<0>(b)) {
            return thrust::get<0>(a) < thrust::get<0>(b);
        }

        // If tags are equal, sort by data
        return thrust::get<1>(a).coords[dim] < thrust::get<1>(b).coords[dim];
    }
};

// DEVICE HELPER FUNCTIONS BEGIN
static __device__ __inline__ int num_nodes_in_full_tree(const int depth) {
    return (1 << depth) - 1;
}

static __device__ __inline__ int l_child(const int i) {
    return 2 * i + 1;
}

static __device__ __inline__ int r_child(const int i) {
    return 2 * i + 2;
}

static __device__ __inline__ int level(const int i) {
    return 31 - __clz(i + 1);
}

static __device__ __inline__ int num_levels(const int N) {
    return level(N - 1) + 1;
}

// N won't be an argument in the future; it should be part of the template params.
static __device__ __inline__ int subtree_size(const int s, const int N) {
    const auto total_levels = num_levels(N);
    const auto current_level = level(s);
    const auto full_inner = (1 << (total_levels - current_level - 1)) - 1;
    const auto fllc_s = ~((~s) << (total_levels - current_level - 1));
    const auto lowest = thrust::min(thrust::max(0, N - fllc_s), 1 << (total_levels - current_level - 1));
    return full_inner + lowest;
}

static __device__ __inline__  int segment_begin(const int s, const int l, const int N) {
    const auto L = num_levels(N);
    const auto top_levels = (1 << l) - 1;
    const auto nls_s = s - top_levels;
    const auto inner = nls_s * ((1 << (L - l - 1)) - 1);
    const auto lowest = thrust::min(nls_s * (1 << (L - l - 1)), N - ((1 << (L - 1)) - 1));
    return top_levels + inner + lowest;
}

// DEVICE HELPER FUNCTIONS END

static __global__ void update_tags(int* tags, const int l, const int N) {
    const int array_idx = static_cast<int>(threadIdx.x);
    if (array_idx >= N || array_idx < num_nodes_in_full_tree(l))
        return;

    const int current_tag = tags[array_idx];
    if (const int pivot_pos = segment_begin(current_tag, l, N) + subtree_size(l_child(current_tag), N); array_idx < pivot_pos)
        tags[array_idx] = l_child(current_tag);
    else if (array_idx > pivot_pos)
        tags[array_idx] = r_child(current_tag);
    // else tag remains unchanged: node is already at its correct position.
}

/*
 * Takes a device buffer and creates a kd-tree inplace.
 */
template<typename P>
__host__ void build_kd_tree(P *d_points, const size_t N) {
    int* d_tags;
    const size_t tag_buffer_size =  N * sizeof(int);
    // Allocate and initialise tags buffer.
    cudaMalloc(&d_tags, tag_buffer_size);
    cudaMemset(d_tags, 0, tag_buffer_size);

    // Create Thrust iterators for sorting
    const auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(d_tags, d_points)
    );
    const auto zip_end = zip_begin + N;

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>(N + threads_per_block - 1) / threads_per_block;

    // Equivalent to log2(N), the number of levels in a size N binary tree.
    const int max_levels = 31 - std::countl_zero(static_cast<uint32_t>(N));
    for (int l = 0; l < max_levels; ++l) {
        sort(thrust::device, zip_begin, zip_end, ZipCompare<P> { l % P::dimension });
        update_tags<<<blocks, threads_per_block>>>(d_tags, l, static_cast<int>(N));
    }

    // One last sort to ensure every point is where it should be.
    // They each have a unique tag, so dimension doesn't matter.
    sort(thrust::device, zip_begin, zip_end, ZipCompare<P>());

    cudaFree(d_tags);
}

static constexpr float INFTY = std::numeric_limits<float>::infinity();

template<int K>
struct QueryResult {
    size_t *pointIndices;
    float *pointDistances;
    size_t foundPoints = 0; /* <= K */

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

        return pointDistances[K - 1];
    }
};

using FcpResult = QueryResult<1>;

template<int K>
__device__ QueryResult<K> *alloc_query_result() {
    QueryResult<K> *result_ptr = static_cast<QueryResult<K> *>(malloc(sizeof(QueryResult<K>)));
    auto idx_buf = static_cast<size_t *>(malloc(K * sizeof(size_t)));
    auto *dist_buf = static_cast<float *>(malloc(K * sizeof(float)));

#pragma unroll
    for (int i = 0; i < K; ++i) {
        idx_buf[i] = 0;
        dist_buf[i] = INFTY;
    }

    result_ptr->pointDistances = dist_buf;
    result_ptr->pointIndices = idx_buf;

    return result_ptr;
}

static __device__ __inline__ int parent_node(const int p) {
    return (p + 1) / 2 - 1;
}

/* Return an index buffer with (at most) K values that index into the initial point buffer
* QueryResult lifetime and allocation is managed by the caller.
* */
template<int K, typename P>
__device__ void get_closest_k_points_in_range(const float *query_pos, const P *tree_buf, const size_t N,
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

template<int K, typename P>
__device__ __inline__ void knn(const float *query_pos, const P *tree_buf, const size_t N, QueryResult<K> *result) {
    const float query_range = INFTY;
    get_closest_k_points_in_range(query_pos, tree_buf, N, query_range, result);
}
