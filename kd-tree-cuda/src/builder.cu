#include "builder.cuh"
#include <thrust/iterator/zip_iterator.h>

// DEVICE HELPER FUNCTIONS BEGIN
__device__ int num_nodes_in_full_tree(const int depth) {
    return (1 << depth) - 1;
}

__device__ int lChild(const int i) {
    return 2 * i + 1;
}

__device__ int rChild(const int i) {
    return 2 * i + 2;
}

__device__ int level(const int i) {
    return 31 - __clz(i + 1);
}

__device__ int numLevels(const int N) {
    return level(N - 1) + 1;
}

// N won't be an argument in the future; it should be part of the template params.
__device__ int subtreeSize(const int s, const int N) {
    const auto L = numLevels(N);
    const auto l = level(s);
    const auto fullInner = (1 << (L - l - 1)) - 1;
    const auto fllc_s = ~((~s) << (L - l - 1));
    const auto lowest = thrust::min(thrust::max(0, N - fllc_s), 1 << (L - l - 1));
    return fullInner + lowest;
}

__device__ int segmentBegin(const int s, const int l, const int N) {
    const auto L = numLevels(N);
    const auto topLevels = (1 << l) - 1;
    const auto nls_s = s - topLevels;
    const auto inner = nls_s * ((1 << (L - l - 1)) - 1);
    const auto lowest = thrust::min(nls_s * (1 << (L - l - 1)), N - ((1 << (L - 1)) - 1));
    return topLevels + inner + lowest;
}

// DEVICE HELPER FUNCTIONS END

__global__ void updateTags(int* tags, const int l, const int N) {
    const int arrayIdx = threadIdx.x;
    if (arrayIdx >= N || arrayIdx < num_nodes_in_full_tree(l))
        return;

    const int currentTag = tags[arrayIdx];
    if (const int pivotPos = segmentBegin(currentTag, l, N) + subtreeSize(lChild(currentTag), N); arrayIdx < pivotPos)
        tags[arrayIdx] = lChild(currentTag);
    else if (arrayIdx > pivotPos)
        tags[arrayIdx] = rChild(currentTag);
    // else tag remains unchanged: node is already at its correct position.
}

static void sort_tagged_data(int* d_tags, Point* d_data, const int n, const int l) {
    const auto zip_begin = make_zip_iterator(
        thrust::make_tuple(d_tags, d_data)
    );
    const auto zip_end = zip_begin + n;
    // Sort with custom comparator
    auto less_op = custom_less {l};
    sort(thrust::device, zip_begin, zip_end, less_op);
}

void __host__ buildKDTree(Point* points, const size_t N) {
    int* d_tags;
    cudaMalloc(&d_tags, N * sizeof(int));
    cudaMemset(d_tags, 0, N * sizeof(int));

    const int max_levels = 31 - std::countl_zero(static_cast<uint32_t>(N));

    Point* d_points;
    const size_t bufferSize = N * sizeof(Point);
    cudaMalloc(&d_points, bufferSize);
    cudaMemcpy(d_points, points, bufferSize, cudaMemcpyHostToDevice);

    // TODO: Don't hard-code
    constexpr int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;
    for (int l = 0; l < max_levels; ++l) {
        sort_tagged_data(d_tags, d_points, N, l);
        updateTags<<<blocks, threads_per_block>>>(d_tags, l, N);
    }

    cudaFree(d_tags);
    cudaMemcpy(points, d_points, bufferSize, cudaMemcpyDeviceToHost);
}
