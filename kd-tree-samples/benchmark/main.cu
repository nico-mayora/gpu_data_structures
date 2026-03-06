#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../common/kdtree/builder.cuh"
#include "generators/cube.cuh"
#include "kernels/knn_local.cuh"
#include "kernels/knn_global.cuh"
#include "kernels/knn_shared.cuh"
#include "kernels/knn_cukd.cuh"
#include "cukd/builder.h"

// Cube Generator Params
#define SIDE_LENGTH  10.0f
#define SIDE_DENSITY 100

// Query Params
#define NUM_QUERIES       1000000
#define THREADS_PER_BLOCK 256
#define NUM_RUNS          1

// K values to benchmark (rows of the table)
static const int K_VALUES[] = { 8, 16, 32, 64, 128, 256, 512, 1024 };
static const int NUM_K = sizeof(K_VALUES) / sizeof(K_VALUES[0]);

enum MemoryStrategy { LOCAL, GLOBAL, SHARED, CUKD, NUM_STRATEGIES };
enum Generator { CUBE };

static const char* strategy_label[] = { "LOCAL", "GLOBAL", "SHARED", "CUKD" };

static void generate_points(Generator g, Point<3> *points, int density, float length) {
    switch (g) {
        case CUBE:
            generate_cube_points(points, density, length);
            break;
    }
}

static void generate_random_queries(float *out, int n, float length) {
    const float half = length / 2.0f;
    srand(42);
    for (int i = 0; i < n * 3; i++)
        out[i] = ((float)rand() / RAND_MAX) * length - half;
}

// --------------- kernel launch + timing (our tree) ---------------

template<int K_VAL>
static bool launch_and_time(
    MemoryStrategy strategy,
    const Point<3> *d_points, size_t num_points,
    const float *d_queries, int num_queries,
    float *out_ms)
{
    const int blocks = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    uint32_t *d_validation = nullptr;
    cudaMalloc(&d_validation, 3 * sizeof(uint32_t));
    cudaMemset(d_validation, 0, 3 * sizeof(uint32_t));

    size_t *d_result_indices   = nullptr;
    float  *d_result_distances = nullptr;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);

    switch (strategy) {
        case LOCAL:
            knn_query_local<K_VAL><<<blocks, THREADS_PER_BLOCK>>>(
                d_points, num_points, d_queries, num_queries, d_validation);
            break;
        case GLOBAL:
            cudaMalloc(&d_result_indices,   sizeof(size_t) * num_queries * K_VAL);
            cudaMalloc(&d_result_distances, sizeof(float)  * num_queries * K_VAL);
            knn_query_global<K_VAL><<<blocks, THREADS_PER_BLOCK>>>(
                d_points, num_points, d_queries,
                d_result_indices, d_result_distances,
                num_queries, d_validation);
            break;
        case SHARED: {
            const size_t smem_bytes = THREADS_PER_BLOCK * K_VAL
                * (sizeof(size_t) + sizeof(float));
            knn_query_shared<K_VAL><<<blocks, THREADS_PER_BLOCK, smem_bytes>>>(
                d_points, num_points, d_queries, num_queries, d_validation);
            break;
        }
        default: break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        if (d_result_indices)   cudaFree(d_result_indices);
        if (d_result_distances) cudaFree(d_result_distances);
        cudaFree(d_validation);
        return false;
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(out_ms, t0, t1);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    if (d_result_indices)   cudaFree(d_result_indices);
    if (d_result_distances) cudaFree(d_result_distances);
    cudaFree(d_validation);
    return true;
}

// --------------- kernel launch + timing (cudaKDTree) ---------------

static bool launch_and_time_cukd(
    int k,
    const float3 *d_cukd_tree, int num_points,
    const float *d_queries, int num_queries,
    float *out_ms)
{
    const int blocks = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    uint64_t *d_candidate_mem = nullptr;
    cudaMalloc(&d_candidate_mem, sizeof(uint64_t) * num_queries * k);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    knn_query_cukd<<<blocks, THREADS_PER_BLOCK>>>(
        d_cukd_tree, num_points, d_queries, num_queries, d_candidate_mem, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(d_candidate_mem);
        return false;
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(out_ms, t0, t1);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_candidate_mem);
    return true;
}

// --------------- dispatch runtime K -> compile-time K ---------------

static bool dispatch_launch(
    int k, MemoryStrategy strategy,
    const Point<3> *d_points, size_t num_points,
    const float3 *d_cukd_tree,
    const float *d_queries, int num_queries,
    float *out_ms)
{
    // CUKD uses FlexHeapCandidateList (runtime k), no template dispatch needed.
    if (strategy == CUKD)
        return launch_and_time_cukd(
            k, d_cukd_tree, (int)num_points, d_queries, num_queries, out_ms);

    // Our kernels require compile-time K template parameter.
    #define DISPATCH_K(K_VAL)                                                       \
        case K_VAL:                                                                 \
            return launch_and_time<K_VAL>(                                          \
                strategy, d_points, num_points, d_queries, num_queries, out_ms);

    switch (k) {
        DISPATCH_K(8)
        DISPATCH_K(16)
        DISPATCH_K(32)
        DISPATCH_K(64)
        DISPATCH_K(128)
        DISPATCH_K(256)
        DISPATCH_K(512)
        DISPATCH_K(1024)
        default:
            printf("  [ERROR] Unsupported K=%d\n", k);
            return false;
    }
    #undef DISPATCH_K
}

// --------------- table output ---------------

static void print_table(float results[][NUM_STRATEGIES]) {
    printf("\n");
    printf("=== Average Query Time (ms) over %d runs ===\n\n", NUM_RUNS);

    // Header
    printf("  %5s |", "K");
    for (int s = 0; s < NUM_STRATEGIES; s++)
        printf(" %13s |", strategy_label[s]);
    printf("\n");

    // Separator
    printf("  ------|");
    for (int s = 0; s < NUM_STRATEGIES; s++)
        printf("---------------|");
    printf("\n");

    // Rows
    for (int ki = 0; ki < NUM_K; ki++) {
        printf("  %5d |", K_VALUES[ki]);
        for (int si = 0; si < NUM_STRATEGIES; si++) {
            if (results[ki][si] < 0)
                printf(" %13s |", "N/A");
            else
                printf(" %10.3f ms |", (double)results[ki][si]);
        }
        printf("\n");
    }
}

// --------------- main ---------------

int main() {
    const size_t total_points = (size_t)SIDE_DENSITY * SIDE_DENSITY * SIDE_DENSITY;

    printf("=== KD-Tree KNN Benchmark ===\n");
    printf("  Points: %zu | Queries: %d | Runs per config: %d\n",
           total_points, NUM_QUERIES, NUM_RUNS);
    printf("  Side length: %.1f | Side density: %d\n\n",
           (double)SIDE_LENGTH, SIDE_DENSITY);

    // --- Generate points ---
    auto *h_points = new Point<3>[total_points];
    generate_points(CUBE, h_points, SIDE_DENSITY, SIDE_LENGTH);

    Point<3> *d_points = nullptr;
    cudaMalloc(&d_points, sizeof(Point<3>) * total_points);
    cudaMemcpy(d_points, h_points, sizeof(Point<3>) * total_points, cudaMemcpyHostToDevice);

    // --- Build our KD-Tree ---
    cudaEvent_t bt0, bt1;
    cudaEventCreate(&bt0);
    cudaEventCreate(&bt1);

    cudaEventRecord(bt0);
    build_kd_tree<Point<3>>(d_points, total_points);
    cudaEventRecord(bt1);
    cudaEventSynchronize(bt1);

    float build_ms = 0;
    cudaEventElapsedTime(&build_ms, bt0, bt1);
    printf("  Our build time:  %.3f ms\n", (double)build_ms);

    // --- Build cudaKDTree (float3 copy of the same points) ---
    auto *h_f3 = new float3[total_points];
    for (size_t i = 0; i < total_points; i++)
        h_f3[i] = make_float3(h_points[i].coords[0],
                               h_points[i].coords[1],
                               h_points[i].coords[2]);

    float3 *d_cukd_tree = nullptr;
    cudaMalloc(&d_cukd_tree, sizeof(float3) * total_points);
    cudaMemcpy(d_cukd_tree, h_f3, sizeof(float3) * total_points, cudaMemcpyHostToDevice);

    cudaEventRecord(bt0);
    cukd::buildTree(d_cukd_tree, (int)total_points);
    cudaEventRecord(bt1);
    cudaEventSynchronize(bt1);

    float cukd_build_ms = 0;
    cudaEventElapsedTime(&cukd_build_ms, bt0, bt1);
    printf("  CUKD build time: %.3f ms\n\n", (double)cukd_build_ms);

    cudaEventDestroy(bt0);
    cudaEventDestroy(bt1);
    delete[] h_f3;

    // --- Generate queries ---
    auto *h_queries = new float[NUM_QUERIES * 3];
    generate_random_queries(h_queries, NUM_QUERIES, SIDE_LENGTH);

    float *d_queries = nullptr;
    cudaMalloc(&d_queries, sizeof(float) * NUM_QUERIES * 3);
    cudaMemcpy(d_queries, h_queries, sizeof(float) * NUM_QUERIES * 3, cudaMemcpyHostToDevice);

    // --- Run benchmarks ---
    float results[NUM_K][NUM_STRATEGIES];

    for (int ki = 0; ki < NUM_K; ki++) {
        const int k = K_VALUES[ki];
        printf("  K = %3d  ", k);
        fflush(stdout);

        for (int si = 0; si < NUM_STRATEGIES; si++) {
            auto strategy = static_cast<MemoryStrategy>(si);
            printf("%-6s ", strategy_label[si]);
            fflush(stdout);

            float total = 0;
            int ok = 0;

            for (int r = 0; r < NUM_RUNS; r++) {
                float ms = 0;
                if (dispatch_launch(k, strategy,
                                    d_points, total_points,
                                    d_cukd_tree,
                                    d_queries, NUM_QUERIES, &ms)) {
                    total += ms;
                    ok++;
                }
            }

            if (ok > 0) {
                results[ki][si] = total / ok;
                printf("[%.1f ms]  ", (double)results[ki][si]);
            } else {
                results[ki][si] = -1.0f;
                printf("[N/A]  ");
            }
            fflush(stdout);
        }
        printf("\n");
    }

    // --- Print results table ---
    print_table(results);

    printf("\n=== Benchmark Complete ===\n");

    // --- Cleanup ---
    cudaFree(d_points);
    cudaFree(d_cukd_tree);
    cudaFree(d_queries);
    delete[] h_points;
    delete[] h_queries;

    return 0;
}
