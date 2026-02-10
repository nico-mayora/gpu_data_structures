#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../common/kdtree/builder.cuh"
#include "generators/cube.cuh"
#include "kernels/knn_local.cuh"
#include "kernels/knn_global.cuh"
#include "kernels/knn_shared.cuh"

// Cube Generator Params
#define SIDE_LENGTH  10.0f
#define SIDE_DENSITY 100

// Query Params
#define K            500
#define NUM_QUERIES  1000000
#define THREADS_PER_BLOCK 256

enum MemoryStrategy { LOCAL, GLOBAL, SHARED };
enum Generator { CUBE };

static const char* strategy_name(MemoryStrategy s) {
    switch (s) {
        case LOCAL:  return "LOCAL  (per-thread local arrays)";
        case GLOBAL: return "GLOBAL (pre-allocated cudaMalloc)";
        case SHARED: return "SHARED (block shared memory)";
        default:     return "UNKNOWN";
    }
}

static const char* generator_name(Generator g) {
    switch (g) {
        case CUBE: return "CUBE (evenly spaced grid)";
        default:   return "UNKNOWN";
    }
}

static void generate_points(Generator g, Point<3> *points, int density, float length) {
    switch (g) {
        case CUBE:
            generate_cube_points(points, density, length);
            break;
    }
}

// Mover adentro de generator (para que cada query sea si o si adentro de la nube generada)
static void generate_random_queries(float *out, int n, float length) {
    const float half = length / 2.0f;
    srand(42);
    for (int i = 0; i < n * 3; i++)
        out[i] = ((float)rand() / RAND_MAX) * length - half;
}

static bool launch_query_kernel(
    MemoryStrategy strategy,
    const Point<3> *d_points, size_t num_points,
    const float *d_queries, int num_queries,
    uint32_t *d_validation)
{
    const int blocks = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t *d_result_indices   = nullptr;
    float  *d_result_distances = nullptr;

    switch (strategy) {
        case LOCAL: {
            knn_query_local<K><<<blocks, THREADS_PER_BLOCK>>>(
                d_points, num_points, d_queries, num_queries, d_validation
            );
            break;
        }
        case GLOBAL: {
            cudaMalloc(&d_result_indices,   sizeof(size_t) * num_queries * K);
            cudaMalloc(&d_result_distances, sizeof(float)  * num_queries * K);
            knn_query_global<K><<<blocks, THREADS_PER_BLOCK>>>(
                d_points, num_points, d_queries,
                d_result_indices, d_result_distances,
                num_queries, d_validation
            );
            break;
        }
        case SHARED: {
            const size_t smem_bytes = THREADS_PER_BLOCK * K
                * (sizeof(size_t) + sizeof(float));
            knn_query_shared<K><<<blocks, THREADS_PER_BLOCK, smem_bytes>>>(
                d_points, num_points, d_queries, num_queries, d_validation
            );
            break;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  [ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        if (d_result_indices)   cudaFree(d_result_indices);
        if (d_result_distances) cudaFree(d_result_distances);
        return false;
    }

    cudaDeviceSynchronize();

    if (d_result_indices)   cudaFree(d_result_indices);
    if (d_result_distances) cudaFree(d_result_distances);
    return true;
}

static void run_benchmark(Generator generator, MemoryStrategy strategy) {
    const size_t total_points = (size_t)SIDE_DENSITY * SIDE_DENSITY * SIDE_DENSITY;

    printf("--- %s | %s ---\n", generator_name(generator), strategy_name(strategy));
    printf("  Points: %zu   K: %d   Queries: %d\n", total_points, K, NUM_QUERIES);

    auto *h_points = new Point<3>[total_points];
    generate_points(generator, h_points, SIDE_DENSITY, SIDE_LENGTH);

    Point<3> *d_points = nullptr;
    cudaMalloc(&d_points, sizeof(Point<3>) * total_points);
    cudaMemcpy(d_points, h_points, sizeof(Point<3>) * total_points, cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    build_kd_tree<Point<3>>(d_points, total_points);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float build_ms = 0;
    cudaEventElapsedTime(&build_ms, t0, t1);
    printf("  Build time:       %.3f ms\n", (double)build_ms);

    auto *h_queries = new float[NUM_QUERIES * 3];
    generate_random_queries(h_queries, NUM_QUERIES, SIDE_LENGTH);

    float *d_queries = nullptr;
    cudaMalloc(&d_queries, sizeof(float) * NUM_QUERIES * 3);
    cudaMemcpy(d_queries, h_queries, sizeof(float) * NUM_QUERIES * 3, cudaMemcpyHostToDevice);

    uint32_t *d_validation = nullptr;
    cudaMalloc(&d_validation, 3 * sizeof(uint32_t));
    cudaMemset(d_validation, 0, 3 * sizeof(uint32_t));

    cudaEventRecord(t0);
    bool launched = launch_query_kernel(
        strategy, d_points, total_points, d_queries, NUM_QUERIES, d_validation
    );
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    if (launched) {
        float query_ms = 0;
        cudaEventElapsedTime(&query_ms, t0, t1);
        printf("  Total query time: %.3f ms\n", (double)query_ms);
        printf("  Avg per query:    %.4f us\n", (double)(query_ms * 1000.0f) / NUM_QUERIES);

        uint32_t h_val[3];
        cudaMemcpy(h_val, d_validation, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        uint32_t completed = h_val[0];
        uint32_t passed    = h_val[1];
        uint32_t failed    = h_val[2];

        if (completed != (uint32_t)NUM_QUERIES)
            printf("  [WARNING] Only %u / %d threads completed\n", completed, NUM_QUERIES);

        if (failed > 0)
            printf("  [FAIL] Closest-point check: %u / %u FAILED\n", failed, passed + failed);
        else if (passed > 0)
            printf("  Validation: %u / %u closest-point checks passed\n", passed, passed + failed);
    } else {
        printf("  [SKIPPED] Queries not run due to launch failure\n");
    }

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_points);
    cudaFree(d_queries);
    cudaFree(d_validation);
    delete[] h_points;
    delete[] h_queries;

    printf("\n");
}

int main() {
    printf("=== KD-Tree Benchmark ===\n");
    printf("  Side length: %.1f   Side density: %d\n\n", (double)SIDE_LENGTH, SIDE_DENSITY);

    run_benchmark(CUBE, LOCAL);
    run_benchmark(CUBE, GLOBAL);
    run_benchmark(CUBE, SHARED);


    printf("=== Benchmark Complete ===\n");
    return 0;
}
