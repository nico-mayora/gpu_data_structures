#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../common/kdtree/builder.cuh"
#include "../common/data/photon/photon-file-manager.cuh"
#include "generators/cube.cuh"
#include "kernels/knn_local.cuh"
#include "kernels/knn_global.cuh"
#include "kernels/knn_shared.cuh"

// Cube Generator Params
#define SIDE_LENGTH  40.0f
#define SIDE_DENSITY 40

// Query Params
#define K            5000
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

static void hsv_to_rgb(float h, float s, float v, float &r, float &g, float &b) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    if      (h < 60)  { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else               { r = c; g = 0; b = x; }
    r += m; g += m; b += m;
}

// Maps normalized distance t in [0,1] (0 = center, 1 = far corner) to a color.
// Warm golden-yellow at center → cool violet at edges, all bright for black backgrounds.
static void distance_to_color(float t, float &r, float &g, float &b) {
    float hue = 50.0f + t * 230.0f;   // 50° (yellow) → 280° (violet)
    float sat = 0.85f + 0.15f * t;    // slightly desaturated at center for a warm glow
    float val = 1.0f - 0.15f * t;     // slightly dimmer at edges, still bright
    hsv_to_rgb(hue, sat, val, r, g, b);
}

static void save_points_and_knn(Generator generator) {
    const size_t total_points = (size_t)SIDE_DENSITY * SIDE_DENSITY * SIDE_DENSITY;

    printf("--- Saving cube photons and KNN results ---\n");

    // Generate cube points
    auto *h_points = new Point<3>[total_points];
    generate_points(generator, h_points, SIDE_DENSITY, SIDE_LENGTH);

    // Max distance from origin to a cube corner, used to normalize the gradient
    const float half = SIDE_LENGTH / 2.0f;
    const float max_dist = sqrtf(3.0f * half * half);

    // Convert Point<3> to Photon for saving, with distance-based color gradient
    std::vector<Photon> photons(total_points);
    for (size_t i = 0; i < total_points; i++) {
        memset(&photons[i], 0, sizeof(Photon));
        photons[i].coords[0] = h_points[i].coords[0];
        photons[i].coords[1] = h_points[i].coords[1];
        photons[i].coords[2] = h_points[i].coords[2];

        float dx = h_points[i].coords[0];
        float dy = h_points[i].coords[1];
        float dz = h_points[i].coords[2];
        float t = sqrtf(dx*dx + dy*dy + dz*dz) / max_dist;
        distance_to_color(t, photons[i].colour[0], photons[i].colour[1], photons[i].colour[2]);
    }

    // Save all cube points
    PhotonFileManager::savePhotonsToFile(photons, "cube_photons.txt");

    // Upload and build KD-tree on GPU
    Point<3> *d_points = nullptr;
    cudaMalloc(&d_points, sizeof(Point<3>) * total_points);
    cudaMemcpy(d_points, h_points, sizeof(Point<3>) * total_points, cudaMemcpyHostToDevice);
    build_kd_tree<Point<3>>(d_points, total_points);

    // Single KNN query at the center of the cube (0, 0, 0)
    float h_query[3] = {0.0f, 0.0f, 0.0f};
    float *d_query = nullptr;
    cudaMalloc(&d_query, sizeof(float) * 3);
    cudaMemcpy(d_query, h_query, sizeof(float) * 3, cudaMemcpyHostToDevice);

    size_t *d_indices = nullptr;
    float  *d_distances = nullptr;
    cudaMalloc(&d_indices,   sizeof(size_t) * K);
    cudaMalloc(&d_distances, sizeof(float)  * K);

    uint32_t *d_validation = nullptr;
    cudaMalloc(&d_validation, 3 * sizeof(uint32_t));
    cudaMemset(d_validation, 0, 3 * sizeof(uint32_t));

    knn_query_global<K><<<1, 1>>>(
        d_points, total_points, d_query,
        d_indices, d_distances,
        1, d_validation
    );
    cudaDeviceSynchronize();

    // Copy back the KNN result indices
    auto *h_indices = new size_t[K];
    cudaMemcpy(h_indices, d_indices, sizeof(size_t) * K, cudaMemcpyDeviceToHost);

    // Copy back the tree-ordered points (build_kd_tree rearranges them)
    cudaMemcpy(h_points, d_points, sizeof(Point<3>) * total_points, cudaMemcpyDeviceToHost);

    // Convert KNN result points to Photon for saving, with distance-based color gradient
    std::vector<Photon> knn_photons(K);
    for (int i = 0; i < K; i++) {
        memset(&knn_photons[i], 0, sizeof(Photon));
        size_t idx = h_indices[i];
        knn_photons[i].coords[0] = h_points[idx].coords[0];
        knn_photons[i].coords[1] = h_points[idx].coords[1];
        knn_photons[i].coords[2] = h_points[idx].coords[2];

        float dx = h_points[idx].coords[0];
        float dy = h_points[idx].coords[1];
        float dz = h_points[idx].coords[2];
        float t = sqrtf(dx*dx + dy*dy + dz*dz) / max_dist;
        distance_to_color(t, knn_photons[i].colour[0], knn_photons[i].colour[1], knn_photons[i].colour[2]);
    }

    PhotonFileManager::savePhotonsToFile(knn_photons, "knn_results.txt");

    printf("  Saved %zu cube photons to cube_photons.txt\n", total_points);
    printf("  Saved %d KNN results (query at origin) to knn_results.txt\n", K);

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_query);
    cudaFree(d_indices);
    cudaFree(d_distances);
    cudaFree(d_validation);
    delete[] h_points;
    delete[] h_indices;

    printf("\n");
}

int main() {
    printf("=== KD-Tree Benchmark ===\n");
    printf("  Side length: %.1f   Side density: %d\n\n", (double)SIDE_LENGTH, SIDE_DENSITY);

    // run_benchmark(CUBE, LOCAL);
    // run_benchmark(CUBE, GLOBAL);
    // run_benchmark(CUBE, SHARED);

    save_points_and_knn(CUBE);

    printf("=== Benchmark Complete ===\n");
    return 0;
}
