#include "builder.cuh"
#include "queries.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

template<int DIM>
struct PerfPoint {
    float coords[DIM];
    char payload;

    PerfPoint() = default;
    PerfPoint(const std::initializer_list<float>& coords_list, char p) : payload(p) {
        std::copy(coords_list.begin(), coords_list.end(), coords);
    }

    __host__ __device__
    float dist2(const float* x) const {
        float acum = 0.;
        for (int i = 0; i < DIM; ++i) {
            const float diff = coords[i] - x[i];
            acum += diff * diff;
        }
        return acum;
    }

    static constexpr int dimension = DIM;
};

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;

public:
    Timer(const std::string& timer_name) : name(timer_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << name << ": " << std::fixed << std::setprecision(3)
                  << duration.count() / 1000.0 << " ms" << "\n";
    }

    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

template<int DIM>
std::vector<PerfPoint<DIM>> generate_random_points(int N, float min_coord = 0.0f, float max_coord = 1000.0f) {
    std::vector<PerfPoint<DIM>> points(N);
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < DIM; ++d) {
            float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            points[i].coords[d] = min_coord + r * (max_coord - min_coord);
        }
        points[i].payload = static_cast<char>('a' + (i % 26));
    }

    return points;
}

template<int DIM, int K>
__global__ void knn_batch_kernel(const float* query_points, const PerfPoint<DIM>* tree_buf, const size_t N, const int num_queries) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    const float* query = query_points + query_idx * DIM;

    QueryResult<K>* result = alloc_query_result<K>();
    knn(query, tree_buf, N, result);

    cudaFree(result->pointIndices);
    cudaFree(result->pointDistances);
    cudaFree(result);
}

void test_build_performance() {
    std::cout << "\n=== KD-Tree Build Performance Test ===" << "\n";

    const std::vector<int> sizes = {1000, 10000, 100000, 1000000, 5000000};
    constexpr int DIM = 2;

    std::cout << std::setw(10) << "Size" << std::setw(15) << "Build Time (ms)"
              << std::setw(15) << "Points/sec" << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (int N : sizes) {
        
        auto points = generate_random_points<DIM>(N);

        
        PerfPoint<DIM>* d_points;
        cudaMalloc(&d_points, N * sizeof(PerfPoint<DIM>));
        cudaMemcpy(d_points, points.data(), N * sizeof(PerfPoint<DIM>), cudaMemcpyHostToDevice);

        
        cudaDeviceSynchronize();

        
        Timer timer("Building KD-Tree for " + std::to_string(N) + " points");
        build_kd_tree(d_points, N);
        cudaDeviceSynchronize();
        double build_time = timer.elapsed_ms();

        
        double points_per_sec = (N / build_time) * 1000.0;

        std::cout << std::setw(10) << N
                  << std::setw(15) << std::fixed << std::setprecision(2) << build_time
                  << std::setw(15) << std::fixed << std::setprecision(0) << points_per_sec << "\n";

        cudaFree(d_points);
    }
}


void test_knn_performance() {
    std::cout << "\n=== KNN Query Performance Test ===" << "\n";

    const std::vector<int> tree_sizes = {10000, 100000, 1000000};
    const std::vector<int> k_values = {1, 5, 10, 50, 100};
    constexpr int DIM = 2;
    constexpr int num_queries = 1000;

    std::cout << "Testing with " << num_queries << " queries per configuration" << "\n";
    std::cout << std::setw(10) << "Tree Size" << std::setw(10) << "K"
              << std::setw(15) << "Query Time (ms)" << std::setw(15) << "Queries/sec" << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (int N : tree_sizes) {
        
        auto points = generate_random_points<DIM>(N);

        PerfPoint<DIM>* d_points;
        cudaMalloc(&d_points, N * sizeof(PerfPoint<DIM>));
        cudaMemcpy(d_points, points.data(), N * sizeof(PerfPoint<DIM>), cudaMemcpyHostToDevice);

        build_kd_tree(d_points, N);

        
        auto query_points = generate_random_points<DIM>(num_queries);

        for (int K : k_values) {
            if (K > N) continue; 

            
            float* d_query_points;
            cudaMalloc(&d_query_points, num_queries * DIM * sizeof(float));

            
            std::vector<float> flat_queries;
            for (const auto& qp : query_points) {
                for (int d = 0; d < DIM; ++d) {
                    flat_queries.push_back(qp.coords[d]);
                }
            }
            cudaMemcpy(d_query_points, flat_queries.data(),
                      num_queries * DIM * sizeof(float), cudaMemcpyHostToDevice);

            
            constexpr int KERNEL_K = 100; 
            cudaDeviceSynchronize();
            
            Timer timer("KNN queries for K=" + std::to_string(K) + " on tree of size " + std::to_string(N));

            const int threads_per_block = 256;
            const int blocks = (num_queries + threads_per_block - 1) / threads_per_block;

            knn_batch_kernel<DIM, KERNEL_K><<<blocks, threads_per_block>>>(d_query_points, d_points, N, num_queries);
            cudaDeviceSynchronize();

            double query_time = timer.elapsed_ms();
            double queries_per_sec = (num_queries / query_time) * 1000.0;

            std::cout << std::setw(10) << N
                      << std::setw(10) << K
                      << std::setw(15) << std::fixed << std::setprecision(2) << query_time
                      << std::setw(15) << std::fixed << std::setprecision(0) << queries_per_sec << "\n";

            cudaFree(d_query_points);
        }

        cudaFree(d_points);
    }
}

void test_memory_usage() {
    std::cout << "\n=== Memory Usage Test ===" << "\n";

    const std::vector<int> sizes = {1000, 10000, 100000, 1000000};
    constexpr int DIM = 2;

    std::cout << std::setw(10) << "Size" << std::setw(15) << "Memory (MB)"
              << std::setw(15) << "Bytes/Point" << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (int N : sizes) {
        size_t free_mem_before, total_mem;
        cudaMemGetInfo(&free_mem_before, &total_mem);

        
        auto points = generate_random_points<DIM>(N);

        PerfPoint<DIM>* d_points;
        cudaMalloc(&d_points, N * sizeof(PerfPoint<DIM>));
        cudaMemcpy(d_points, points.data(), N * sizeof(PerfPoint<DIM>), cudaMemcpyHostToDevice);

        build_kd_tree(d_points, N);

        size_t free_mem_after;
        cudaMemGetInfo(&free_mem_after, &total_mem);

        size_t used_mem = free_mem_before - free_mem_after;
        double used_mb = used_mem / (1024.0 * 1024.0);
        double bytes_per_point = static_cast<double>(used_mem) / N;

        std::cout << std::setw(10) << N
                  << std::setw(15) << std::fixed << std::setprecision(2) << used_mb
                  << std::setw(15) << std::fixed << std::setprecision(1) << bytes_per_point << "\n";

        cudaFree(d_points);
    }
}

int main() {
    std::cout << "Starting KD-Tree Performance Tests...\n";
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << "\n";
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << "\n";

    try {
        test_build_performance();
        test_knn_performance();
        test_memory_usage();

        std::cout << "\nAll performance tests completed!\n";
    } catch (const std::exception& e) {
        std::cerr << "Performance test failed with exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
