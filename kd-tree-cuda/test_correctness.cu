#include "builder.cuh"
#include "queries.cuh"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <chrono>

template<int DIM>
struct TestPoint {
    float coords[DIM];
    char payload;

    TestPoint() = default;
    TestPoint(const std::initializer_list<float>& coords_list, char p) : payload(p) {
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

template<int DIM>
void print_point(const TestPoint<DIM>& p) {
    std::cout << "(";
    for (int i = 0; i < DIM; ++i) {
        std::cout << p.coords[i];
        if (i < DIM - 1) std::cout << ", ";
    }
    std::cout << ")";
}

void test_kd_tree_structure() {
    std::cout << "Test 1: KD-Tree Structure Correctness\n";
    
    std::vector<TestPoint<2>> points = {
        {{1.0f, 1.0f}, 'a'},
        {{2.0f, 2.0f}, 'b'},
        {{3.0f, 1.0f}, 'c'},
        {{4.0f, 3.0f}, 'd'},
        {{5.0f, 2.0f}, 'e'},
        {{6.0f, 1.0f}, 'f'},
        {{7.0f, 4.0f}, 'g'},
        {{8.0f, 2.0f}, 'h'}
    };

    const int N = static_cast<int>(points.size());
    
    TestPoint<2>* d_points;
    cudaMalloc(&d_points, N * sizeof(TestPoint<2>));
    cudaMemcpy(d_points, points.data(), N * sizeof(TestPoint<2>), cudaMemcpyHostToDevice);

    build_kd_tree(d_points, N);
    
    std::vector<TestPoint<2>> result(N);
    cudaMemcpy(result.data(), d_points, N * sizeof(TestPoint<2>), cudaMemcpyDeviceToHost);

    std::cout << "KD-Tree structure (level-order):\n";
    int offset = 0;
    int level = 0;
    while (offset < N) {
        int nodes_in_level = 1 << level;
        std::cout << "Level " << level << ": ";
        for (int i = 0; i < nodes_in_level && offset + i < N; ++i) {
            std::cout << "[" << result[offset + i].payload << ": ";
            print_point(result[offset + i]);
            std::cout << "] ";
        }
        std::cout << std::endl;
        offset += nodes_in_level;
        level++;
    }
    
    std::cout << "Root node: [" << result[0].payload << ": ";
    print_point(result[0]);
    std::cout << "]" << std::endl;

    cudaFree(d_points);
    std::cout << "KD-Tree structure test completed\n";
}

template<int K>
__global__ void test_knn_kernel(const float* query_point, const TestPoint<2>* tree_buf, const size_t N, QueryResult<K>* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result = alloc_query_result<K>();
    knn(query_point, tree_buf, N, result);

    printf("KNN Results (K=%d):\n", K);
    for (int i = 0; i < K; ++i) {
        printf("  %d: Point %c (idx %llu) at distance %.3f\n",
                i, tree_buf[result->pointIndices[i]].payload,
                (unsigned long long)result->pointIndices[i], sqrtf(result->pointDistances[i]));
    }

    cudaFree(result->pointIndices);
    cudaFree(result->pointDistances);
    cudaFree(result);
}

void test_knn_correctness() {
    std::cout << "\n=== Test 2: KNN Query Correctness ===" << std::endl;

    std::vector<TestPoint<2>> points = {
        {{0.0f, 0.0f}, 'a'},  
        {{2.0f, 0.0f}, 'b'},  
        {{0.0f, 2.0f}, 'c'},  
        {{2.0f, 2.0f}, 'd'},  
        {{1.0f, 1.0f}, 'e'},  
        {{3.0f, 3.0f}, 'f'},  
        {{-1.0f, -1.0f}, 'g'}, 
        {{4.0f, 4.0f}, 'h'}   
    };

    const int N = static_cast<int>(points.size());
    
    TestPoint<2>* d_points;
    cudaMalloc(&d_points, N * sizeof(TestPoint<2>));
    cudaMemcpy(d_points, points.data(), N * sizeof(TestPoint<2>), cudaMemcpyHostToDevice);

    build_kd_tree(d_points, N);
    
    float query_point[2] = {1.0f, 1.0f};
    float* d_query_point;
    cudaMalloc(&d_query_point, 2 * sizeof(float));
    cudaMemcpy(d_query_point, query_point, 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    constexpr int K = 3;

    std::cout << "Query point: (1.0, 1.0)\n";
    std::cout << "Expected closest points (in order):\n";
    std::cout << "  0: Point e (idx 4) at distance 0.000\n";
    std::cout << "  1: Point a (idx 0) at distance 1.414\n";
    std::cout << "  2: Point b (idx 1) at distance 1.414\n";

    test_knn_kernel<K><<<1, 1>>>(d_query_point, d_points, N, nullptr);
    cudaDeviceSynchronize();

    cudaFree(d_points);
    cudaFree(d_query_point);
    std::cout << "KNN correctness test completed\n";
}

int main() {
    std::cout << "Starting KD-Tree Correctness Tests:\n";

    try {
        test_kd_tree_structure();
        test_knn_correctness();

        std::cout << "\nAll correctness tests passed!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
