#include "builder.cuh"
#include "queries.cuh"
#include <iostream>
#include <vector>

template<int DIM>
static std::string print_coords(const Point<DIM> p) {
    std::string result = "(";
#pragma unroll
    for (int d = 0; d < DIM; ++d) {
        result += std::to_string(p.coords[d]) + " ";
    }
    result.pop_back();
    return result + ")";
}

template<int DIM>
static void print_point_buffer(const Point<DIM> *points, const int N) {
    for (int i = 0; i < N; ++i) {
        std::cout << "Coords: " << print_coords(points[i]) << " Payload: '" << points[i].payload << "'\n";
    }
}

static int nodes_in_level(const int l) {
    return 1 << l;
}

static int levels_count(const int N) {
    return 32 - std::countl_zero(static_cast<uint32_t>(N));
}

template<int DIM>
static void print_kd_tree(const Point<DIM> *points, const int N) {
    const int maxLvl = levels_count(N);
    int offset = 0;
    for (int l = 0; l < maxLvl; ++l) {
        const int inCurrLevel = nodes_in_level(l);
        std::cout << "Level " << l << ": ";
        for (int k = 0; k < inCurrLevel; ++k) {
            const int idx =  offset + k;
            if (idx >= N) break;
            std::cout << "[" << points[idx].payload << ": " << print_coords(points[idx]) << "]";
        }
        std::cout << "\n";
        offset += inCurrLevel;
    }
}

// Just for testing
template<int DIM>
__global__ void fcp_kernel(const float *query_point, const Point<DIM> *tree_buf, const size_t N, FcpResult *result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result = alloc_query_result<1>();
    fcp(query_point, tree_buf, N, result);
    printf("idx: %lu\n", result->pointIndices[0]);
    cudaFree(result);
}

int main() {
    std::vector points_vec = {
        Point { {10., 15.}, 'a' },
        Point { {46., 63.}, 'b' },
        Point { {68., 21.}, 'c' },
        Point { {40., 33.}, 'd' },
        Point { {25., 54.}, 'e' },
        Point { {15., 43.}, 'f' },
        Point { {44., 58.}, 'g' },
        Point { {45., 40.}, 'h' },
        Point { {62., 69.}, 'i' },
        Point { {53., 67.}, 'j' },
    };

    const auto points = &points_vec[0];
    const int N = static_cast<int>(points_vec.size());

    std::cout << "Points is: \n";
    print_point_buffer(points, N);

    constexpr int dim = 2;

    // Create device buffer for points
    Point<dim> *d_points;
    const size_t device_buffer_size = N * sizeof(points[0]);
    cudaMalloc(&d_points, device_buffer_size);
    cudaMemcpy(d_points, points, device_buffer_size, cudaMemcpyHostToDevice);

    // Build the Kd-tree
    std::cout << "Building KDTREE... \n";
    build_kd_tree(d_points, N);

    // Copy it back to the host
    const auto host_points = new Point<2>[N];
    cudaMemcpy(host_points, d_points, device_buffer_size, cudaMemcpyDeviceToHost);

    // Print it in level-order
    std::cout << "KDTREE is: \n";
    print_kd_tree(host_points, N);

    // Perform simple query
    float query_point[2] = {68.5f, 21.9f};
    float *d_query_point;
    cudaMalloc(&d_query_point, 2 * sizeof(float));
    cudaMemcpy(d_query_point, query_point, 2 * sizeof(float), cudaMemcpyHostToDevice);

    FcpResult *d_result = nullptr;

    std::cout << "FCP is: \n";
    fcp_kernel<<<1,1>>>(d_query_point, d_points, N, d_result);
    cudaDeviceSynchronize();

    FcpResult result;
    cudaMemcpy(&result, d_result, sizeof(FcpResult), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_points);
    delete[] host_points;
}
