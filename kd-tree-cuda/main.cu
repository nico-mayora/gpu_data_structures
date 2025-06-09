#include "builder.cuh"
#include "queries.cuh"
#include <iostream>

static std::string print_coords(const Point p) {
    return "("
        + std::to_string(p.coords[0]) + ", "
        + std::to_string(p.coords[1]) + /* ", "
        + std::to_string(p.coords[2]) + */")";
}

static void print_point_buffer(const Point *points, const int N) {
    for (int i = 0; i < N; ++i) {
        std::cout << "Coords: " << print_coords(points[i]) << " Payload: '" << points[i].payload << "'\n";
    }
}

static int nodes_in_level(const int l) {
    return 1 << l;
}
static int num_levels(const int N) {
    return 32 - std::countl_zero(static_cast<uint32_t>(N));
}

static void print_kd_tree(const Point *points, const int N) {
    const int maxLvl = num_levels(N);
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


int main() {
    std::vector points_vec = {
        Point { 'a', {10., 15.} },
        Point { 'b', {46., 63.} },
        Point { 'c', {68., 21.} },
        Point { 'd', {40., 33.} },
        Point { 'e', {25., 54.} },
        Point { 'f', {15., 43.} },
        Point { 'g', {44., 58.} },
        Point { 'h', {45., 40.} },
        Point { 'i', {62., 69.} },
        Point { 'j', {53., 67.} }
    };

    const auto points = &points_vec[0];
    const int N = points_vec.size();

    std::cout << "Points is: \n";
    print_point_buffer(points, N);

    // Create device buffer for points
    Point *d_points;
    const size_t device_buffer_size = N * sizeof(points[0]);
    cudaMalloc(&d_points, device_buffer_size);
    cudaMemcpy(d_points, points, device_buffer_size, cudaMemcpyHostToDevice);

    // Build the Kd-tree
    std::cout << "Building KDTREE... \n";
    build_kd_tree(d_points, N);

    // Copy it back to the host
    const auto host_points = new Point[N];
    cudaMemcpy(host_points, d_points, device_buffer_size, cudaMemcpyDeviceToHost);

    // Print it in level-order
    std::cout << "KDTREE is: \n";
    print_kd_tree(host_points, N);

    // Clean up
    cudaFree(points);
    delete[] host_points;
}
