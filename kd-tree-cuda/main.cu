#include <builder.cuh>
#include <iostream>

static std::string pointCoords(const Point p) {
    return "("
        + std::to_string(p.coords[0]) + ", "
        + std::to_string(p.coords[1]) + /* ", "
        + std::to_string(p.coords[2]) + */")";
}

static void printPointBuffer(const Point *points, const int N) {
    for (int i = 0; i < N; ++i) {
        std::cout << "Coords: " << pointCoords(points[i]) << " Payload: '" << points[i].payload << "'\n";
    }
}

static int nodesInlevel(const int l) {
    return 1 << l;
}
static int numLevels(const int N) {
    return 32 - std::countl_zero(static_cast<uint32_t>(N));
}

static void printKdTree(const Point *points, const int N) {
    const int maxLvl = numLevels(N);
    int offset = 0;
    for (int l = 0; l < maxLvl; ++l) {
        const int inCurrLevel = nodesInlevel(l);
        std::cout << "Level " << l << ": ";
        for (int k = 0; k < inCurrLevel; ++k) {
            const int idx =  offset + k;
            if (idx >= N) break;
            std::cout << "[" << points[idx].payload << ": " << pointCoords(points[idx]) << "]";
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

    auto points = &points_vec[0];
    const int N = points_vec.size();

    std::cout << "Points is: \n";
    printPointBuffer(points, N);

    std::cout << "Building KDTREE... \n";
    buildKDTree(points, N);
    const auto host_points = new Point[N];
    cudaMemcpy(host_points, points, N * sizeof(Point), cudaMemcpyDeviceToHost);
    std::cout << "KDTREE is: \n";
    printKdTree(host_points, N);

    cudaFree(points);
    delete[] host_points;
}
