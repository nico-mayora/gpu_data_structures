#include <iostream>
#include <vector>
#include <algorithm>
#include "src/point.h"

// Helper Functions

int clz(int x) {
    return std::countl_zero((uint32_t)x);
}

int level(int i) {
    return 31 - clz(i + 1);
}

int numLevels(int N) {
    return 32 - clz(N);
}

int lChild(int i) {
    return 2 * i + 1;
}

int rChild(int i) {
    return 2 * i + 2;
}

int subtreeSize(int s, int N) {
    int L = numLevels(N);
    int l = level(s);
    int fullInner = (1 << (L - l - 1)) - 1;
    int fllc_s = ~((~s) << (L - l - 1));
    int lowest = std::min(std::max(0, N - fllc_s), 1 << (L - l - 1));
    return fullInner + lowest;
}

int segmentBegin(int s, int l, int N) {
    int L = numLevels(N);
    int nls_s = s - ((1 << l) - 1);
    int topLevels = (1 << l) - 1;
    int inner = nls_s * ((1 << (L - l - 1)) - 1);
    int lowest = std::min(nls_s * (1 << (L - l - 1)), N - ((1 << (L - 1)) - 1));
    return topLevels + inner + lowest;
}

// KD-Tree Main Function

void buildKDTree(std::vector<Point>& points, int k) {
    int N = points.size();
    int L = numLevels(N);

    for (int l = 0; l < L; ++l) {

    }
}

int main() {
    std::vector<Point> data = {
        Point({1, 1, 1}),
        Point({2, 2, 2}),
        Point({3, 3, 3}),
        Point({4, 4, 4}),
        Point({5, 5, 5})
    };

    std::cout << data.size() << std::endl;
    std::cout << data[0].get(0) << data[0].get(1) << data[0].get(2) << std::endl;

    // buildKDTree(data, 3);

    return 0;
}