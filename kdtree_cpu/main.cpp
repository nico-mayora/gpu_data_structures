#include <iostream>
#include <vector>
#include <algorithm>
#include "src/point.h"

// Helper Functions

int clz(const int x) {
    return std::countl_zero(static_cast<uint32_t>(x));
}

int level(const int i) {
    return 31 - clz(i + 1);
}

int numLevels(const int N) {
    return 32 - clz(N);
}

int lChild(const int i) {
    return 2 * i + 1;
}

int rChild(const int i) {
    return 2 * i + 2;
}

int subtreeSize(const int s, const int N) {
    const auto L = numLevels(N);
    const auto l = level(s);
    const auto fullInner = (1 << (L - l - 1)) - 1;
    const auto fllc_s = ~((~s) << (L - l - 1));
    const auto lowest = std::min(std::max(0, N - fllc_s), 1 << (L - l - 1));
    return fullInner + lowest;
}

int segmentBegin(const int s, const int l, const int N) {
    const auto L = numLevels(N);
    const auto topLevels = (1 << l) - 1;
    const auto nls_s = s - topLevels;
    const auto inner = nls_s * ((1 << (L - l - 1)) - 1);
    const auto lowest = std::min(nls_s * (1 << (L - l - 1)), N - ((1 << (L - 1)) - 1));
    return topLevels + inner + lowest;
}

// KD-Tree Main Function

// void updateTags (int tags[] , Point points[] , int N , int l)
// {
//     int arrayIdx = CUDA thread index ;
//     if ( arrayIdx >= N || arrayIdx < F ( l ) )
//         /* invalid index , or already done */
//             return ;
//     int currentTag = tags [ arrayIdx ] ; // must be a node index on level l
//     int pivotPos = sb ( currentTag ) + ss ( lChild ( currentTag ) )
//     if ( arrayIdx < pivotPos )
//         tags [ arrayIdx ] = lChild ( currentTag )
//     else if ( arrayIdx > pivotPos )
//         tags [ arrayIdx ] = rChild ( currentTag )
//     else
//         tag remains unchanged ; this is the root of this sub - tree
// }

// F in the paper
int num_nodes_in_full_tree(const int depth) {
    return (1 << depth) - 1;
}

bool less(const int idx_a, const int idx_b, const int l, const std::vector<int> &tags, std::vector<Point> &points) {
    const int dim = l % 3;

    return (tags[idx_a] < tags[idx_b])
        || (tags [idx_a] == tags[idx_b])
        && (points[idx_a].get(dim) < points[idx_b].get(dim));
}

void updateTags(std::vector<int> &tags, std::vector<Point> &points, const int N, const int l, const size_t arrayIdx) {
    if (arrayIdx >= N || arrayIdx < num_nodes_in_full_tree(l)) {
        return;
    }
    const int currentTag = tags[arrayIdx];

    if (const int pivotPos = segmentBegin(currentTag, l, N) + subtreeSize(lChild(currentTag), N); arrayIdx < pivotPos) {
        tags[arrayIdx] = lChild(currentTag);
    } else if (arrayIdx > pivotPos) {
        tags[arrayIdx] = rChild(currentTag);
    } // else tag remains unchanged; this is the root of this sub - tree
}

void buildKDTree(const std::vector<Point>& points, int k) {
    const auto N = points.size();
    const auto L = numLevels(static_cast<int>(N));

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