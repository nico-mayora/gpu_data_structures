#pragma once
#include "data.cuh"

struct QueryResult {
    size_t K;
    size_t *pointIndices;
    size_t foundPoints; /* <= K */
};

/* Return a index buffer with (at most) K values that index into the initial point buffer */
__device__ void knn(Point *tree_buf, size_t N, size_t K, float query_range, QueryResult *result);
