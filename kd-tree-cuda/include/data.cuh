#pragma once
#include <cstdio>

constexpr int DIM = 2;

/*
 * TODO: Generalise using templates:
 * * Point struct data fields.
 * * Point dimension
 * * Do we need a "registerPointType" method?
*/
struct Point {
    char payload;
    float coords[DIM];
};

// HELPER FUNCTIONS

/* Return squared norm between to points */
__device__ __inline__ float norm2(const float *x, const float *y) {
    float acum = 0.;
#pragma unroll
    for (int i = 0; i < DIM; ++i) {
        const float diff = y[i] - x[i];
        acum += diff * diff;
    }
    return acum;
}

static __device__ __inline__ int parent_node(const int p) {
    return (p + 1) / 2 - 1;
}
