#pragma once
#include "../../common/kdtree/constants.cuh"
#include "../../common/kdtree/data.cuh"

// validation[0] = threads completed
// validation[1] = brute-force checks passed
// validation[2] = brute-force checks failed
#define VALIDATION_STRIDE 10000

template<int K_VAL>
__device__ __inline__ void validate_knn(
    const Point<3> *tree, size_t num_points,
    const float *qp,
    const size_t *my_indices,
    uint32_t *validation,
    int tid)
{
    atomicAdd(&validation[0], 1);

    if (tid % VALIDATION_STRIDE != 0) return;

    // Brute-force find the actual closest point
    float min_dist = INFTY;
    size_t min_idx = 0;
    for (size_t i = 0; i < num_points; i++) {
        float d = tree[i].dist2(qp);
        if (d < min_dist) {
            min_dist = d;
            min_idx = i;
        }
    }

    // Check that it appears somewhere in the K results
    bool found = false;
    for (int k = 0; k < K_VAL; k++) {
        if (my_indices[k] == min_idx) {
            found = true;
            break;
        }
    }

    if (found) atomicAdd(&validation[1], 1);
    else       atomicAdd(&validation[2], 1);
}
