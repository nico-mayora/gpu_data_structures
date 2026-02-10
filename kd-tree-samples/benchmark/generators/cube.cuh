#pragma once
#include "../../common/kdtree/data.cuh"

static void generate_cube_points(Point<3> *points, int density, float length) {
    const float step = length / (density - 1);
    const float half = length / 2.0f;
    int idx = 0;
    for (int x = 0; x < density; x++)
        for (int y = 0; y < density; y++)
            for (int z = 0; z < density; z++) {
                points[idx].coords[0] = -half + x * step;
                points[idx].coords[1] = -half + y * step;
                points[idx].coords[2] = -half + z * step;
                points[idx].payload = 0;
                idx++;
            }
}
