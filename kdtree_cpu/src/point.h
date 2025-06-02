//
// Created by frankete on 6/1/25.
//

#ifndef POINT_H
#define POINT_H
#include <vector>

class Point {
    std::vector<float> coords;

public:
    Point(std::vector<float> coords);
    float get(int dim);
};

#endif //POINT_H
