//
// Created by frankete on 6/1/25.
//
#pragma once
#include <vector>


class Point {
    std::vector<float> coords;

public:
    Point(std::vector<float> coords);
    float get(int dim);
};