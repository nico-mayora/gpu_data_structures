//
// Created by frankete on 6/1/25.
//

#include "point.h"

Point::Point(std::vector<float> coords) {
    this->coords = coords;
}

float Point::get(int dim) {
    return coords[dim];
}
