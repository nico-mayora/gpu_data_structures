#pragma once

constexpr int DIM = 2;

/*
 * TODO: Generalise using templates and concepts:
 * * Point struct data fields.
 * * Point dimension
 * * Do we need a "registerPointType" method?
*/
struct Point {
    char payload;
    float coords[DIM];
};