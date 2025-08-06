#pragma once
#include <array>
#include <stdexcept>
#include <vector>

#include "owl/common/math/vec.h"

class Mat4f {
    std::array<float, 16> data{};

public:
    Mat4f() : data{0} {}

    /* Create the matrix from 16 column-major numbers separated by spaces. */
    explicit Mat4f(const std::string& str) {
        std::istringstream iss(str);
        float value;
        size_t index = 0;

        // Initialize array to identity matrix in case parsing fails
        data = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

        while (iss >> value && index < 16) {
            data[index++] = value;
        }

        if (index != 16 || iss >> value) {
            throw std::invalid_argument("Matrix string must contain exactly 16 space-separated numbers");
        }
    }

    float& operator()(const size_t row, const size_t col) {
        if (row >= 4 || col >= 4)
            throw std::out_of_range("Matrix index out of bounds");
        return data[row * 4 + col];
    }

    const float& operator()(const size_t row, const size_t col) const {
        if (row >= 4 || col >= 4)
            throw std::out_of_range("Matrix index out of bounds");
        return data[row * 4 + col];
    }

    // Matrix addition
    Mat4f operator+(const Mat4f& other) const {
        Mat4f result;
        for (size_t i = 0; i < 16; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    // Matrix subtraction
    Mat4f operator-(const Mat4f& other) const {
        Mat4f result;
        for (size_t i = 0; i < 16; ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    // Matrix multiplication
    Mat4f operator*(const Mat4f& other) const {
        Mat4f result;
        for (size_t row = 0; row < 4; ++row) {
            for (size_t col = 0; col < 4; ++col) {
                float sum = 0.0f;
                for (size_t k = 0; k < 4; ++k) {
                    sum += (*this)(row, k) * other(k, col);
                }
                result(row, col) = sum;
            }
        }
        return result;
    }

    // Matrix-vector multiplication
    owl::vec4f operator*(const owl::vec4f& vec) const {
        return {
            data[0] * vec[0] + data[1] * vec[1] + data[2] * vec[2] + data[3] * vec[3],
            data[4] * vec[0] + data[5] * vec[1] + data[6] * vec[2] + data[7] * vec[3],
            data[8] * vec[0] + data[9] * vec[1] + data[10] * vec[2] + data[11] * vec[3],
            data[12] * vec[0] + data[13] * vec[1] + data[14] * vec[2] + data[15] * vec[3]
        };
    }

    // Scalar multiplication
    Mat4f operator*(const float scalar) const {
        Mat4f result;
        for (size_t i = 0; i < 16; ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    // Print matrix (for debugging)
    void print() const {
        for (size_t row = 0; row < 4; ++row) {
            for (size_t col = 0; col < 4; ++col) {
                std::cout << (*this)(row, col) << " ";
            }
            std::cout << std::endl;
        }
    }
};

owl::vec3f parseVec3f(const std::string& str);