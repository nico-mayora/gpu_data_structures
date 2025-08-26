#include "pt-math.h"

owl::vec3f parseVec3f(const std::string& str) {
    std::istringstream ss(str);
    std::string token;
    std::vector<float> values;

    while (std::getline(ss, token, ',')) {
        std::erase_if(token, isspace);
        values.push_back(std::stof(token));
    }

    if (values.size() != 3) {
        throw std::invalid_argument("Input string must contain exactly 3 float values");
    }

    return { values[0], values[1], values[2] };
}