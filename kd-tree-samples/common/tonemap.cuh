#pragma once
// Display transforms shared by the ray-generation program and the host-side denoise pass.

#include <cstdint>
#include "owl/common/math/vec.h"
#include "data/world.cuh"

__host__ __device__ constexpr
float hable(const float x) {
    constexpr float A = 0.15f, B = 0.50f, C = 0.10f;
    constexpr float D = 0.20f, E = 0.02f, F = 0.30f;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

// Hejl / Burgess-Dawson filmic curve (as in Tungsten); output already display-encoded.
__host__ __device__ inline
float filmic_hejl(float x) {
    x = fmaxf(0.f, x - 0.004f);
    return (x * (6.2f * x + 0.5f)) / (x * (6.2f * x + 1.7f) + 0.06f);
}

__host__ __device__ inline
owl::vec3f filter_colour(owl::vec3f colour, const float exposure, const int tonemap) {
    constexpr float W = 11.2f;
    constexpr float inv_white = 1.0f / hable(W);

    colour *= exposure;
    if (tonemap == TONEMAP_FILMIC)
        return owl::vec3f(filmic_hejl(colour.x), filmic_hejl(colour.y), filmic_hejl(colour.z));
    colour = owl::vec3f(hable(colour.x), hable(colour.y), hable(colour.z)) * inv_white;
    colour = owl::vec3f(sqrtf(colour.x), sqrtf(colour.y), sqrtf(colour.z));
    return colour;
}

// Same packing as owl::make_rgba.
__host__ __device__ inline
uint32_t pack_rgba8(const owl::vec3f &c) {
    const auto to8 = [](float f) {
        return uint32_t(fminf(255.f, fmaxf(0.f, f * 256.f)));
    };
    return to8(c.x) | (to8(c.y) << 8) | (to8(c.z) << 16) | (0xffU << 24);
}
