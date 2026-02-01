#pragma once
#include "owl/APIHandle.h"
#include "include/owl/common/math/random.h"
#include "../../common/data/world.cuh"

enum RayTypes {
    PRIMARY,
    SHADOW,
    RAY_TYPES_COUNT
};

struct MissProgData {
    owl::vec3f sky_colour;
};

struct RayGenData {
    uint32_t *fbPtr;
    owl::vec2i resolution;
    OptixTraversableHandle world;
    int depth;
    int pixel_samples;
    int num_diffuse_scattered;

    Photon *photon_map;
    int num_photons;
    Photon *caustic_map;
    int num_caustic;

    size_t *heap_indices;
    float *heap_distances;

    struct {
        owl::vec3f pos;
        owl::vec3f dir_00;
        owl::vec3f dir_dv;
        owl::vec3f dir_du;
    } camera;

    PointLight *scene_light;
};

struct PerRayData {
    Random random;
    RayEvent event;
    owl::vec3f colour;

    Material hpMaterial;
    owl::vec3f hitPoint;
    owl::vec3f normalAtHp;
};