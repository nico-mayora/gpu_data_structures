#pragma once
#include "owl/APIHandle.h"
#include "owl/include/owl/common/math/random.h"
#include "../../common/data/world.cuh"

enum RayTypes {
    PRIMARY,
    SHADOW,
    RAY_TYPES_COUNT
};

struct TrianglesGeomData {
    Material *material;
    owl::vec3f *vertex;
    owl::vec3i *index;
    owl::vec3f *normal;
    bool faceted;
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

    struct {
        owl::vec3f pos;
        owl::vec3f dir_00;
        owl::vec3f dir_dv;
        owl::vec3f dir_du;
    } camera;

    PointLight *scene_light;
};

typedef owl::LCG<> Random;

enum RayEvent {
    MISSED,
    REFLECTED_DIFFUSE,
    REFLECTED_SPECULAR,
    CANCELLED
};

struct PerRayData {
    Random random;
    RayEvent event;
    owl::vec3f colour;

    Material hpMaterial;
    owl::vec3f hitPoint;
    owl::vec3f normalAtHp;
};