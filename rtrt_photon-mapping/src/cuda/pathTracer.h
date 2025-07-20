#pragma once
#include "owl/APIHandle.h"
#include "owl/include/owl/common/math/random.h"

enum RayTypes {
    PRIMARY,
    RAY_TYPES_COUNT
};

enum MaterialType {
    LAMBERTIAN,
    EMISSIVE
};

struct Material {
    MaterialType matType;
    owl::vec3f albedo;
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

    struct {
        owl::vec3f pos;
        owl::vec3f dir_00;
        owl::vec3f dir_dv;
        owl::vec3f dir_du;
    } camera;
};

typedef owl::LCG<> Random;

enum RayEvent {
    MISSED,
    REFLECTED_DIFFUSE,
    CANCELLED
};

struct PerRayData {
    Random random;
    RayEvent event;

    struct {
        owl::vec3f emitted;
        owl::vec3f reflected;
    } colour;

    owl::vec3f hitPoint;
    owl::vec3f normalAtHp;
};