#pragma once
#include "owl/APIHandle.h"
#include "owl/include/owl/common/math/random.h"

enum RayTypes {
    PRIMARY,
    SHADOW,
    RAY_TYPES_COUNT
};

enum MaterialType {
    LAMBERTIAN,
    DIELECTRIC,
    CONDUCTOR,
};

struct Material {
    MaterialType matType;
    owl::vec3f albedo;
    float diffuse;
    float specular;
    float ior;
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

struct PointLight {
    owl::vec3f position;
    owl::vec3f power;
};

struct Photon {
    static constexpr uint32_t DIM = 3;
    //Required member
    float coords[DIM]; //xyz

    float colour[3];
    float power[3];
    float dir[DIM];

    /* Required method for performing queries.
     * Returns distance between this and a point buffer x.
     * We assume x's dimension is DIM.
     */

    __device__ __inline__ float dist2(const float *x) const {
        float acum = 0.;
#pragma unroll
        for (int i = 0; i < DIM; ++i) {
            const float diff = coords[i] - x[i];
            acum += diff * diff;
        }
        return acum;
    }
};

struct RayGenData {
    uint32_t *fbPtr;
    owl::vec2i resolution;
    OptixTraversableHandle world;
    int depth;
    int pixel_samples;
    int num_diffuse_scattered;

    Photon *photon_map;
    size_t num_photons;

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