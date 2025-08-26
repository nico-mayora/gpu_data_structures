#pragma once
#include "owl/APIHandle.h"
#include "owl/include/owl/common/math/random.h"

enum RayTypes {
    PRIMARY,
    SHADOW,
    RAY_TYPES_COUNT
};

enum MaterialType {
    LAMBERTIAN
};

enum ScatterEvent {
    SCATTER_DIFFUSE = 1,
    SCATTER_SPECULAR = 2,
    SCATTER_REFRACT = 4,
    ABSORBED = 8,
    MISS = 16
};

struct Material {
    MaterialType matType;
    owl::vec3f albedo;
    float diffuse;
    float specular;
    float transmission;
    float refraction_idx;
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
    int light_samples;

        struct { // we'll just support one light in the xy plane to begin with.
            owl::vec3f centre;
            owl::vec2f sides;
            owl::vec3f radiance; // RGB channels
        } lightSource;

    struct {
        owl::vec3f pos;
        owl::vec3f dir_00;
        owl::vec3f dir_dv;
        owl::vec3f dir_du;
    } camera;
};

typedef owl::LCG<> Random;

struct Photon {
    static constexpr uint32_t DIM = 3;
    //Required member
    float coords[DIM]; //xyz

    float colour[3];
    float power[3];
    float dir[DIM];
    bool isDead;

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

struct PerRayData {
    owl::LCG<> random;
    owl::vec3f colour;
    ScatterEvent event;
    
    struct {
        owl::vec3f origin;
        owl::vec3f direction;
        owl::vec3f color;
    } scattered;
};

struct Light {
    Photon *photons;
    OptixTraversableHandle world;
    owl::vec3f position;
    owl::vec3f colour;
    float power;
    
    int type;
    int maxDepth;
};
