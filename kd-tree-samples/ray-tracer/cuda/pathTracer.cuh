#pragma once
#include "owl/APIHandle.h"
#include "owl/include/owl/common/math/random.h"
#include "../../common/data/world.cuh"

#ifndef K_GLOBAL_PHOTONS
constexpr int K_GLOBAL_PHOTONS = 24;
#endif
#ifndef K_CAUSTIC_PHOTONS
constexpr int K_CAUSTIC_PHOTONS = 1;
#endif

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
    owl::vec3f *accumBuffer;   // linear radiance accumulated across launches (one vec3f/pixel)
    int accumID;               // launches since last reset; 0 overwrites instead of accumulating
    owl::vec2i resolution;
    OptixTraversableHandle world;
    int depth;
    int pixel_samples;
    int num_diffuse_scattered;
    float indirect_intensity; // artistic gain on the final-gather term (1.0 = physical)
    float caustic_intensity;  // artistic gain on the caustic term (1.0 = physical)

    Photon *photon_map;
    PhotonCoord *photon_coords;
    int num_photons;
    Photon *caustic_map;
    PhotonCoord *caustic_coords;
    int num_caustic;

    struct {
        owl::vec3f pos;
        owl::vec3f dir_00;
        owl::vec3f dir_dv;
        owl::vec3f dir_du;
    } camera;

    Light *lights;
    int num_lights;
};

struct PerRayData {
    Random random;
    RayEvent event;

    const Material *hpMaterial;
    owl::vec3f albedo;       // albedo at the hit point (texture sample or flat material albedo)
    owl::vec3f hitPoint;
    owl::vec3f normalAtHp;
};