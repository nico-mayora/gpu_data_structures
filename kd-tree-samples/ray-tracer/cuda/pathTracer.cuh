#pragma once
#include "owl/APIHandle.h"
#include "owl/include/owl/common/math/random.h"
#include "../../common/data/world.cuh"

#ifndef K_GLOBAL_PHOTONS
constexpr int K_GLOBAL_PHOTONS = 32;
#endif
#ifndef K_CAUSTIC_PHOTONS
constexpr int K_CAUSTIC_PHOTONS = 128;
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
    owl::vec3f *albedoBuffer;  // denoiser guides, accumulated like accumBuffer (null = off)
    owl::vec3f *normalBuffer;
    int accumID;               // launches since last reset; 0 overwrites instead of accumulating
    owl::vec2i resolution;
    OptixTraversableHandle world;
    int depth;
    int num_diffuse_scattered;
    float indirect_intensity;
    float caustic_intensity;
    float exposure;
    int tonemap;
    int area_light_samples;
    float global_gather_radius;   // <= 0: unbounded

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
    owl::vec3f albedo;
    owl::vec3f hitPoint;
    owl::vec3f normalAtHp;   // shading normal, flipped to face the incoming ray
    owl::vec3f missColour;
    owl::vec3f emission;     // radiance emitted towards the ray origin (0 unless an emitter's front face)
    bool frontFace;          // ray hit the side the stored normal points to
    // Low-discrepancy sampling state of the camera path (see qmc2): a per-pixel seed and
    // the index of the current sample (accumID).
    uint32_t pixelSeed;
    int sampleIndex;
};