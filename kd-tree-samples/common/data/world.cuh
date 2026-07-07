#pragma once
#include <vector>
#include <string>
#include <cuda_runtime.h>

#include "owl/common/math/vec.h"
#include "owl/include/owl/common/math/random.h"
#include "pt-math.cuh"

struct Mesh {
    std::vector<owl::vec3i> indices;
    std::vector<owl::vec3f> vertices;
    std::vector<owl::vec3f> normals;
    std::vector<owl::vec2f> uvs; // Aligned with `vertices`. Empty when the source has no UVs.
    bool faceted = true; // True -> face normals. False -> vertex normals.

    static Mesh *makeBaseRectangle() {
        const auto mesh = new Mesh;
        mesh->vertices = {{-1,1,0}, {-1,-1,0}, {1,1,0}, {1,-1,0}};
        mesh->indices = {{0,1,2}, {1,3,2}};
        mesh->normals = {{0,0,1}, {0,0,1}};
        mesh->faceted = true;

        return mesh;
    }

    static Mesh *makeBaseCube() {
        const auto mesh = new Mesh;
        mesh->vertices = {
            {-1,-1,1}, {-1,-1,-1}, {1,-1,1}, {1,-1,-1},
            {-1,+1,1}, {-1,+1,-1}, {1,+1,1}, {1,+1,-1}
        };

        mesh->indices = {
            {0,1,2}, {1,3,2},
            {4,5,6}, {5,7,6},
            {1,7,5}, {1,3,7},
            {0,5,4}, {0,1,5},
            {2,4,6}, {2,0,4},
            {3,6,7}, {3,2,6}
        };

        mesh->faceted = true;
        for (const auto tri: mesh->indices) {
            owl::vec3f v1 = mesh->vertices.at(tri.z) - mesh->vertices.at(tri.x);
            owl::vec3f v2 = mesh->vertices.at(tri.y) - mesh->vertices.at(tri.x);
            auto normal = normalize(cross(v1, v2));
            mesh->normals.emplace_back(normal);
        }

        return mesh;
    }

    void applyTransform(const Mat4f& tf) {
        for (auto &v: vertices) {
            auto transformed_vtx = tf * owl::vec4f(v, 1);
            v = owl::vec3f(transformed_vtx);
        }

        const auto rotMatrix = tf.getRotation();
        for (auto &n: normals) {
            auto transformed_vec = rotMatrix * owl::vec4f(n, 0);
            n = normalize(owl::vec3f(transformed_vec));
        }
    }
};

// For simplicity, we only handle materials that
// are ONE of the following, not combinations.
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
    // Microfacet alpha for CONDUCTOR, approximated at scatter time as a "fuzzy
    // mirror" (reflection dir perturbed proportionally). 0 = perfect mirror.
    float roughness = 0.f;
};

struct Model {
    Mesh *mesh;
    Material *material;
    // Host-only: filesystem path to the albedo (diffuse) texture, resolved
    // absolute. Empty = untextured (fall back to material->albedo). Consumed at
    // geometry-upload time to create the per-geom OWL texture; never uploaded.
    std::string albedo_texture_path;
};

struct Camera {
    owl::vec3f lookFrom;
    owl::vec3f lookAt;
    owl::vec3f up;

    struct {
        int depth;
        int pixel_samples;
        int num_diffuse_scattered;
        float indirect_intensity; // artistic gain on indirect term (1.0 = physical)
        float caustic_intensity;  // artistic gain on caustic term (1.0 = physical)
        owl::vec2i resolution;
        float fov;
    } image;
};


enum LightType {
    LIGHT_POINT,
    LIGHT_SPOT,        // reserved for Phase 2.2
    LIGHT_DIRECTIONAL, // reserved for Phase 2.2
};

// Tagged AoS light record, uploaded as an OWL_USER_TYPE buffer. POD so it copies
// straight to the device. Only LIGHT_POINT is consumed in Phase 2.1; the spot/
// directional fields are present so the layout is stable when 2.2 lands.
struct Light {
    LightType type;
    owl::vec3f position;   // point / spot
    owl::vec3f direction;  // spot / directional (unit)
    owl::vec3f power;      // RGB radiant intensity (point/spot) or irradiance (directional)
    float cos_inner;       // spot inner cone (cos), unused otherwise
    float cos_outer;       // spot outer cone (cos), unused otherwise
};

// Back-compat alias: the photon emitter's per-launch point-light path still calls
// this a "point light". It now carries a `Light`.
using PointLight = Light;

// TODO: Remove this, deprecated
struct EmittedPhoton
{
    owl::vec3f pos;
    owl::vec3f dir;
    int power;
    owl::vec3f color;
};

struct Photon {
    static constexpr int DIM = 3;
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

    static constexpr int dimension = DIM;
};

// Parallel "coords-only" view of a photon for kd-tree traversal.
// The traversal only needs xyz (12 bytes) per node; reading the full 48-byte
// Photon wastes ~75% of memory bandwidth on the hot path.
struct PhotonCoord {
    static constexpr int DIM = 3;
    float coords[DIM];

    __device__ __inline__ float dist2(const float *x) const {
        float acum = 0.f;
#pragma unroll
        for (int i = 0; i < DIM; ++i) {
            const float diff = coords[i] - x[i];
            acum += diff * diff;
        }
        return acum;
    }

    static constexpr int dimension = DIM;
};

struct World {
    std::vector<Model*> models;
    std::vector<Light*> lights;

    Photon *photon_map;
    PhotonCoord *photon_coords;
    int num_photons;
    Photon *caustic_map;
    PhotonCoord *caustic_coords;
    int num_caustic;

    // Photon-emitter budget, set from the scene XML (<default name="casted_*_photons">).
    // Consumed by photon-mapper/main.cu; ignored by the path tracer. Defaults apply when
    // the scene omits them.
    int casted_diffuse_photons = 750'000;
    int casted_caustic_photons = 100;

    // Backdrop radiance for rays that escape the scene (<default name="sky_colour">).
    // Cosmetic only: shown to camera/specular paths by the path tracer's miss program,
    // never sampled as a light. Black when the scene omits it.
    owl::vec3f sky_colour = 0.f;

    Camera *cam;
};

// Device types

struct TrianglesGeomData {
    Material *material;
    owl::vec3f *vertex;
    owl::vec3i *index;
    owl::vec3f *normal;
    owl::vec2f *texCoord;          // null when the mesh has no UVs
    cudaTextureObject_t albedoTexture; // 0 when no albedo texture is bound
    bool faceted;
};

typedef owl::LCG<> Random;

enum RayEvent {
    MISS,
    SCATTER_DIFFUSE,
    SCATTER_SPECULAR,
    SCATTER_REFRACT,
    ABSORBED,
};