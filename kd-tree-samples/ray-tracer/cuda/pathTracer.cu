#include "pathTracer.cuh"
#include "helpers.cu"
#include "../../common/kdtree/queries.cuh"
#include "../../common/helpers.cuh"
#include <optix_device.h>

template <int K>
inline __device__
owl::vec3f gather_photons(const owl::vec3f &query_pos,
                          const Photon *photon_map,
                          const PhotonCoord *coord_map,
                          const int num_photons,
                          const PerRayData &prd) {
    constexpr float k_filter = 1.f;
    constexpr float inv_k = 1.f / k_filter;
#if defined(CUBIC)
    constexpr float p_val = 1.0f / 3.0f;
#elif defined(QUADRATIC)
    constexpr float p_val = 0.5f;
#else
    constexpr float p_val = 1.0f;
#endif
    constexpr float kernel_factor = 1.0f - 2.0f / (k_filter * (p_val + 2.0f));

    const float query[] = { query_pos.x, query_pos.y, query_pos.z };

    // Heap lives in per-thread local memory: CUDA interleaves it across the warp,
    // so warp accesses to result.photonData[i] coalesce into 1 cache line instead
    // of 32 (which is what the previous threadID*K layout in global memory caused).
    uint64_t heap[K];
    HeapQueryResult<K> result;
    result.initialize(heap);
    // Traverse the coords-only array (12 bytes/node) — full photons (48 bytes)
    // are only loaded for the K survivors below.
    get_closest_k_points_in_range<K, PhotonCoord, HeapQueryResult<K>>(query, coord_map, num_photons, 0.01,&result);

    const float radiusSqr = result.getQueryRadiusSqr();
    const float inv_radius = 1.f / sqrtf(radiusSqr);
    const float inv_normalization = 1.0f / (M_PI * radiusSqr * kernel_factor);

    owl::vec3f illumination = 0.f;
    for (int p = 0; p < K; p++) {
        if (result.getDistance(p) == INFTY) break;
        const Photon &photon = photon_map[result.getIndex(p)];
        illumination += calculate_photon_contrib(photon, prd, inv_radius, inv_k, inv_normalization);
    }
    return illumination;
}

inline __device__
owl::vec3f trace_path(const RayGenData &self, owl::Ray &ray, PerRayData &prd) {
    owl::vec3f colour_acum = 0.f;

    for (int32_t i = 0; i < self.depth; ++i) {
        uint32_t p0, p1;
        owl::packPointer(&prd, p0, p1);
        optixTrace(
            self.world,
            ray.origin,
            ray.direction,
            EPS,
            INFTY,
            0.f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            PRIMARY,
            RAY_TYPES_COUNT,
            PRIMARY,
            p0, p1
        );

        if (prd.event == MISS || prd.event == ABSORBED)
            return colour_acum;

        if (prd.event == SCATTER_SPECULAR) {
            owl::vec3f new_ray_dir = reflect_or_refract_ray(
                *prd.hpMaterial, ray.direction, prd.normalAtHp, prd.random
            );

            ray = owl::Ray(prd.hitPoint, new_ray_dir, EPS, INFTY);
            continue;
        }

        auto direct_illumination_fact = calculateDirectIllumination(self, prd);
        colour_acum += direct_illumination_fact;

        owl::vec3f diffuse_contrib = 0.f;
        // "Reach out" into the scene and perform gathers, this gives us global lighting with less local variance.
        for (uint32_t j = 0; j < self.num_diffuse_scattered; ++j) {
            const owl::vec3f diffuse_vector_dir =
                cosine_weighted_hemisphere(prd.normalAtHp, prd.random);

            uint32_t q0, q1;
            PerRayData sprd;
            owl::packPointer(&sprd, q0, q1);
            optixTrace(
                self.world,
                prd.hitPoint,
                diffuse_vector_dir,
                EPS,
                INFTY,
                0.f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                PRIMARY,
                RAY_TYPES_COUNT,
                PRIMARY,
                q0, q1
            );

            // Skip MISS: the miss program only sets `event`, leaving sprd.hpMaterial
            // dangling. The final-gather evaluates the photon density / BRDF at the
            // SECONDARY hit, so pass sprd (not prd) for its hitPoint/normal/material.
            if (sprd.event != MISS) {
                diffuse_contrib += gather_photons<K_GLOBAL_PHOTONS>(
                    sprd.hitPoint, self.photon_map, self.photon_coords, self.num_photons, sprd);
            }
        }

        const owl::vec3f caustic_term = gather_photons<K_CAUSTIC_PHOTONS>(
            prd.hitPoint, self.caustic_map, self.caustic_coords, self.num_caustic, prd);

        const float inv_N = 1.f / float(self.num_diffuse_scattered);
        colour_acum += diffuse_contrib * inv_N * prd.hpMaterial->albedo + caustic_term;
        break;
    }

    return colour_acum;
}

OPTIX_RAYGEN_PROGRAM(ptRayGen)()  {
    const RayGenData &self = owl::getProgramData<RayGenData>();
    const owl::vec2i pixelID = owl::getLaunchIndex();

    PerRayData prd;
    prd.random.init(pixelID.x,pixelID.y);
    owl::vec3f colour = 0.f;

    for (int sampleID=0; sampleID < self.pixel_samples; sampleID++) {
        owl::Ray ray;

        const owl::vec2f pixelSample(prd.random(),prd.random());
        const owl::vec2f screen
          = (owl::vec2f(pixelID)+pixelSample)
          / owl::vec2f(self.resolution);
        const owl::vec3f origin = self.camera.pos;
        const owl::vec3f direction
            = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);

        ray.origin = origin;
        ray.direction = direction;

        colour += trace_path(self, ray, prd);
    }

    colour *= 1.f / self.pixel_samples;
    colour = filter_colour(colour);

    const int fbOfs = pixelID.x+self.resolution.x*pixelID.y;
    self.fbPtr[fbOfs] = owl::make_rgba(colour);
}


OPTIX_MISS_PROGRAM(miss)()
{
    auto &prd = owl::getPRD<PerRayData>();
    prd.event = MISS;
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    auto &prd = owl::getPRD<PerRayData>();
    const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();

    const int primID = optixGetPrimitiveIndex();
    const auto [u, v] = optixGetTriangleBarycentrics();
    const owl::vec3f Ng = get_normal_at_hp(self, u, v, primID);
    const owl::vec3f rayDir = optixGetWorldRayDirection();
    const float tMax = optixGetRayTmax();
    const owl::vec3f rayOrg = optixGetWorldRayOrigin();

    prd.hpMaterial = self.material;
    prd.event = (self.material->matType == LAMBERTIAN) ? SCATTER_DIFFUSE : SCATTER_SPECULAR;
    prd.hitPoint = rayOrg + tMax * rayDir;
    prd.normalAtHp = (dot(Ng, rayDir) > 0.f) ? -Ng : Ng;
}

OPTIX_MISS_PROGRAM(shadow)()
{
    // we didn't hit anything, so the light is visible
    owl::vec3f &lightVisbility = owl::getPRD<owl::vec3f>();
    lightVisbility = owl::vec3f(1.f);
}

OPTIX_CLOSEST_HIT_PROGRAM(shadow)() { /* unused */ }
OPTIX_ANY_HIT_PROGRAM(shadow)() { /* unused */ }
