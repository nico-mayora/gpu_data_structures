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
                          const PerRayData &prd,
                          const float max_radius = INFTY) {
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
    // max_radius caps the search. The global map passes INFTY (adaptive): photons cover
    // every surface, so a query always finds K neighbors quickly and the kd-tree self-prunes
    // once the heap fills. The CAUSTIC map must pass a bounded radius — caustic photons are
    // localized, so an INFTY query from a pixel far from any caustic explores ~the whole tree
    // (O(N) per pixel), which on a caustic-heavy scene makes a frame effectively never finish.
    get_closest_k_points_in_range<K, PhotonCoord, HeapQueryResult<K>>(query, coord_map, num_photons, max_radius, &result);

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
    // Product of albedos along the specular prefix of the path, so radiance seen
    // through mirrors/glass is tinted by them (e.g. steel reflects ~58% gray).
    owl::vec3f throughput = 1.f;

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

        if (prd.event == MISS) {
            // Sky backdrop: cosmetic radiance for camera/specular paths that escape
            // the scene (e.g. out a window). It is NOT a light source — final-gather
            // rays ignore it and the photon maps never see it; scene lights (like the
            // kitchen's WindowLight) carry the actual energy.
            colour_acum += throughput * prd.missColour;
            return colour_acum;
        }
        if (prd.event == ABSORBED)
            return colour_acum;

        if (prd.event == SCATTER_SPECULAR) {
            owl::vec3f new_ray_dir = reflect_or_refract_ray(
                *prd.hpMaterial, ray.direction, prd.normalAtHp, prd.random
            );

            throughput *= prd.albedo;
            ray = owl::Ray(prd.hitPoint, new_ray_dir, EPS, INFTY);
            continue;
        }

        auto direct_illumination_fact = calculateDirectIllumination(self, prd);
        colour_acum += throughput * direct_illumination_fact;

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
            // dangling. The final-gather evaluates radiance leaving the SECONDARY hit
            // toward the primary, so pass sprd (not prd) for its hitPoint/normal/material.
            if (sprd.event != MISS && self.num_photons > 0) {
                // Design B: the global map includes the first (directly-lit) bounce,
                // so a single gather at the secondary hit already yields its full
                // radiance (direct + indirect) — this is the term that carries colour
                // bleeding. Adding analytic direct here would double-count.
                diffuse_contrib += gather_photons<K_GLOBAL_PHOTONS>(
                    sprd.hitPoint, self.photon_map, self.photon_coords, self.num_photons, sprd);
            }
        }

        // Caustics use a BOUNDED gather radius (unlike the adaptive global gather): caustic
        // photons are localized, so an unbounded query from a pixel far from any caustic would
        // scan ~the whole tree. The radius suits the unit-scale caustic scenes (cornell/water);
        // it's world-space, so a much larger caustic scene would want a bigger value.
        constexpr float CAUSTIC_GATHER_RADIUS = 0.1f;
        owl::vec3f caustic_term = 0.f;
        if (self.num_caustic > 0) {
            caustic_term = gather_photons<K_CAUSTIC_PHOTONS>(
                prd.hitPoint, self.caustic_map, self.caustic_coords, self.num_caustic, prd,
                CAUSTIC_GATHER_RADIUS);
        }

        // Cosine-weighted MC of the Lambertian hemisphere integral: L_indirect =
        // rho_x * (1/M) * sum_j L(y_j). The cos/pdf cancels to give rho_x (prd.albedo);
        // the 1/M is the sample average over the M final-gather rays.
        const float inv_M = (self.num_diffuse_scattered > 0)
                          ? 1.f / float(self.num_diffuse_scattered) : 0.f;
        // indirect_intensity is an artistic gain (1.0 = physically correct); it lets a
        // scene exaggerate colour bleeding where it is geometrically faint (e.g. Sponza).
        colour_acum += throughput
                     * (diffuse_contrib * inv_M * prd.albedo * self.indirect_intensity
                        + caustic_term * self.caustic_intensity);
        break;
    }

    return colour_acum;
}

OPTIX_RAYGEN_PROGRAM(ptRayGen)()  {
    const RayGenData &self = owl::getProgramData<RayGenData>();
    const owl::vec2i pixelID = owl::getLaunchIndex();
    const int fbOfs = pixelID.x + self.resolution.x * pixelID.y;

    // Progressive accumulation: each launch contributes SAMPLES_PER_FRAME samples and
    // is averaged into accumBuffer. The viewer launches one of these per displayed
    // frame, so the window stays responsive and the image refines over time. The RNG
    // is seeded with accumID so successive launches draw different (decorrelated) samples.
    constexpr int SAMPLES_PER_FRAME = 1;

    PerRayData prd;
    prd.random.init(fbOfs, self.accumID);
    owl::vec3f colour = 0.f;

    for (int sampleID = 0; sampleID < SAMPLES_PER_FRAME; sampleID++) {
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
    colour *= 1.f / float(SAMPLES_PER_FRAME);   // this launch's mean radiance (linear)

    // accumID == 0 starts a fresh accumulation (camera moved / resized); otherwise add on.
    const owl::vec3f accum = (self.accumID == 0)
                           ? colour
                           : self.accumBuffer[fbOfs] + colour;
    self.accumBuffer[fbOfs] = accum;

    // Display the running mean, tonemapped. Tonemapping happens here (not in the accum
    // buffer) so accumulation stays in linear radiance.
    const owl::vec3f mean = accum / float(self.accumID + 1);
    self.fbPtr[fbOfs] = owl::make_rgba(filter_colour(mean));
}


OPTIX_MISS_PROGRAM(miss)()
{
    const auto &self = owl::getProgramData<MissProgData>();
    auto &prd = owl::getPRD<PerRayData>();
    prd.event = MISS;
    prd.missColour = self.sky_colour;
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
    prd.albedo = get_albedo_at_hp(self, u, v, primID);
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
