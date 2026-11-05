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

    uint64_t heap[K];
    HeapQueryResult<K> result;
    result.initialize(heap);
    get_closest_k_points_in_range<K, PhotonCoord, HeapQueryResult<K>>(query, coord_map, num_photons, max_radius, &result);

    // With a bounded search that found fewer than K photons, the farthest photon's distance
    // can be arbitrarily small (1-2 photons right next to the query) and blow the density
    // up into fireflies; normalise by the search radius instead (fixed-radius estimate).
    const float radiusSqr = (max_radius < INFTY && !result.isFull())
                          ? max_radius * max_radius
                          : result.getQueryRadiusSqr();
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
void trace_primary(const RayGenData &self, const owl::vec3f &origin, const owl::vec3f &dir, PerRayData &prd) {
    uint32_t p0, p1;
    owl::packPointer(&prd, p0, p1);
    optixTrace(
        self.world,
        origin,
        dir,
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
}

// True when a masked (partially transparent) surface lets this ray through.
inline __device__
bool passes_through(const PerRayData &prd, Random &random) {
    return prd.hpMaterial->opacity < 1.f && random() >= prd.hpMaterial->opacity;
}

// Follows a ray through specular surfaces until it lands on a diffuse one, and returns the
// photon-map radiance estimate there. Emission is only counted when `include_emission`:
// for diffuse final-gather rays direct light is already handled by shadow rays (and
// light->specular->here paths by the caustic map); for coat reflections it is not.
inline __device__
owl::vec3f final_gather_ray(const RayGenData &self, owl::vec3f origin, owl::vec3f dir,
                            Random &random, const bool include_emission = false,
                            float bsdf_pdf = PDF_DELTA) {
    owl::vec3f throughput = 1.f;
    owl::vec3f result = 0.f;
    owl::vec3f prev_pos = origin;

    for (int32_t i = 0; i < self.depth; ++i) {
        PerRayData sprd;
        trace_primary(self, origin, dir, sprd);

        if (sprd.event == MISS)
            return include_emission ? result + throughput * sprd.missColour : result;

        if (include_emission)
            result += throughput * sprd.emission * emission_mis_weight(self, sprd, prev_pos, bsdf_pdf);

        origin = sprd.hitPoint;
        if (passes_through(sprd, random))
            continue;

        if (sprd.event == SCATTER_DIFFUSE) {
            if (self.num_photons > 0)
                result += throughput * gather_photons<K_GLOBAL_PHOTONS>(
                    sprd.hitPoint, self.photon_map, self.photon_coords, self.num_photons, sprd,
                    self.global_gather_radius > 0.f ? self.global_gather_radius : INFTY);
            return result;
        }

        // Glossy reflections of emitters (e.g. a worktop mirroring the window via the steel
        // splashback): sample the lights here too, MIS-weighted with the BSDF sample.
        const Material &m = *sprd.hpMaterial;
        if (include_emission && m.matType == CONDUCTOR && m.roughness >= GGX_MIN_ALPHA)
            result += throughput * sprd.albedo * glossy_direct(self, sprd, -dir, m.roughness, -1.f, random);

        throughput *= sample_specular(sprd, dir, random, bsdf_pdf);
        prev_pos = sprd.hitPoint;
        if (throughput.x <= 0.f && throughput.y <= 0.f && throughput.z <= 0.f)
            return result;
    }

    return result;
}

inline __device__
owl::vec3f trace_path(const RayGenData &self, owl::Ray &ray, PerRayData &prd,
                      owl::vec3f &guide_albedo, owl::vec3f &guide_normal) {
    // Denoiser guides come from the first non-delta vertex (seen through mirrors/glass).
    bool guide_set = false;
    const auto set_guide = [&](const owl::vec3f &a, const owl::vec3f &n) {
        if (guide_set) return;
        guide_albedo = a; guide_normal = n; guide_set = true;
    };
    owl::vec3f colour_acum = 0.f;
    owl::vec3f throughput = 1.f;
    // Last scattering vertex and the pdf its direction was sampled with (for MIS on emitters).
    owl::vec3f prev_pos = ray.origin;
    float prev_pdf = PDF_DELTA;

    for (int32_t i = 0; i < self.depth; ++i) {
        trace_primary(self, ray.origin, ray.direction, prd);

        if (prd.event == MISS) {
            colour_acum += throughput * prd.missColour;
            return colour_acum;
        }
        if (prd.event == ABSORBED)
            return colour_acum;

        // Emitters seen directly or through a chain of specular/glossy bounces.
        colour_acum += throughput * prd.emission * emission_mis_weight(self, prd, prev_pos, prev_pdf);
        if (prd.emission.x > 0.f || prd.emission.y > 0.f || prd.emission.z > 0.f)
            set_guide(owl::vec3f(1.f), prd.normalAtHp);

        if (passes_through(prd, prd.random)) {
            ray = owl::Ray(prd.hitPoint, ray.direction, EPS, INFTY);
            continue;
        }

        const owl::vec3f wo = -ray.direction;

        if (prd.event == SCATTER_SPECULAR) {
            const Material &m = *prd.hpMaterial;
            // Rough metals: sample the lights through the GGX lobe (MIS with the BSDF sample below).
            if (m.matType == CONDUCTOR && m.roughness >= GGX_MIN_ALPHA) {
                colour_acum += throughput * prd.albedo * glossy_direct(self, prd, wo, m.roughness, -1.f, prd.random);
                set_guide(throughput * prd.albedo, prd.normalAtHp);
            }

            owl::vec3f new_ray_dir = ray.direction;
            throughput *= sample_specular(prd, new_ray_dir, prd.random, prev_pdf);
            if (throughput.x <= 0.f && throughput.y <= 0.f && throughput.z <= 0.f)
                return colour_acum;
            prev_pos = prd.hitPoint;
            ray = owl::Ray(prd.hitPoint, new_ray_dir, EPS, INFTY);
            continue;
        }

        const Material &mat = *prd.hpMaterial;
        set_guide(throughput * prd.albedo, prd.normalAtHp);
        const bool black = prd.albedo.x <= 0.f && prd.albedo.y <= 0.f && prd.albedo.z <= 0.f;

        // Clear coat (plastics): GGX reflection lobe on top of the diffuse base, which keeps
        // the light the coat does not reflect (1 - F towards the viewer).
        float coat_F = 0.f;
        if (mat.coat_r0 > 0.f) {
            const owl::vec3f n = prd.normalAtHp;
            coat_F = schlick(mat.coat_r0, fmaxf(0.f, dot(wo, n)));
            owl::vec3f refl_dir;
            owl::vec3f weight;
            float pdf = PDF_DELTA;
            if (mat.coat_roughness < GGX_MIN_ALPHA) {
                refl_dir = reflect(ray.direction, n);
                weight = owl::vec3f(coat_F);
            } else {
                colour_acum += throughput * glossy_direct(self, prd, wo, mat.coat_roughness, mat.coat_r0, prd.random);
                owl::vec3f h;
                const float w = ggx_sample(n, wo, mat.coat_roughness, prd.random(), prd.random(), refl_dir, h);
                weight = owl::vec3f(w * schlick(mat.coat_r0, dot(wo, h)));
                if (w > 0.f) pdf = ggx_pdf(n, wo, refl_dir, mat.coat_roughness);
            }
            if (weight.x > 0.f)
                colour_acum += throughput * weight
                             * final_gather_ray(self, prd.hitPoint, refl_dir, prd.random,
                                                /*include_emission=*/true, pdf);
        }
        if (black) break;

        auto direct_illumination_fact = calculateDirectIllumination(self, prd);

        owl::vec3f diffuse_contrib = 0.f;
        // "Reach out" into the scene and perform gathers, this gives us global lighting with less local variance.
        if (self.num_photons > 0) {
            for (uint32_t j = 0; j < self.num_diffuse_scattered; ++j) {
                const owl::vec2f u = qmc2(prd.pixelSeed, QMC_GATHER,
                                          uint32_t(prd.sampleIndex * self.num_diffuse_scattered + j));
                const owl::vec3f diffuse_vector_dir = cosine_weighted_hemisphere_uv(prd.normalAtHp, u.x, u.y);
                diffuse_contrib += final_gather_ray(self, prd.hitPoint, diffuse_vector_dir, prd.random);
            }
        }

        constexpr float CAUSTIC_GATHER_RADIUS = 0.1f;
        owl::vec3f caustic_term = 0.f;
        if (self.num_caustic > 0) {
            caustic_term = gather_photons<K_CAUSTIC_PHOTONS>(
                prd.hitPoint, self.caustic_map, self.caustic_coords, self.num_caustic, prd,
                CAUSTIC_GATHER_RADIUS);
        }

        const float inv_M = (self.num_diffuse_scattered > 0)
                          ? 1.f / float(self.num_diffuse_scattered) : 0.f;
        colour_acum += throughput * (1.f - coat_F)
                     * (direct_illumination_fact
                        + diffuse_contrib * inv_M * prd.albedo * self.indirect_intensity
                        + caustic_term * self.caustic_intensity);
        break;
    }

    return colour_acum;
}

OPTIX_RAYGEN_PROGRAM(ptRayGen)()  {
    const RayGenData &self = owl::getProgramData<RayGenData>();
    const owl::vec2i pixelID = owl::getLaunchIndex();
    const int fbOfs = pixelID.x + self.resolution.x * pixelID.y;

    PerRayData prd;
    prd.random.init(fbOfs, self.accumID);
    prd.pixelSeed = uint32_t(fbOfs);
    prd.sampleIndex = self.accumID;
    owl::vec3f colour = 0.f;

    owl::Ray ray;
    const owl::vec2f pixelSample = qmc2(prd.pixelSeed, QMC_PIXEL, uint32_t(self.accumID));
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

    owl::vec3f guide_albedo = 0.f, guide_normal = 0.f;
    colour += trace_path(self, ray, prd, guide_albedo, guide_normal);

    if (self.albedoBuffer) {
        // Camera-space normal: x right, y up, z towards the viewer.
        const owl::vec3f right = normalize(self.camera.dir_du);
        const owl::vec3f up = normalize(self.camera.dir_dv);
        const owl::vec3f n_cam(dot(guide_normal, right), dot(guide_normal, up),
                               dot(guide_normal, cross(right, up)));
        const bool first = self.accumID == 0;
        self.albedoBuffer[fbOfs] = first ? guide_albedo : self.albedoBuffer[fbOfs] + guide_albedo;
        self.normalBuffer[fbOfs] = first ? n_cam : self.normalBuffer[fbOfs] + n_cam;
    }

    const owl::vec3f accum = (self.accumID == 0)
                           ? colour
                           : self.accumBuffer[fbOfs] + colour;
    self.accumBuffer[fbOfs] = accum;

    const owl::vec3f mean = accum / static_cast<float>(self.accumID + 1);
    self.fbPtr[fbOfs] = owl::make_rgba(filter_colour(mean, self.exposure, self.tonemap));
}


OPTIX_MISS_PROGRAM(miss)()
{
    const auto &self = owl::getProgramData<MissProgData>();
    auto &prd = owl::getPRD<PerRayData>();
    prd.event = MISS;
    prd.missColour = self.sky_colour;
    prd.emission = 0.f;
}

// Mitsuba-style bump mapping: tilt the shading normal by the height-map gradient,
// using the triangle's UV parameterisation (dp/du, dp/dv) as tangent frame.
inline __device__
owl::vec3f apply_bump(const TrianglesGeomData &self, const owl::vec3f &N, const float u, const float v, const int primID) {
    const owl::vec3i tri = self.index[primID];
    const owl::vec2f t0 = self.texCoord[tri.x], t1 = self.texCoord[tri.y], t2 = self.texCoord[tri.z];
    const owl::vec3f e1 = self.vertex[tri.y] - self.vertex[tri.x];
    const owl::vec3f e2 = self.vertex[tri.z] - self.vertex[tri.x];
    const owl::vec2f d1 = t1 - t0, d2 = t2 - t0;
    const float det = d1.x * d2.y - d1.y * d2.x;
    if (fabsf(det) < 1e-12f) return N;
    const float inv = 1.f / det;
    const owl::vec3f dp_du = (d2.y * e1 - d1.y * e2) * inv;
    const owl::vec3f dp_dv = (d1.x * e2 - d2.x * e1) * inv;

    const owl::vec2f tc = (1.f - u - v) * t0 + u * t1 + v * t2;
    constexpr float H = 1.f / 2048.f; // finite-difference step in uv
    const auto height = [&](const float x, const float y) {
        const float4 s = tex2D<float4>(self.bumpTexture, x, y);
        return (s.x + s.y + s.z) * (1.f / 3.f);
    };
    const float h0 = height(tc.x, tc.y);
    const float scale = self.material->bump_scale;
    const float grad_u = scale * (height(tc.x + H, tc.y) - h0) / H;
    const float grad_v = scale * (height(tc.x, tc.y + H) - h0) / H;

    const owl::vec3f pu = dp_du + N * (grad_u - dot(N, dp_du));
    const owl::vec3f pv = dp_dv + N * (grad_v - dot(N, dp_dv));
    owl::vec3f Nb = cross(pu, pv);
    const float len = length(Nb);
    if (!(len > 0.f)) return N;
    Nb = Nb / len;
    return dot(Nb, N) < 0.f ? -Nb : Nb;
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    auto &prd = owl::getPRD<PerRayData>();
    const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();

    const int primID = optixGetPrimitiveIndex();
    const auto [u, v] = optixGetTriangleBarycentrics();
    owl::vec3f Ng = get_normal_at_hp(self, u, v, primID);
    if (self.bumpTexture != 0 && self.texCoord != nullptr)
        Ng = apply_bump(self, Ng, u, v, primID);
    const owl::vec3f rayDir = optixGetWorldRayDirection();
    const float tMax = optixGetRayTmax();
    const owl::vec3f rayOrg = optixGetWorldRayOrigin();

    prd.hpMaterial = self.material;
    prd.albedo = get_albedo_at_hp(self, u, v, primID);
    prd.event = (self.material->matType == LAMBERTIAN) ? SCATTER_DIFFUSE : SCATTER_SPECULAR;
    prd.hitPoint = rayOrg + tMax * rayDir;
    prd.frontFace = dot(Ng, rayDir) < 0.f;
    prd.normalAtHp = prd.frontFace ? Ng : -Ng;
    prd.emission = prd.frontFace ? self.material->emission : owl::vec3f(0.f);
}

OPTIX_MISS_PROGRAM(shadow)()
{
    // we didn't hit anything, so the light is visible
    owl::vec3f &lightVisbility = owl::getPRD<owl::vec3f>();
    lightVisbility = owl::vec3f(1.f);
}

OPTIX_CLOSEST_HIT_PROGRAM(shadow)() { /* unused */ }
OPTIX_ANY_HIT_PROGRAM(shadow)() { /* unused */ }
