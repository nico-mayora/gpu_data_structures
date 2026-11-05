#define PI_INV (float)0.3183098861

#include "../../common/kdtree/constants.cuh"
#include "../../common/microfacet.cuh"

inline __device__
float norm_squared(const owl::vec3f &v) {
    return dot(v, v);
}

inline __device__
uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// 2D R2 low-discrepancy point n (Roberts' generalised golden ratio), Cranley-Patterson
// rotated by a per-pixel, per-dimension hash so pixels stay decorrelated.
inline __device__
owl::vec2f qmc2(const uint32_t pixelSeed, const uint32_t dim, const uint32_t n) {
    const uint32_t h1 = hash_u32(pixelSeed * 0x9E3779B9U + dim * 0x85EBCA6BU + 1U);
    const uint32_t h2 = hash_u32(h1 ^ 0x68E31DA4U);
    const float rx = float(h1 >> 8) * (1.f / 16777216.f);
    const float ry = float(h2 >> 8) * (1.f / 16777216.f);
    // Integer arithmetic on 2^32-scaled constants keeps full precision for large n.
    const uint32_t ax = 3242174889U; // 0.7548776662 * 2^32
    const uint32_t ay = 2447445413U; // 0.5698402910 * 2^32
    const float x = float((ax * n) >> 8) * (1.f / 16777216.f) + rx;
    const float y = float((ay * n) >> 8) * (1.f / 16777216.f) + ry;
    return owl::vec2f(x - floorf(x), y - floorf(y));
}

// QMC dimensions used along the camera path.
enum QmcDim : uint32_t { QMC_PIXEL = 0, QMC_GATHER = 1, QMC_LIGHT0 = 2 };

inline __device__
owl::vec3f cosine_weighted_hemisphere_uv(const owl::vec3f &normal, const float r1, const float r2) {
    const float sqrt_r1 = sqrtf(r1);
    const float phi = 2.f * float(M_PI) * r2;

    const float lx = sqrt_r1 * cosf(phi);
    const float ly = sqrt_r1 * sinf(phi);
    const float lz = sqrtf(1.f - r1);

    owl::vec3f b1, b2;
    if (normal.z < -0.9999999f) {
        b1 = owl::vec3f(0.f, -1.f, 0.f);
        b2 = owl::vec3f(-1.f, 0.f, 0.f);
    } else {
        const float a = 1.f / (1.f + normal.z);
        const float b = -normal.x * normal.y * a;
        b1 = owl::vec3f(1.f - normal.x * normal.x * a, b, -normal.x);
        b2 = owl::vec3f(b, 1.f - normal.y * normal.y * a, -normal.y);
    }

    return b1 * lx + b2 * ly + normal * lz;
}

inline __device__
float traceShadowRay(const RayGenData &self, const owl::vec3f &origin, const owl::vec3f &dir, const float tmax) {
    owl::vec3f light_visibility = 0.f;
    uint32_t u0, u1;
    owl::packPointer(&light_visibility, u0, u1);
    optixTrace(
        self.world,
        origin,
        dir,
        EPS,
        tmax,
        0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT
        | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
        | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        SHADOW,
        RAY_TYPES_COUNT,
        SHADOW,
        u0, u1
    );
    return light_visibility.x;
}

// Rectangle area light: `samples` uniformly distributed points on the emitter, one-sided.
inline __device__
owl::vec3f areaLightContribution(const RayGenData &self, PerRayData &prd, const Light &light, const int li) {
    const owl::vec3f hit = prd.hitPoint;
    const int n = self.area_light_samples > 0 ? self.area_light_samples : 1;
    owl::vec3f sum = 0.f;
    for (int s = 0; s < n; ++s) {
        // Low-discrepancy points over all of this pixel's samples cover the light evenly.
        const owl::vec2f u = qmc2(prd.pixelSeed, QMC_LIGHT0 + li, uint32_t(prd.sampleIndex * n + s));
        const float a = 2.f * u.x - 1.f;
        const float b = 2.f * u.y - 1.f;
        const owl::vec3f y = light.position + a * light.edge_u + b * light.edge_v;
        const owl::vec3f to_light = y - hit;
        const float dist2 = norm_squared(to_light);
        const float dist = sqrtf(dist2);
        const owl::vec3f wi = to_light / dist;
        const float cos_l = -dot(wi, light.direction);
        const float cos_s = dot(wi, prd.normalAtHp);
        if (cos_l <= 0.f || cos_s <= 0.f) continue;
        const float vis = traceShadowRay(self, hit, wi, dist * (1.f - 1e-3f));
        sum += vis * (cos_l * cos_s * light.area / dist2) * light.power;
    }
    return sum * (prd.albedo * PI_INV) / float(n);
}

inline __device__
owl::vec3f lightContribution(const RayGenData &self, PerRayData &prd, const Light &light, const int li) {
    if (light.type == LIGHT_AREA) return areaLightContribution(self, prd, light, li);
    const owl::vec3f hit = prd.hitPoint;

    owl::vec3f light_dir;
    float shadow_tmax;
    float attenuation;
    float spot = 1.f;

    if (light.type == LIGHT_DIRECTIONAL) {
        light_dir = -light.direction;
        shadow_tmax = INFTY;
        attenuation = 1.f;
    } else {
        owl::vec3f to_light = light.position - hit;
        const float dist = sqrtf(norm_squared(to_light));
        light_dir = to_light / dist;
        shadow_tmax = dist * (1.f - EPS);
        attenuation = 1.f / (dist * dist);

        if (light.type == LIGHT_SPOT) {
            const float cosA = dot(-light_dir, light.direction);
            if (cosA <= light.cos_outer) return 0.f;
            float t = (light.cos_inner > light.cos_outer)
                    ? (cosA - light.cos_outer) / (light.cos_inner - light.cos_outer)
                    : 1.f;
            t = fminf(fmaxf(t, 0.f), 1.f);
            spot = t * t * (3.f - 2.f * t);
        }
    }

    const float light_dot_norm = dot(light_dir, prd.normalAtHp);
    if (light_dot_norm < 0.f) return 0.f;

    owl::vec3f light_visibility = 0.f;
    uint32_t u0, u1;
    owl::packPointer(&light_visibility, u0, u1);
    optixTrace(
        self.world,
        hit,
        light_dir,
        EPS,
        shadow_tmax,
        0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT
        | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
        | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        SHADOW,
        RAY_TYPES_COUNT,
        SHADOW,
        u0, u1
    );

    owl::vec3f diffuse_brdf = prd.albedo * PI_INV;

    return light_visibility
      * light_dot_norm
      * attenuation
      * spot
      * diffuse_brdf * light.power
    ;
}

inline __device__
owl::vec3f calculateDirectIllumination(const RayGenData &self, PerRayData &prd) {
    owl::vec3f total = 0.f;
    for (int i = 0; i < self.num_lights; ++i) {
        total += lightContribution(self, prd, self.lights[i], i);
    }
    return total;
}

inline __device__
owl::vec3f reflect(const owl::vec3f &incoming, const owl::vec3f &normal) {
    return incoming - 2.f * dot(incoming, normal) * normal;
}

// Unpolarised Fresnel reflectance for a dielectric interface. eta = n_incident / n_transmitted.
// Returns 1 on total internal reflection; cos_t receives the transmitted cosine otherwise.
inline __device__
float fresnel_dielectric(const float cos_i, const float eta, float &cos_t) {
    const float sin2_t = eta * eta * fmaxf(0.f, 1.f - cos_i * cos_i);
    if (sin2_t >= 1.f) { cos_t = 0.f; return 1.f; }
    cos_t = sqrtf(1.f - sin2_t);
    const float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    const float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5f * (rs * rs + rp * rp);
}

inline __device__
float schlick(const float r0, const float cos_i) {
    const float m = fminf(fmaxf(1.f - cos_i, 0.f), 1.f);
    return r0 + (1.f - r0) * m * m * m * m * m;
}

// Pdf marker for delta (mirror / refraction) and camera vertices: emission is counted in full.
constexpr float PDF_DELTA = -1.f;

// Samples the next direction at a non-diffuse surface. `dir` is updated in place and the
// path throughput must be multiplied by the returned weight (0 = path absorbed). `pdf`
// receives the solid-angle pdf of a glossy sample, or PDF_DELTA.
inline __device__
owl::vec3f sample_specular(const PerRayData &prd, owl::vec3f &dir, Random &rand, float &pdf) {
    const Material &m = *prd.hpMaterial;
    const owl::vec3f n = prd.normalAtHp; // faces the incoming ray
    pdf = PDF_DELTA;

    if (m.matType == CONDUCTOR) {
        if (m.roughness < GGX_MIN_ALPHA) {
            dir = reflect(dir, n);
            return prd.albedo;
        }
        const owl::vec3f wo = -dir;
        owl::vec3f wi, h;
        const float u1 = rand(), u2 = rand();
        const float w = ggx_sample(n, wo, m.roughness, u1, u2, wi, h);
        if (w <= 0.f) return owl::vec3f(0.f);
        pdf = ggx_pdf(n, wo, wi, m.roughness);
        dir = wi;
        return prd.albedo * w;
    }

    if (m.matType == DIELECTRIC) {
        const float cos_i = fminf(fmaxf(-dot(dir, n), 0.f), 1.f);
        const float ior = m.ior > 0.f ? m.ior : 1.5f;
        const float eta = (prd.frontFace || m.thin) ? 1.f / ior : ior;
        float cos_t;
        float F = fresnel_dielectric(cos_i, eta, cos_t);

        if (m.thin) {
            // Thin slab: inter-reflections between the two faces, no net bending.
            if (F < 1.f) F = 2.f * F / (1.f + F);
            if (m.transmittance <= 0.f) { dir = reflect(dir, n); return owl::vec3f(F); }
            if (rand() < F) dir = reflect(dir, n);
            else return owl::vec3f(m.transmittance);
            return owl::vec3f(1.f);
        }
        if (m.transmittance <= 0.f) {
            // Opaque glossy glass (e.g. oven door): only the Fresnel reflection survives.
            dir = reflect(dir, n);
            return owl::vec3f(F);
        }
        if (rand() < F) {
            dir = reflect(dir, n);
            return owl::vec3f(1.f);
        }
        dir = normalize(eta * dir + (eta * cos_i - cos_t) * n);
        return owl::vec3f(m.transmittance);
    }
    return owl::vec3f(0.f);
}

// Light sampling for a GGX lobe at prd.hitPoint (area lights only), MIS-weighted against
// BSDF sampling. Returns radiance * BRDF * cos / pdf; Fresnel is Schlick(r0) per sample
// when r0 >= 0, otherwise 1 (the caller applies the conductor tint).
inline __device__
owl::vec3f glossy_direct(const RayGenData &self, const PerRayData &prd, const owl::vec3f &wo,
                         const float alpha, const float r0, Random &rand) {
    const int n = self.area_light_samples > 0 ? self.area_light_samples : 1;
    owl::vec3f sum = 0.f;
    for (int li = 0; li < self.num_lights; ++li) {
        const Light &light = self.lights[li];
        if (light.type != LIGHT_AREA) continue;
        for (int s = 0; s < n; ++s) {
            const float a = 2.f * ((float(s) + rand()) / float(n)) - 1.f;
            const float b = 2.f * rand() - 1.f;
            const owl::vec3f y = light.position + a * light.edge_u + b * light.edge_v;
            const owl::vec3f to_light = y - prd.hitPoint;
            const float dist2 = norm_squared(to_light);
            const float dist = sqrtf(dist2);
            const owl::vec3f wi = to_light / dist;
            const float cos_l = -dot(wi, light.direction);
            if (cos_l <= 0.f) continue;
            const float fcos = ggx_eval_cos(prd.normalAtHp, wo, wi, alpha);
            if (fcos <= 0.f) continue;
            const float p_light = dist2 / (cos_l * light.area);
            const float w = power_heuristic(float(n) * p_light, ggx_pdf(prd.normalAtHp, wo, wi, alpha));
            const float F = (r0 >= 0.f) ? schlick(r0, dot(wo, normalize(wo + wi))) : 1.f;
            const float vis = traceShadowRay(self, prd.hitPoint, wi, dist * (1.f - 1e-3f));
            sum += (vis * F * fcos * w / p_light) * light.power;
        }
    }
    return sum / float(n);
}

// MIS weight for emission found by a BSDF-sampled ray that left `from` with `bsdf_pdf`.
inline __device__
float emission_mis_weight(const RayGenData &self, const PerRayData &hit, const owl::vec3f &from,
                          const float bsdf_pdf) {
    const float area = hit.hpMaterial->emitter_area;
    if (bsdf_pdf <= 0.f || area <= 0.f) return 1.f;
    const owl::vec3f d = hit.hitPoint - from;
    const float dist2 = norm_squared(d);
    const float cos_l = fabsf(dot(d, hit.normalAtHp)) / sqrtf(dist2);
    if (cos_l <= 0.f) return 1.f;
    const int n = self.area_light_samples > 0 ? self.area_light_samples : 1;
    return power_heuristic(bsdf_pdf, float(n) * dist2 / (cos_l * area));
}

inline __device__
owl::vec3f into_vec3f(const float *arr) {
    return owl::vec3f(arr[0], arr[1], arr[2]);
}

#define LINEAR

inline __device__
owl::vec3f calculate_photon_contrib(
    const Photon& photon, 
    const PerRayData& prd, 
    const float inv_radius,
    const float inv_k,
    const float inv_normalization) 
{
    const owl::vec3f diff = into_vec3f(photon.coords) - prd.hitPoint;
    const float distance = length(diff); 

    const owl::vec3f wi = -into_vec3f(photon.dir);
    const float cosTheta = max(0.f, dot(prd.normalAtHp, wi));

    if (cosTheta <= EPS) return 0.f;

    const float ratio = distance * inv_radius;
    float p_term;

#if defined(CUBIC)
    p_term = cbrtf(ratio);
#elif defined(QUADRATIC)
    p_term = sqrtf(ratio);
#else
    p_term = ratio;
#endif

    const float cone_weight = max(0.f, 1.0f - (p_term * inv_k));
    return into_vec3f(photon.colour) * prd.albedo * PI_INV * cone_weight * inv_normalization;
}

#include "../../common/tonemap.cuh"
