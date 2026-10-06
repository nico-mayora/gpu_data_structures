#define PI_INV (float)0.3183098861

#include "../../common/kdtree/constants.cuh"

inline __device__
float norm_squared(const owl::vec3f &v) {
    return dot(v, v);
}

inline __device__
owl::vec3f cosine_weighted_hemisphere(const owl::vec3f &normal, Random &rand) {
    const float r1 = rand();
    const float r2 = rand();
    const float sqrt_r1 = sqrtf(r1);
    const float phi = 2.f * float(M_PI) * r2;

    // Local-space hemisphere direction (z-up), distribution proportional to cos(θ).
    const float lx = sqrt_r1 * cosf(phi);
    const float ly = sqrt_r1 * sinf(phi);
    const float lz = sqrtf(1.f - r1);

    // Frisvad's branchless orthonormal basis around `normal`.
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

// Contribution of a single light to the outgoing radiance at the hit point.
inline __device__
owl::vec3f lightContribution(const RayGenData &self, const PerRayData &prd, const Light &light) {
    const owl::vec3f hit = prd.hitPoint;

    owl::vec3f light_dir;    // unit vector from the surface toward the light
    float shadow_tmax;       // how far to trace the shadow ray
    float attenuation;       // 1/d^2 (point/spot) or 1 (directional)
    float spot = 1.f;        // spot cone falloff (1 for point/directional)

    if (light.type == LIGHT_DIRECTIONAL) {
        // Parallel light: arrives from -direction, no distance falloff; an occluder
        // anywhere along the ray blocks it.
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
            // cos of the angle between the spot axis and the direction to this surface.
            const float cosA = dot(-light_dir, light.direction);
            if (cosA <= light.cos_outer) return 0.f;              // outside the cone
            float t = (light.cos_inner > light.cos_outer)
                    ? (cosA - light.cos_outer) / (light.cos_inner - light.cos_outer)
                    : 1.f;
            t = fminf(fmaxf(t, 0.f), 1.f);
            spot = t * t * (3.f - 2.f * t);                       // smoothstep penumbra
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

    // L = (rho/pi) * power * attenuation * cos(theta) * spot * visibility. `power` is
    // radiant intensity (point/spot, W/sr) or irradiance (directional, W/m^2); the
    // attenuation term selects 1/d^2 vs none accordingly.
    return light_visibility
      * light_dot_norm
      * attenuation
      * spot
      * diffuse_brdf * light.power
    ;
}

// Sum direct illumination over every light. Looping (rather than stochastically
// sampling one light) is noise-free and fine for the handful of lights these scenes
// carry; revisit if a many-light scene ever needs it.
inline __device__
owl::vec3f calculateDirectIllumination(const RayGenData &self, const PerRayData &prd) {
    owl::vec3f total = 0.f;
    for (int i = 0; i < self.num_lights; ++i) {
        total += lightContribution(self, prd, self.lights[i]);
    }
    return total;
}

inline __device__
owl::vec3f reflect(const owl::vec3f &incoming, const owl::vec3f &normal) {
    return incoming - 2.f * dot(incoming, normal) * normal;
}

inline __device__
float calculate_fresnel(float ior, float cos_theta) {
    cos_theta = fabsf(cos_theta);

    // Schlick
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    float one_minus_cos = 1.0f - cos_theta;
    float one_minus_cos5 = one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos;

    return r0 + (1.0f - r0) * one_minus_cos5;
}

inline __device__
owl::vec3f calculate_refracted(const Material& material,
                               const owl::vec3f& ray_dir,
                               const owl::vec3f& normal,
                               Random& rand) {
    float cos_i = dot(-ray_dir, normal);
    float etai_over_etat;
    owl::vec3f outward_normal;

    if (cos_i > 0.0f) {
        etai_over_etat = 1.0f / material.ior;
        outward_normal = normal;
    } else {
        etai_over_etat = material.ior;
        outward_normal = -normal;
    }

    float cos_theta = fminf(dot(-ray_dir, outward_normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    if (etai_over_etat * sin_theta > 1.0f) {
        // Total internal reflection - reflect using correct normal
        return reflect(ray_dir, outward_normal);
    }

    owl::vec3f r_out_perp = etai_over_etat * (ray_dir + cos_theta * outward_normal);
    owl::vec3f r_out_parallel = -sqrtf(fabsf(1.0f - dot(r_out_perp, r_out_perp)))
                                * outward_normal;
    return r_out_perp + r_out_parallel;
}

inline __device__
owl::vec3f random_unit_vector(Random &rand) {
    const float theta = 2.f * float(M_PI) * rand();
    const float phi = acosf(2.f * rand() - 1.f);
    return owl::vec3f(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));
}

inline __device__
owl::vec3f reflect_or_refract_ray(const Material& material,
                                  const owl::vec3f& ray_dir,
                                  const owl::vec3f& normal,
                                  Random& rand)
{
    if (material.matType == CONDUCTOR) {
        const owl::vec3f reflected = reflect(ray_dir, normal);
        if (material.roughness > 0.f) {
            // "Fuzzy mirror" stand-in for a microfacet lobe: jitter the mirror
            // direction inside a sphere scaled by 2*alpha (roughly matching a GGX
            // lobe's spread). Keep the mirror dir if the jitter dips below horizon.
            const owl::vec3f fuzzed = normalize(
                reflected + 2.f * material.roughness * random_unit_vector(rand));
            if (dot(fuzzed, normal) > 0.f)
                return fuzzed;
        }
        return reflected;
    }

    if (material.matType == DIELECTRIC) {
        float cos_theta = dot(-ray_dir, normal);
        float fresnel = calculate_fresnel(material.ior, cos_theta);

        if (rand() < fresnel) {
            // Use correct normal for reflection
            owl::vec3f outward_normal = (cos_theta > 0.0f) ? normal : -normal;
            return reflect(ray_dir, outward_normal);
        }

        return calculate_refracted(material, ray_dir, normal, rand);
    }
    return 0.;
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

    // Reject photons arriving from behind the surface; the basic Lambertian estimate
    // does not weight by cos(theta) again (the photon flux already carries the
    // incident geometry), so cosTheta is only used as a validity test here.
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
    // L = (rho/pi) * sum_p dPhi_p * cone / (pi r^2 kf). photon.colour is the photon
    // flux dPhi_p; prd.albedo is rho at the gather surface; inv_normalization = 1/(pi r^2 kf).
    return into_vec3f(photon.colour) * prd.albedo * PI_INV * cone_weight * inv_normalization;
}

constexpr __device__
float hable(const float x) {
    constexpr float A = 0.15f, B = 0.50f, C = 0.10f;
    constexpr float D = 0.20f, E = 0.02f, F = 0.30f;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

inline __device__
owl::vec3f filter_colour(owl::vec3f colour) {
    constexpr float exposure = .7f;
    constexpr float W = 11.2f;
    constexpr float inv_white = 1.0f / hable(W);

    colour *= exposure;
    colour = owl::vec3f(hable(colour.x), hable(colour.y), hable(colour.z)) * inv_white;
    colour = owl::vec3f(sqrtf(colour.x), sqrtf(colour.y), sqrtf(colour.z));
    return colour;
}