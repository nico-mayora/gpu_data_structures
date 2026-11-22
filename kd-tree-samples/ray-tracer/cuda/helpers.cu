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
owl::vec3f lightContribution(const RayGenData &self, const PerRayData &prd, const Light &light) {
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
owl::vec3f random_unit_vector(Random &rand) {
    const float theta = 2.f * float(M_PI) * rand();
    const float phi = acosf(2.f * rand() - 1.f);
    return owl::vec3f(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));
}

inline __device__
owl::vec3f reflect_or_refract_ray(const Material& material,
                                  const owl::vec3f& ray_dir,
                                  const owl::vec3f& normal,   // faces the incoming ray
                                  const bool front_face,      // ray hit the side the stored normal points to
                                  Random& rand)
{
    if (material.matType == CONDUCTOR) {
        const owl::vec3f reflected = reflect(ray_dir, normal);
        if (material.roughness > 0.f) {
            const owl::vec3f fuzzed = normalize(
                reflected + 2.f * material.roughness * random_unit_vector(rand));
            if (dot(fuzzed, normal) > 0.f)
                return fuzzed;
        }
        return reflected;
    }

    if (material.matType == DIELECTRIC) {
        // `normal` always faces the ray, so it can't tell entering from exiting:
        // the IOR ratio comes from which side was hit.
        const float cos_i = fminf(fmaxf(-dot(ray_dir, normal), 0.f), 1.f);
        const float ior = material.ior > 0.f ? material.ior : 1.5f;
        const float eta = front_face ? 1.f / ior : ior;
        float cos_t;
        const float fresnel = fresnel_dielectric(cos_i, eta, cos_t);

        if (rand() < fresnel)
            return reflect(ray_dir, normal);

        return normalize(eta * ray_dir + (eta * cos_i - cos_t) * normal);
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