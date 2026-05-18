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

inline __device__
owl::vec3f calculateDirectIllumination(const RayGenData &self, const PerRayData &prd) {
    auto light = self.scene_light;
    auto shadow_ray_org = prd.hitPoint;
    auto light_dir = light->position - shadow_ray_org;
    auto distance_to_light = sqrt(norm_squared(light_dir));
    light_dir = normalize(light_dir);

    auto light_dot_norm = dot(light_dir, prd.normalAtHp);
    if (light_dot_norm < 0.f) return 0.f;

    owl::vec3f light_visibility = 0.f;
    uint32_t u0, u1;
    owl::packPointer(&light_visibility, u0, u1);
    optixTrace(
        self.world,
        shadow_ray_org,
        light_dir,
        EPS,
        distance_to_light * (1.f - EPS),
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

    owl::vec3f diffuse_brdf = prd.hpMaterial->albedo * PI_INV;

    return light_visibility
      * light_dot_norm
      * (1.f / (distance_to_light * distance_to_light))
      * diffuse_brdf * 2.f
    ;
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
owl::vec3f reflect_or_refract_ray(const Material& material,
                                  const owl::vec3f& ray_dir,
                                  const owl::vec3f& normal,
                                  Random& rand)
{
    if (material.matType == CONDUCTOR) {
        return reflect(ray_dir, normal);
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

    // Multiply by 4: combines two missing factors —
    //   ×4π for total emitted power of an isotropic point light (the photon-mapper
    //         saves photon.color = intensity/N, but each photon should carry
    //         total_power/N = 4π·intensity/N).
    //   ÷π   for the Lambertian BRDF (albedo/π, not albedo).
    return into_vec3f(photon.colour) * prd.hpMaterial->albedo * (4.f * cosTheta * cone_weight * inv_normalization);
}

constexpr __device__
float hable(const float x) {
    constexpr float A = 0.15f, B = 0.50f, C = 0.10f;
    constexpr float D = 0.20f, E = 0.02f, F = 0.30f;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

inline __device__
owl::vec3f filter_colour(owl::vec3f colour) {
    constexpr float exposure = 0.5f;
    constexpr float W = 11.2f;
    constexpr float inv_white = 1.0f / hable(W);

    colour *= exposure;
    colour = owl::vec3f(hable(colour.x), hable(colour.y), hable(colour.z)) * inv_white;
    colour = owl::vec3f(sqrtf(colour.x), sqrtf(colour.y), sqrtf(colour.z));
    return colour;
}