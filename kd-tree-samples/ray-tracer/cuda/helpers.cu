#define PI_INV (float)0.3183098861

#include "../../common/kdtree/constants.cuh"

inline __device__
float norm_squared(owl::vec3f v) {
    return dot(v, v);
}

inline __device__
owl::vec3f random_in_unit_sphere(Random rand) {
    owl::vec3f v;
    do {
        v = {rand()*2-1, rand()*2-1, rand()*2-1};
    } while (norm_squared(v) > 1.f);

    return v;
}

inline __device__
owl::vec3f calculateDirectIllumination(const RayGenData &self, PerRayData &prd) {
    auto light = self.scene_light;
    auto shadow_ray_org = prd.hitPoint;
    auto light_dir = light->position - shadow_ray_org;
    light_dir = normalize(light_dir);
    auto distance_to_light = owl::common::polymorphic::rsqrt(norm_squared(light_dir));

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

    owl::vec3f diffuse_brdf = prd.hpMaterial.albedo * PI_INV;

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
                               Random rand) {
    float cos_i = dot(-ray_dir, normal);
    float etai_over_etat;
    owl::vec3f outward_normal;

    // Determine if we're entering or exiting the material
    if (cos_i > 0.0f) {
        // Entering material (air -> glass)
        etai_over_etat = 1.0f / material.ior;
        outward_normal = normal;
    } else {
        // Exiting material (glass -> air)
        etai_over_etat = material.ior;
        outward_normal = -normal;
    }

    // Check for total internal reflection using discriminant
    float cos_theta = fminf(dot(-ray_dir, outward_normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    if (etai_over_etat * sin_theta > 1.0f) {
        // Total internal reflection
        return owl::vec3f(0.0f);
    }

    owl::vec3f r_out_perp = etai_over_etat * (ray_dir + cos_theta * outward_normal);
    owl::vec3f r_out_parallel = - owl::common::polymorphic::rsqrt(fabsf(1.0f - dot(r_out_perp, r_out_perp)))
                                * outward_normal;
    return r_out_perp + r_out_parallel;
}

inline __device__
owl::vec3f reflect_or_refract_ray(const Material& material,
                                  const owl::vec3f& ray_dir,
                                  const owl::vec3f& normal,
                                  Random& rand)
{
    float coef = 0.f;
    if (material.matType == CONDUCTOR) {
        coef = material.specular;
        return reflect(ray_dir, normal);
    }

    if (material.matType == DIELECTRIC) {
        float cos_theta = dot(-ray_dir, normal);
        float fresnel = calculate_fresnel(material.ior, cos_theta);

        if (rand() < fresnel) { // Reflect
            coef = fresnel;
            return reflect(ray_dir, normal);
        }

        // Refract
        coef = 1.0f - fresnel;
        owl::vec3f refracted = calculate_refracted(material, ray_dir, normal, rand);

        if (length(refracted) == 0.0f) { // Total Internal Reflection
            coef = 1.0f;
            return reflect(ray_dir, normal);
        }

        return refracted;
    }
    return 0.;
}

inline __device__
owl::vec3f into_vec3f(const float *arr) {
    return owl::vec3f(arr[0], arr[1], arr[2]);
}

inline __device__
owl::vec3f calculate_photon_contrib(const Photon& photon, const PerRayData& prd, const float radius) {
    const owl::vec3f diff = into_vec3f(photon.coords) - prd.hitPoint;
    const float distance = length(diff);

    // lambert
    const owl::vec3f wi = -into_vec3f(photon.dir);
    const float cosTheta = max(0.f, dot(prd.normalAtHp, wi));
    if (cosTheta <= 0.0f) {
        return 0.f;
    }

    // cone filter
    float cone_weight = max(0.f, 1.f - distance / radius);
    cone_weight *= 3.f;

    const owl::vec3f brdf = prd.hpMaterial.albedo / static_cast<float>(M_PI);
    return into_vec3f(photon.colour) * brdf * cosTheta * cone_weight;
}