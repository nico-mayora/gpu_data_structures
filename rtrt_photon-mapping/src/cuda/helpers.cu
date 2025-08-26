#define PI_INV (float)0.3183098861

#include "kdtree/constants.cuh"

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

inline __device__ owl::vec3f calculateDirectIllumination(const RayGenData &self, PerRayData &prd) {
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

inline __device__ owl::vec3f reflect(const owl::vec3f &incoming, const owl::vec3f &normal) {
    return incoming - 2.f * dot(incoming, normal) * normal;
}

inline __device__
owl::vec3f reflect_or_refract_ray(const Material& material,
                                  const owl::vec3f& ray_dir,
                                  const owl::vec3f& normal,
                                  Random rand,
                                  bool& absorbed,
                                  float& coef)
{
    absorbed = false;

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
}