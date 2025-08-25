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

    // TODO
    // auto specular_brdf = specularBrdf(prd.hit_record.material.specular,
    // light_dir,
    // ray.direction,
    // prd.hit_record.normal_at_hitpoint);

    return light_visibility
      * light_dot_norm
      * (1.f / (distance_to_light * distance_to_light))
      * (diffuse_brdf /* + specular_brdf */) *2.f
    ;
}