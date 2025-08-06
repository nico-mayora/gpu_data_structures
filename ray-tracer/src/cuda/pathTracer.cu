#include "pathTracer.h"
#include <optix_device.h>

#define EPS 1e-3f
#define INFTY 1e10f

inline __device__
owl::vec3f trace_shadow(const RayGenData &self, PerRayData &prd) {
    if (!prd.hit) return 1.f;

    owl::vec3f shadow_fact = 0.f;
    for (int sample = 0; sample < self.light_samples; sample++) {
        const owl::vec3f light_pos = self.lightSource.centre
            + (prd.random() * 2 - 1) * self.lightSource.sides.u
            + (prd.random() * 2 - 1) * self.lightSource.sides.v;

        owl::vec3f light_dir = light_pos - prd.hitPoint;
        float light_dist = length(light_dir);
        light_dir = normalize(light_dir);

        const float NdotL = dot(light_dir, prd.normalAtHp);
        if (NdotL >= 0.f) {
            owl::vec3f lightVisibility = 0.f;
            uint32_t u0, u1;
            owl::packPointer(&lightVisibility, u0, u1);
            optixTrace(self.world,
                       prd.hitPoint + EPS * prd.normalAtHp,
                       light_dir,
                       EPS,
                       light_dist * (1.f-EPS),
                       0.0f, // rayTime
                       OptixVisibilityMask( 255 ),
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT
                       | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                       | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                       SHADOW,          // SBT offset
                       RAY_TYPES_COUNT, // SBT stride
                       SHADOW,          // missSBTIndex
                       u0, u1 );

            shadow_fact
              += lightVisibility
              * self.lightSource.radiance
              * (NdotL / (light_dist * light_dist));
        }
    }
    return shadow_fact / static_cast<float>(self.light_samples);
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
        uint32_t u0, u1;
        owl::packPointer(&prd, u0, u1);
        optixTrace(self.world,
                   ray.origin,
                   ray.direction,
                   EPS,
                   INFTY,
                   0.0f, // rayTime
                   OptixVisibilityMask( 255 ),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   PRIMARY,          // SBT offset
                   RAY_TYPES_COUNT, // SBT stride
                   PRIMARY,          // missSBTIndex
                   u0, u1 );

        const owl::vec3f shadow_factor = trace_shadow(self, prd);

        colour += prd.colour * shadow_factor;
    }

    colour = colour * (1.f / self.pixel_samples);

    const int fbOfs = pixelID.x+self.resolution.x*pixelID.y;
    self.fbPtr[fbOfs] = owl::make_rgba(colour);
}


OPTIX_MISS_PROGRAM(miss)()
{
    const owl::vec2i pixelID = owl::getLaunchIndex();
    const MissProgData &self = owl::getProgramData<MissProgData>();

    auto &prd = owl::getPRD<PerRayData>();

    owl::vec3f rayDir = optixGetWorldRayDirection();
    rayDir = normalize(rayDir);
    prd.hit = false;
    prd.colour = self.sky_colour * (rayDir.y * .5f + 1.f);
}

OPTIX_MISS_PROGRAM(shadow)()
{
    owl::vec3f &light_visibility = owl::getPRD<owl::vec3f>();
    light_visibility = owl::vec3f(1.f);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    auto &prd = owl::getPRD<PerRayData>();

    const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
    const auto colour = self.material->albedo;
    const int   primID = optixGetPrimitiveIndex();
    const owl::vec3f Ng = normalize(self.normal[primID]);

    const owl::vec3f rayDir = optixGetWorldRayDirection();
    const owl::vec3f tMax = optixGetRayTmax();
    const owl::vec3f rayOrg = optixGetWorldRayOrigin();

    prd.hit = true;
    prd.hitPoint = rayOrg + tMax * rayDir;
    prd.normalAtHp = (dot(Ng, rayDir) > 0.f) ? -Ng : Ng;
    prd.colour = colour;
}

OPTIX_CLOSEST_HIT_PROGRAM(shadow)() { /* unused */}