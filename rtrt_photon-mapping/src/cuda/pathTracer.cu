#include "pathTracer.h"
#include "helpers.cu"
#include <optix_device.h>


inline __device__
owl::vec3f trace_path(const RayGenData &self, owl::Ray &ray, PerRayData &prd) {
    owl::vec3f colour_acum = 1.f;

    for (int depth = 0; depth < self.depth; depth++) {
        traceRay(self.world, ray, prd);

        if (prd.event == MISSED || prd.event == CANCELLED)
            return colour_acum * prd.colour.emitted;

        // if (prd.event == REFLECTED_DIFFUSE):
        colour_acum *= prd.colour.reflected;
        owl::vec3f org = prd.hitPoint;
        owl::vec3f dir = prd.normalAtHp + random_in_unit_sphere(prd.random);
        dir = (length(dir) < EPS) ? prd.normalAtHp : normalize(dir);
        ray = owl::Ray(org, dir, EPS, INFTY);
    }
    return colour_acum;
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

        colour += trace_path(self, ray, prd);
    }

    colour *= 1.f / self.pixel_samples;

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
    prd.event = MISSED;
    prd.colour.emitted = self.sky_colour * (rayDir.y * .5f + 1.f);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
    auto &prd = owl::getPRD<PerRayData>();
    const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();

    const int primID = optixGetPrimitiveIndex();
    const owl::vec3f Ng = normalize(self.normal[primID]);
    const owl::vec3f rayDir = optixGetWorldRayDirection();
    const owl::vec3f tMax = optixGetRayTmax();
    const owl::vec3f rayOrg = optixGetWorldRayOrigin();

    if (self.material->matType == EMISSIVE) {
        prd.colour.reflected = 0.f;
        const float dir_dot_ng = dot(Ng, rayDir);
        prd.colour.emitted = dir_dot_ng < 0.f ? self.material->albedo : 0.f;
        prd.event = CANCELLED;
    }
    if (self.material->matType == LAMBERTIAN) {
        prd.colour.reflected = self.material->albedo;
        prd.colour.emitted = 0.f;
        prd.event = REFLECTED_DIFFUSE;
    }

    prd.hitPoint = rayOrg + tMax * rayDir;
    prd.normalAtHp = (dot(Ng, rayDir) > 0.f) ? -Ng : Ng;
}