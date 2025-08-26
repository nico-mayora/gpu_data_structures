#include "pathTracer.cuh"
#include "helpers.cu"
#include "kdtree/queries.cuh"
#include <optix_device.h>

#define K_PHOTONS 100

inline __device__
owl::vec3f trace_path(const RayGenData &self, owl::Ray &ray, PerRayData &prd) {
    owl::vec3f colour_acum = 0.f;

    for (int32_t i = 0; i < self.depth; ++i) {
        uint32_t p0, p1;
        owl::packPointer(&prd, p0, p1);
        optixTrace(
            self.world,
            ray.origin,
            ray.direction,
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

        if (prd.event == MISSED || prd.event == CANCELLED)
            return colour_acum;

##### Updated upstream
    // if (prd.event == REFLECTED_SPECULAR) {
    //
    //     continue;
    // }
%%%%%%%%%%%%%
        if (prd.event == REFLECTED_SPECULAR) {
            
            continue;
        }

        auto direct_illumination_fact = calculateDirectIllumination(self, prd);
        colour_acum += direct_illumination_fact;

        // TODO Add photon gather. That handles caustics and diffuse reflections.
    }
######## Stashed changes

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
    const MissProgData &self = owl::getProgramData<MissProgData>();

    auto &prd = owl::getPRD<PerRayData>();

    owl::vec3f rayDir = optixGetWorldRayDirection();
    rayDir = normalize(rayDir);
    prd.event = MISSED;
    prd.colour = self.sky_colour * (rayDir.y * .5f + 1.f);
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

    switch (self.material->matType) {
        case LAMBERTIAN: {
            prd.hpMaterial.matType = LAMBERTIAN;
            prd.hpMaterial.albedo = self.material->albedo;
            prd.hpMaterial.diffuse = self.material->diffuse;

            prd.colour = self.material->albedo;
            prd.event = REFLECTED_DIFFUSE;
            break;
        }
        case DIELECTRIC: {
            prd.hpMaterial.matType = DIELECTRIC;
            prd.hpMaterial.ior = self.material->ior;

            prd.colour = self.material->albedo;
            prd.event = REFLECTED_DIFFUSE;
            break;
        }
        default:
            printf("[WARNING] - Material type not implemented. Expect visual glitches!");
    }

    prd.hitPoint = rayOrg + tMax * rayDir;
    prd.normalAtHp = (dot(Ng, rayDir) > 0.f) ? -Ng : Ng;
}

OPTIX_MISS_PROGRAM(shadow)()
{
    // we didn't hit anything, so the light is visible
    owl::vec3f &lightVisbility = owl::getPRD<owl::vec3f>();
    lightVisbility = owl::vec3f(1.f);
}

OPTIX_CLOSEST_HIT_PROGRAM(shadow)() { /* unused */ }
OPTIX_ANY_HIT_PROGRAM(shadow)() { /* unused */ }