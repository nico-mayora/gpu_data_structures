#include "pathTracer.cuh"
#include "helpers.cu"
#include "kdtree/queries.cuh"
#include <optix_device.h>

#define K_PHOTONS 1
#define PI float(3.141592653)

inline __device__
owl::vec3f trace_path(const RayGenData &self, owl::Ray &ray, PerRayData &prd) {
    owl::vec3f colour_acum = 0.f;

    for (int32_t i = 0; i < 1; ++i) {
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

        if (prd.event == REFLECTED_SPECULAR) {
            owl::vec3f new_ray_dir = reflect_or_refract_ray(
                prd.hpMaterial, ray.direction, prd.normalAtHp, prd.random
            );

            ray = owl::Ray(prd.hitPoint, new_ray_dir, EPS, INFTY);
            continue;
        }

        auto direct_illumination_fact = calculateDirectIllumination(self, prd);
        colour_acum += direct_illumination_fact;

        const float query_pos[] = {prd.hitPoint.x, prd.hitPoint.y, prd.hitPoint.z};
        size_t point_indices[K_PHOTONS] = {0};
        float point_distances[K_PHOTONS] = {0.f};
        HeapQueryResult<K_PHOTONS> photon_result {&point_indices[0], &point_distances[0]};

        knn<K_PHOTONS, Photon, HeapQueryResult<K_PHOTONS>>(
            &query_pos[0],
            self.photon_map,
            self.num_photons,
            &photon_result
        );

        owl::vec3f photon_illumination = 0.f;
#pragma unroll
        for (int p = 0; p < K_PHOTONS; p++) {
            if (point_distances[p] == 0.f) break;

            const Photon &photon = self.photon_map[point_indices[p]];
            const float distance2 = point_distances[p];
            const float weight = 1.f - owl::common::polymorphic::rsqrt(distance2) / owl::common::polymorphic::rsqrt(point_distances[0]);
            photon_illumination += owl::vec3f(photon.colour[0], photon.colour[1], photon.colour[2]) * weight;

            photon_illumination = owl::vec3f(photon.colour[0], photon.colour[1], photon.colour[2]);
        }

//        colour_acum += photon_illumination * (prd.hpMaterial.diffuse / PI);
        colour_acum = photon_illumination;
        // TODO Photon bounce When a ray hits a diffuse material, perform one bounce (random) and gather photons.
        break;
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

            prd.event = REFLECTED_SPECULAR;
            break;
        }
        case CONDUCTOR: {
            prd.hpMaterial.matType = CONDUCTOR;
            prd.hpMaterial.specular = self.material->specular;

            prd.event = REFLECTED_SPECULAR;
            break;
        }
        default:
            printf("[WARNING] - Material type not implemented. Expect visual glitches!");
    }


    prd.hitPoint = rayOrg + tMax * rayDir;
    prd.normalAtHp = /*(dot(Ng, rayDir) > 0.f) ? -Ng :*/ Ng;
}

OPTIX_MISS_PROGRAM(shadow)()
{
    // we didn't hit anything, so the light is visible
    owl::vec3f &lightVisbility = owl::getPRD<owl::vec3f>();
    lightVisbility = owl::vec3f(1.f);
}

OPTIX_CLOSEST_HIT_PROGRAM(shadow)() { /* unused */ }
OPTIX_ANY_HIT_PROGRAM(shadow)() { /* unused */ }