#include "pathTracer.cuh"
#include "helpers.cu"
#include "../../common/kdtree/queries.cuh"
#include <optix_device.h>

#define K_GLOBAL_PHOTONS 1
#define K_CAUSTIC_PHOTONS 1
#define SECONDARY_RAYS 1
#define PI float(3.141592653)

inline __device__
owl::vec3f trace_path(const RayGenData &self, owl::Ray &ray, PerRayData &prd, int threadID) {
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

        if (prd.event == MISS || prd.event == ABSORBED)
            return colour_acum;

        if (prd.event == SCATTER_SPECULAR) {
            owl::vec3f new_ray_dir = reflect_or_refract_ray(
                prd.hpMaterial, ray.direction, prd.normalAtHp, prd.random
            );

            ray = owl::Ray(prd.hitPoint, new_ray_dir, EPS, INFTY);
            continue;
        }

        auto direct_illumination_fact = calculateDirectIllumination(self, prd);
        colour_acum += direct_illumination_fact;

        owl::vec3f diffuse_contrib = 0.f;
        // "Reach out" into the scene and perform gathers, this gives us global lighting with less local variance.
#pragma unroll
        for (uint32_t j = 0; j < SECONDARY_RAYS; ++j) {
            owl::vec3f rand_offset = owl::vec3f { prd.random(), prd.random(), prd.random() } * 2.f - 1.f;
            owl::vec3f diffuse_vector_dir = normalize(prd.normalAtHp + rand_offset);

            if (norm_squared(diffuse_vector_dir) < 100*EPS) {
                diffuse_vector_dir = prd.normalAtHp;
            }

            uint32_t q0, q1;
            PerRayData sprd;
            owl::packPointer(&sprd, q0, q1);
            optixTrace(
                self.world,
                prd.hitPoint,
                diffuse_vector_dir,
                EPS,
                INFTY,
                0.f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                PRIMARY,
                RAY_TYPES_COUNT,
                PRIMARY,
                q0, q1
            );

            const float query_pos[] = {sprd.hitPoint.x, sprd.hitPoint.y, sprd.hitPoint.z};

            size_t *point_indices = self.heap_indices + threadID * K_GLOBAL_PHOTONS;
            float *point_distances = self.heap_distances + threadID * K_GLOBAL_PHOTONS;
#pragma unroll
            for (int k = 0; k < K_GLOBAL_PHOTONS; k++) {
                point_indices[k] = 0;
                point_distances[k] = INFTY;
            }

            HeapQueryResult<K_GLOBAL_PHOTONS> photon_result {point_indices, point_distances};

            knn<K_GLOBAL_PHOTONS, Photon, HeapQueryResult<K_GLOBAL_PHOTONS>>(
                query_pos,
                self.photon_map,
                self.num_photons,
                &photon_result
            );

            owl::vec3f photon_illumination = 0.f;
            const float radius = owl::sqrt(point_distances[0]);
#pragma unroll
            for (int p = 0; p < K_GLOBAL_PHOTONS; p++) {
                if (point_distances[p] == INFTY) break;

                const Photon &photon = self.photon_map[point_indices[p]];
                photon_illumination += calculate_photon_contrib(photon, sprd, radius);
            }

            photon_illumination = photon_illumination / (PI * point_distances[0]);
            diffuse_contrib += photon_illumination;
        }

        // Perform caustic gather
        size_t *point_indices = self.heap_indices + threadID * K_CAUSTIC_PHOTONS;
        float *point_distances = self.heap_distances + threadID * K_CAUSTIC_PHOTONS;
#pragma unroll
        for (int k = 0; k < K_CAUSTIC_PHOTONS; k++) {
            point_indices[k] = 0;
            point_distances[k] = INFTY;
        }

        HeapQueryResult<K_CAUSTIC_PHOTONS> caustic_photon_result { point_indices, point_distances };

        const float query_point[] = { prd.hitPoint[0], prd.hitPoint[1], prd.hitPoint[2] };

        knn<K_CAUSTIC_PHOTONS, Photon, HeapQueryResult<K_CAUSTIC_PHOTONS>>(
            query_point,
            self.caustic_map,
            self.num_caustic,
            &caustic_photon_result
        );

        owl::vec3f caustic_term = 0.f;
        const float radius = owl::sqrt(point_distances[0]);
#pragma unroll
        for (int p = 0; p < K_CAUSTIC_PHOTONS; p++) {
            if (point_distances[p] == INFTY) break;

            const Photon &photon = self.caustic_map[point_indices[p]];
            caustic_term += calculate_photon_contrib(photon, prd, radius);
        }

        colour_acum += 0.0005f * diffuse_contrib * prd.hpMaterial.albedo + caustic_term;
        break;
    }

    return colour_acum;
}

OPTIX_RAYGEN_PROGRAM(ptRayGen)()  {
    const RayGenData &self = owl::getProgramData<RayGenData>();
    const owl::vec2i pixelID = owl::getLaunchIndex();

    const int threadID = pixelID.x + self.resolution.x * pixelID.y;

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

        colour += trace_path(self, ray, prd, threadID);
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
    prd.event = MISS;
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
            prd.event = SCATTER_DIFFUSE;
            break;
        }
        case DIELECTRIC: {
            prd.hpMaterial.matType = DIELECTRIC;
            prd.hpMaterial.ior = self.material->ior;

            prd.event = SCATTER_SPECULAR;
            break;
        }
        case CONDUCTOR: {
            prd.hpMaterial.matType = CONDUCTOR;
            prd.hpMaterial.specular = self.material->specular;

            prd.event = SCATTER_SPECULAR;
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
