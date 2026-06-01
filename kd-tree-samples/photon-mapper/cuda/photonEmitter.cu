#include "./photonEmitter.cuh"
#include "./helpers.cu"
#include "owl/common/math/vec.h"
#include "../../common/helpers.cuh"

#define PHOTON_ATTENUATION_FACTOR 150
#define ATTENUATE_PHOTONS false

#include <optix_device.h>

using namespace owl;

inline __device__ void savePhoton(const PhotonMapperRGD &self, PhotonMapperPRD &prd) {
  int photonIndex = atomicAdd(self.photonsCount, 1);

  auto photon = &self.photons[photonIndex];
  // Per-photon flux dPhi = Phi / N = 4*pi*I / N. prd.color tracks I * throughput
  // (radiant intensity, W/sr); the 4*pi is the solid angle photons are emitted into,
  // making the stored value true power so the path tracer's density estimate matches
  // the direct term's units.
  photon->color = prd.color * (4.f * PI) / static_cast<float>(self.totalPhotons);
  photon->pos = prd.scattered.origin;
  photon->dir = prd.direction;
}

inline __device__ void updateScatteredRay(Ray &ray, PhotonMapperPRD &prd) {
  ray.origin = prd.scattered.origin;
  ray.direction = prd.scattered.direction;
  prd.direction = prd.scattered.direction;
  prd.color = prd.scattered.color;
}

inline __device__ void shootPhoton(const PhotonMapperRGD &self, Ray &ray, PhotonMapperPRD &prd) {
  // Design B: store EVERY diffuse interaction, including the first (directly-lit)
  // hit. The global map then represents the full incident radiance (direct +
  // indirect) at each surface, so the path tracer's final gather reads complete
  // radiance at gather points and must NOT add analytic direct there. The eye's
  // primary hit still computes direct analytically (the map is only read at the
  // gather points, never at the primary, so there is no double-counting).
  for (int i = 0; i < self.maxDepth; i++) {
    owl::traceRay(self.world, ray, prd);

    if (prd.event == MISS) {
      break;
    }

    if (prd.event == SCATTER_SPECULAR || prd.event == SCATTER_REFRACT) {
      updateScatteredRay(ray, prd);
      continue;
    }

    if (prd.event == SCATTER_DIFFUSE) {
      savePhoton(self, prd);
      updateScatteredRay(ray, prd);
      continue;
    }

    if (prd.event == ABSORBED) {
      savePhoton(self, prd);
      break;
    }
  }
}

inline __device__ void shootCausticsPhoton(const PhotonMapperRGD &self, Ray &ray, PhotonMapperPRD &prd) {
  // Caustic mode: Only save diffuse bounces that occur AFTER at least one caustic bounce.
  for (int i = 0; i < self.maxDepth; i++) {
    owl::traceRay(self.world, ray, prd);

    if (prd.event == MISS) {
      break;
    }

    if (prd.event == SCATTER_SPECULAR || prd.event == SCATTER_REFRACT) {
      updateScatteredRay(ray, prd);
      continue;
    }

    if (prd.event == SCATTER_DIFFUSE) {
      if (i > 0) {
        savePhoton(self, prd);
      }
      break;
    }

    if (prd.event == ABSORBED) {
      break;
    }
  }
}

OPTIX_RAYGEN_PROGRAM(pointLightRayGen)(){
  const auto &self = owl::getProgramData<PointLightRGD>();
  const vec2i id = owl::getLaunchIndex();

  PhotonMapperPRD prd;
  prd.random.init(id.x, id.y);
  prd.color = self.color;

  auto direction = randomPointInUnitSphere(prd.random);

  prd.direction = direction;

  Ray ray;
  ray.origin = self.position;
  ray.direction = direction;
  ray.tmin = EPS;

  if (self.causticsMode) {
    shootCausticsPhoton(self, ray, prd);
  } else {
    shootPhoton(self, ray, prd);
  }
}

inline __device__ void scatterDiffuse(PhotonMapperPRD &prd, const TrianglesGeomData &self, const vec3f &albedo) {
  const vec3f rayDir = optixGetWorldRayDirection();
  const vec3f rayOrg = optixGetWorldRayOrigin();
  const vec3f hitPoint = rayOrg + optixGetRayTmax() * rayDir;
  const auto [u, v] = optixGetTriangleBarycentrics();
  const int primID = optixGetPrimitiveIndex();

  const vec3f normal = get_normal_at_hp(self, u, v, primID);

  prd.event = SCATTER_DIFFUSE;
  prd.scattered.origin = hitPoint;
  prd.scattered.direction = reflectDiffuse(normal, prd.random);
  prd.scattered.color = calculatePhotonColor(prd.color, albedo, prd.debug);
}

inline __device__ void scatterSpecular(PhotonMapperPRD &prd, const TrianglesGeomData &self, const vec3f &albedo) {
  const vec3f rayDir = optixGetWorldRayDirection();
  const vec3f rayOrg = optixGetWorldRayOrigin();
  const vec3f hitPoint = rayOrg + optixGetRayTmax() * rayDir;
  const auto [u, v] = optixGetTriangleBarycentrics();
  const int primID = optixGetPrimitiveIndex();

  const vec3f normal = get_normal_at_hp(self, u, v, primID);

  prd.event = SCATTER_SPECULAR;
  prd.scattered.origin = hitPoint;
  prd.scattered.direction = reflect(rayDir, normal);
  prd.scattered.color = multiplyColor(albedo, prd.color);
}

inline __device__ void scatterRefract(PhotonMapperPRD &prd, const TrianglesGeomData &self, const vec3f &albedo) {
  const vec3f rayDir = optixGetWorldRayDirection();
  const vec3f rayOrg = optixGetWorldRayOrigin();
  const vec3f hitPoint = rayOrg + optixGetRayTmax() * rayDir;
  const auto [u, v] = optixGetTriangleBarycentrics();
  const int primID = optixGetPrimitiveIndex();

  const vec3f normal = get_normal_at_hp(self, u, v, primID);

  prd.event = SCATTER_REFRACT;
  prd.scattered.origin = hitPoint;
  prd.scattered.direction = refract(rayDir, normal, self.material->ior);
  prd.scattered.color = multiplyColor(albedo, prd.color);
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
RayEvent reflect_or_refract_ray(float index_of_refraction,
                                  const owl::vec3f& ray_dir,
                                  const owl::vec3f& normal,
                                  Random& rand)
{
  float cos_theta = dot(-ray_dir, normal);
  float fresnel = calculate_fresnel(index_of_refraction, cos_theta);

  if (rand() < fresnel) { // Reflect
    return SCATTER_SPECULAR;
  }

  return SCATTER_REFRACT;
}

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshClosestHit)(){
  auto &prd = owl::getPRD<PhotonMapperPRD>();
  const auto &self = owl::getProgramData<TrianglesGeomData>();

  // Sample albedo once (texture or flat) so photon colours track textured surfaces.
  const auto [u0, v0] = optixGetTriangleBarycentrics();
  const int primID0 = optixGetPrimitiveIndex();
  const vec3f surfaceAlbedo = get_albedo_at_hp(self, u0, v0, primID0);

  const auto p_index = pIndex(prd.color, surfaceAlbedo, prd.debug);
  float randomProb = prd.random();

  if (randomProb < p_index) {
    switch (self.material->matType) {
      case LAMBERTIAN:
        scatterDiffuse(prd, self, surfaceAlbedo);
        break;
      case CONDUCTOR:
        scatterSpecular(prd, self, surfaceAlbedo);
        break;
      case DIELECTRIC: {
        const vec3f Ng = get_normal_at_hp(self, u0, v0, primID0);
        auto event = reflect_or_refract_ray(self.material->ior,  optixGetWorldRayDirection(), Ng, prd.random);
        if (event == SCATTER_SPECULAR)
          scatterSpecular(prd, self, surfaceAlbedo);
        else
          scatterRefract(prd, self, surfaceAlbedo);
        break;
      }
    }
  } else {
    prd.event = ABSORBED;
    const vec3f rayDir = optixGetWorldRayDirection();
    const vec3f rayOrg = optixGetWorldRayOrigin();
    prd.scattered.origin = rayOrg + optixGetRayTmax() * rayDir;
  }
}

OPTIX_MISS_PROGRAM(miss)(){
  auto &prd = owl::getPRD<PhotonMapperPRD>();
  prd.event = MISS;
}