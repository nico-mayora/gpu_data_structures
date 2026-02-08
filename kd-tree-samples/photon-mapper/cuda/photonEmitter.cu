#include "./photonEmitter.cuh"
#include "./helpers.cu"
#include "owl/common/math/vec.h"
#define PHOTON_ATTENUATION_FACTOR 150
#define ATTENUATE_PHOTONS false

#include <optix_device.h>

using namespace owl;

inline __device__ void savePhoton(const PhotonMapperRGD &self, PhotonMapperPRD &prd) {
  int photonIndex = atomicAdd(self.photonsCount, 1);

  auto photon = &self.photons[photonIndex];
  photon->color = prd.color;
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
  // Normal mode: only save photons from purely diffuse paths.

  for (int i = 0; i < self.maxDepth; i++) {
    owl::traceRay(self.world, ray, prd);

    if (prd.event == SCATTER_SPECULAR || prd.event == SCATTER_REFRACT) {
      break;
    }

    if (prd.event == MISS) {
      break;
    }

    if (prd.event == SCATTER_DIFFUSE) {
      if (i > 0) savePhoton(self, prd);
      updateScatteredRay(ray, prd);
      continue;
    }

    if (prd.event == ABSORBED) {
      if (i > 0) savePhoton(self, prd);
      break;
    }
  }
}

inline __device__ void shootCausticsPhoton(const PhotonMapperRGD &self, Ray &ray, PhotonMapperPRD &prd) {
  // Caustic mode: Only save diffuse bounces that occur AFTER at least one caustic bounce.

  bool hadCausticBounce = false;

  for (int i = 0; i < self.maxDepth; i++) {
    owl::traceRay(self.world, ray, prd);

    if (prd.event == MISS) {
      break;
    }

    if (prd.event == SCATTER_SPECULAR || prd.event == SCATTER_REFRACT) {
      hadCausticBounce = true;
      updateScatteredRay(ray, prd);
      continue;
    }

    if (prd.event == SCATTER_DIFFUSE) {
      if (i > 0 && hadCausticBounce) {
        savePhoton(self, prd);
      }
      updateScatteredRay(ray, prd);
      continue;
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

inline __device__ void scatterDiffuse(PhotonMapperPRD &prd, const TrianglesGeomData &self) {
  const vec3f rayDir = optixGetWorldRayDirection();
  const vec3f rayOrg = optixGetWorldRayOrigin();
  const vec3f hitPoint = rayOrg + optixGetRayTmax() * rayDir;

  const vec3f normal = getPrimitiveNormal(self);

  prd.event = SCATTER_DIFFUSE;
  prd.scattered.origin = hitPoint;
  prd.scattered.direction = reflectDiffuse(normal, prd.random);
  prd.scattered.color = calculatePhotonColor(prd.color, self.material->albedo, prd.debug);
}

inline __device__ void scatterSpecular(PhotonMapperPRD &prd, const TrianglesGeomData &self) {
  const vec3f rayDir = optixGetWorldRayDirection();
  const vec3f rayOrg = optixGetWorldRayOrigin();
  const vec3f hitPoint = rayOrg + optixGetRayTmax() * rayDir;

  const vec3f normal = getPrimitiveNormal(self);

  prd.event = SCATTER_SPECULAR;
  prd.scattered.origin = hitPoint;
  prd.scattered.direction = reflect(rayDir, normal);
  prd.scattered.color = multiplyColor(self.material->albedo, prd.color);
}

inline __device__ void scatterRefract(PhotonMapperPRD &prd, const TrianglesGeomData &self) {
  const vec3f rayDir = optixGetWorldRayDirection();
  const vec3f rayOrg = optixGetWorldRayOrigin();
  const vec3f hitPoint = rayOrg + optixGetRayTmax() * rayDir;

  const vec3f normal = getPrimitiveNormal(self);

  prd.event = SCATTER_REFRACT;
  prd.scattered.origin = hitPoint;
  prd.scattered.direction = refract(rayDir, normal, self.material->ior);
  prd.scattered.color = multiplyColor(self.material->albedo, prd.color);
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

  const auto p_index = pIndex(prd.color, self.material->albedo, prd.debug);
  float randomProb = prd.random();

  if (randomProb < p_index) {
    switch (self.material->matType) {
      case LAMBERTIAN:
        scatterDiffuse(prd, self);
        break;
      case CONDUCTOR:
        scatterSpecular(prd, self);
        break;
      case DIELECTRIC: {
        auto event = reflect_or_refract_ray(self.material->ior,  optixGetWorldRayDirection(), getPrimitiveNormal(self), prd.random);
        if (event == SCATTER_SPECULAR)
          scatterSpecular(prd, self);
        else
          scatterRefract(prd, self);
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