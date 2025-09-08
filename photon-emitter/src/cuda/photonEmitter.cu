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
  photon->dir = prd.scattered.direction;
}

inline __device__ void updateScatteredRay(Ray &ray, PhotonMapperPRD &prd) {
  ray.origin = prd.scattered.origin;
  ray.direction = prd.scattered.direction;
  prd.color = prd.scattered.color;
}

inline __device__ void shootPhoton(const PhotonMapperRGD &self, Ray &ray, PhotonMapperPRD &prd) {
  for (int i = 0; i < self.maxDepth; i++) {
    owl::traceRay(self.world, ray, prd);

    if (prd.event == SCATTER_DIFFUSE) {
      if (i > 0) savePhoton(self, prd);
      updateScatteredRay(ray, prd);
    } else {
      break;
    }
  }
}

inline __device__ void shootCausticsPhoton(const PhotonMapperRGD &self, Ray &ray, PhotonMapperPRD &prd) {
  for (int i = 0; i < self.maxDepth; i++) {
    owl::traceRay(self.world, ray, prd);

    if (i > 0 && prd.event == SCATTER_DIFFUSE) {
      savePhoton(self, prd);
    }

    if (prd.event & (SCATTER_SPECULAR | SCATTER_REFRACT)) {
      updateScatteredRay(ray, prd);
    } else {
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

  Ray ray;
  ray.origin = self.position;
  ray.direction = randomPointInUnitSphere(prd.random);
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
  prd.scattered.color = multiplyColor(self.material->albedo, prd.color);
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
  prd.scattered.direction = refract(rayDir, normal, self.material->refraction_idx);
  prd.scattered.color = multiplyColor(self.material->albedo, prd.color);
}

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshClosestHit)(){
  auto &prd = owl::getPRD<PhotonMapperPRD>();
  const auto &self = owl::getProgramData<TrianglesGeomData>();

  const float diffuseProb = self.material->diffuse;
  const float specularProb = self.material->specular + diffuseProb;
  const float transmissionProb = self.material->transmission + specularProb;

  const float randomProb = prd.random();
  if (randomProb < diffuseProb) {
    scatterDiffuse(prd, self);
  } else if (randomProb < specularProb) {
    scatterSpecular(prd, self);
  } else if (randomProb < transmissionProb) {
    scatterRefract(prd, self);
  } else {
    prd.event = ABSORBED;
  }
}

OPTIX_MISS_PROGRAM(miss)(){
  auto &prd = owl::getPRD<PhotonMapperPRD>();
  prd.event = MISS;
}