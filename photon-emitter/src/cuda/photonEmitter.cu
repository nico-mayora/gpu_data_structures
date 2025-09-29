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

  if (prd.debug) {
//    printf("Saved photon at index %d: pos(%f, %f, %f), dir(%f, %f, %f), color(%f, %f, %f)\n",
//           photonIndex,
//           photon->pos.x, photon->pos.y, photon->pos.z,
//           photon->dir.x, photon->dir.y, photon->dir.z,
//           photon->color.x, photon->color.y, photon->color.z);
  }

  auto id = owl::getLaunchIndex();
  if (photon->color.x > 0.999f && photon->pos.x < -0.999f) {
//    printf("Photon at index %d, %d: pos(%f, %f, %f), dir(%f, %f, %f), color(%f, %f, %f)\n",
//           id.x,
//           id.y,
//           photon->pos.x, photon->pos.y, photon->pos.z,
//           photon->dir.x, photon->dir.y, photon->dir.z,
//           photon->color.x, photon->color.y, photon->color.z);
  }

}

inline __device__ void updateScatteredRay(Ray &ray, PhotonMapperPRD &prd) {
  ray.origin = prd.scattered.origin;
  ray.direction = prd.scattered.direction;
  prd.direction = prd.scattered.direction;
  prd.color = prd.scattered.color;
}

inline __device__ void shootPhoton(const PhotonMapperRGD &self, Ray &ray, PhotonMapperPRD &prd) {
  auto id = owl::getLaunchIndex();

  for (int i = 0; i < self.maxDepth; i++) {
    if (prd.debug) {
      printf("Depth %d, Ray Origin: (%f, %f, %f), Direction: (%f, %f, %f), Color: (%f, %f, %f)\n",
             i,
             ray.origin.x, ray.origin.y, ray.origin.z,
             ray.direction.x, ray.direction.y, ray.direction.z,
             prd.color.x, prd.color.y, prd.color.z);
    }

    owl::traceRay(self.world, ray, prd);

//    vec3f targetColor = {0.14f, 0.45f, 0.091f};
//    vec3f targetDir = {0.939488f, -0.342164f, 0.016907f};
//    if (i == 0 && prd.event == SCATTER_DIFFUSE && prd.scattered.color == targetColor) {
    if (prd.debug && i > 0) {
//    if (ray.direction == targetDir) {
//      printf("HIT WITH TARGET COLOR AT DEPTH 0\n");
      printf("id: (%d, %d),  Ray Origin: (%f, %f, %f), Direction: (%f, %f, %f), Color: (%f, %f, %f),event: %d\n",
             id.x, id.y,
             ray.origin.x, ray.origin.y, ray.origin.z,
             ray.direction.x, ray.direction.y, ray.direction.z,
             prd.scattered.color.x, prd.scattered.color.y, prd.scattered.color.z,
             prd.event);
    }

    if (prd.event == SCATTER_DIFFUSE || prd.event == ABSORBED) {
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

  auto direction = randomPointInUnitSphere(prd.random);

  prd.direction = direction;

  Ray ray;
  ray.origin = self.position;
  ray.direction = direction;
  ray.tmin = EPS;

//  if (id == vec2i(0)) {
//    prd.debug = true;
//    ray.direction = {0.939488f, -0.342164f, 0.016907f};
  if (id.x == 882474 && id.y == 0) {
//    printf("Special ray at id (%d, %d), and random value: %f\n", id.x, id.y, prd.random());
    prd.debug = true;
  } else {
    prd.debug = false;
  }

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

//  vec3f targetDir = {0.939488f, -0.342164f, 0.016907f};
  if (prd.debug) {
    printf("Hit point: (%f, %f, %f), Normal: (%f, %f, %f), length: %f, dot(normal, normal): %f, dot(normal, (0,1,0)): %f\n",
            hitPoint.x, hitPoint.y, hitPoint.z,
            normal.x, normal.y, normal.z,
            length(normal),
            dot(normal, normal),
            dot(normal, {0.0f, 1.0f, 0.0f}));
//    printf("is wall: %d, hitpoint: (%f, %f, %f), normal: (%f, %f, %f), rayDir: (%f, %f, %f)\n",
//           hitPoint.x > 0.999f,
//           hitPoint.x, hitPoint.y, hitPoint.z,
//           normal.x, normal.y, normal.z,
//           rayDir.x, rayDir.y, rayDir.z);
//    printf("HIT WITH TARGET COLOR\n");
//    printf("material albedo: (%f, %f, %f)\n",
//           self.material->albedo.x,
//           self.material->albedo.y,
//           self.material->albedo.z);
  }

  prd.event = SCATTER_DIFFUSE;
  prd.scattered.origin = hitPoint;
  prd.scattered.direction = reflectDiffuse(normal, prd.random);
//  prd.scattered.color = self.material->albedo;
  prd.scattered.color = calculatePhotonColor(prd.color, self.material->albedo, prd.debug);
//  if (prd.debug) {
//    prd.scattered.direction = normal;
//  }
  if (hitPoint.x > 0.999f && prd.color.z > 0.999f) {
//    prd.debug = true;
//    auto dotVal = dot(prd.scattered.direction, normal);
//    printf("Hit with wall, scattered dir: (%f, %f, %f), color: (%f, %f, %f), dot: %f\n",
//           prd.scattered.direction.x,
//           prd.scattered.direction.y,
//           prd.scattered.direction.z,
//           prd.scattered.color.x,
//           prd.scattered.color.y,
//           prd.scattered.color.z,
//           dotVal);
//    printf("previous color: (%f, %f, %f), albedo: (%f, %f, %f), new color: (%f, %f, %f)\n",
//           prd.color.x,
//           prd.color.y,
//           prd.color.z,
//            self.material->albedo.x,
//            self.material->albedo.y,
//            self.material->albedo.z,
//            prd.scattered.color.x,
//            prd.scattered.color.y,
//            prd.scattered.color.z);
  }
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

  // Refract
  return SCATTER_REFRACT;

  // Yikes
//  if (length(refracted) == 0.0f) { // Total Internal Reflection
//    return reflect(ray_dir, normal);
//  }
//
//  return refracted;
}

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshClosestHit)(){
  auto &prd = owl::getPRD<PhotonMapperPRD>();
  const auto &self = owl::getProgramData<TrianglesGeomData>();

//  const float diffuseProb = self.material->diffuse;
//  const float specularProb = self.material->specular + diffuseProb;
//  const float transmissionProb = self.material->ior + specularProb;

//  vec3f targetDir = {0.939488f, -0.342164f, 0.016907f};
//  vec3f dir = optixGetWorldRayDirection();
  const auto p_index = pIndex(prd.color, self.material->albedo, prd.debug);
  float randomProb = prd.random();


  if (prd.debug) {
//    printf("p_index: %f, randomProb: %f\n", p_index, randomProb);
//    printf("HIT WITH TARGET COLOR\n");
//    printf("material albedo: (%f, %f, %f)\n",
//           self.material->albedo.x,
//           self.material->albedo.y,
//           self.material->albedo.z);
//    printf("photon color: (%f, %f, %f)\n",
//           prd.color.x,
//           prd.color.y,
//           prd.color.z);
//
//    printf("p_index: %f, randomProb: %f\n", p_index, randomProb);
//    randomProb = 0.0f; // force to scatter
  }

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

  // if (self.material->matType == LAMBERTIAN) {
  //   if (randomProb < diffuseProb)
  //     scatterDiffuse(prd, self);
  //   else
  //     prd.event = ABSORBED;
  // } else if (self.material->matType == CONDUCTOR) {
  //   if (randomProb < specularProb)
  //     scatterSpecular(prd, self);
  //   else
  //     prd.event = ABSORBED;
  // } else if (self.material->matType == DIELECTRIC) {
  //   auto event = reflect_or_refract_ray(self.material->ior,
  //                                       optixGetWorldRayDirection(),
  //                                       getPrimitiveNormal(self),
  //                                       prd.random);
  //   if (event == SCATTER_SPECULAR)
  //     scatterSpecular(prd, self);
  //   else
  //     scatterRefract(prd, self);
  // }
}

OPTIX_MISS_PROGRAM(miss)(){
  auto &prd = owl::getPRD<PhotonMapperPRD>();
  prd.event = MISS;
}