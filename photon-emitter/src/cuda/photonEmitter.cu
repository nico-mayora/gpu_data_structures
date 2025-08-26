#include "photonEmitter.h"
#include "photonUtils.h"
#include <optix_device.h>

using namespace owl;
#define T_MIN 0.001f

inline __device__ void scatterDiffuse(PerRayData &prd, const TrianglesGeomData &self) {
    const vec3f rayDir = optixGetWorldRayDirection();
    const vec3f rayOrg = optixGetWorldRayOrigin();
    const vec3f hitPoint = rayOrg + optixGetRayTmax() * rayDir;
    const vec3f normal = getPrimitiveNormal(self);

    prd.event = SCATTER_DIFFUSE;
    prd.scattered.origin = hitPoint + EPS * normal;
    prd.scattered.direction = reflectDiffuse(normal, prd.random);
    prd.scattered.color = multiplyColor(self.material->albedo, prd.colour);
}

inline __device__ void scatterSpecular(PerRayData &prd, const TrianglesGeomData &self) {
    const vec3f rayDir = optixGetWorldRayDirection();
    const vec3f rayOrg = optixGetWorldRayOrigin();
    const vec3f hitPoint = rayOrg + optixGetRayTmax() * rayDir;
    const vec3f normal = getPrimitiveNormal(self);

    prd.event = SCATTER_SPECULAR;
    prd.scattered.origin = hitPoint + EPS * normal;
    prd.scattered.direction = reflect(rayDir, normal);
    prd.scattered.color = multiplyColor(self.material->albedo, prd.colour);
}

inline __device__ void scatterRefract(PerRayData &prd, const TrianglesGeomData &self) {
    const vec3f rayDir = optixGetWorldRayDirection();
    const vec3f rayOrg = optixGetWorldRayOrigin();
    const vec3f hitPoint = rayOrg + optixGetRayTmax() * rayDir;
    const vec3f normal = getPrimitiveNormal(self);

    prd.event = SCATTER_REFRACT;
    prd.scattered.origin = hitPoint + EPS * normal;
    prd.scattered.direction = refract(rayDir, normal, self.material->refraction_idx);
    prd.scattered.color = multiplyColor(self.material->albedo, prd.colour);
}

inline __device__ void updateScatteredRay(Ray &ray, PerRayData &prd) {
  ray.origin = prd.scattered.origin;
  ray.direction = prd.scattered.direction;
  prd.colour = prd.scattered.color;
}

inline __device__ owl::vec3f getInitialLightDirection(Light &light, owl::vec2i launchIndex, PerRayData &prd) {
    switch (light.type) {
        case 0:
            return randomPointInUnitSphere(prd.random);
    }
}

OPTIX_RAYGEN_PROGRAM(photonEmitter)() {
    const Light &self = owl::getProgramData<Light>();
    const owl::vec2i launchIndex = owl::getLaunchIndex();

    const int threadId = launchIndex.y * owl::getLaunchDims().x + launchIndex.x;
    const int basePhotonIndex = threadId * self.maxDepth;

    PerRayData prd;
    prd.random.init(launchIndex.x, launchIndex.y);

    Ray ray;
    ray.origin = self.position; // MOVE THIS TO FUNCTION BASED ON TYPE
    ray.direction = getInitialLightDirection(const_cast<Light&>(self), launchIndex, prd);
    ray.tmin = T_MIN;
    prd.colour = self.colour * self.power;

    bool photonDead = false;

    if (threadId == 1) {
      ray.direction = owl::vec3f(0.0f, -1.0f, 0.0f);
      printf("LIGHT DATA: Type: %d, Position: %f, %f, %f, Colour: %f, %f, %f, Power: %f, MaxDepth: %d\n", self.type, self.position.x, self.position.y, self.position.z, self.colour.x, self.colour.y, self.colour.z, self.power, self.maxDepth);

      printf("Photon Emitter Ray Origin: %f, %f, %f\n", ray.origin.x, ray.origin.y, ray.origin.z);
      printf("Photon Emitter Ray Direction: %f, %f, %f\n", ray.direction.x, ray.direction.y, ray.direction.z);
    }
    
    for (int bounce = 0; bounce < self.maxDepth; bounce++) {
        if (photonDead) {
            storePhoton(self, basePhotonIndex + bounce, owl::vec3f(0.0f), owl::vec3f(0.0f), owl::vec3f(0.0f), true);
            continue;
        }

        traceRay(self.world, ray, prd);
        
        if (prd.event == MISS || prd.event == ABSORBED) {
            photonDead = true;
            storePhoton(self, basePhotonIndex + bounce, owl::vec3f(0.0f), owl::vec3f(0.0f), owl::vec3f(0.0f), true);
            continue;
        }
        
        // FIRST BOUNCE GUARDA PHOTON? TODO
        if (prd.event == SCATTER_DIFFUSE) {
            storePhoton(self, basePhotonIndex + bounce, prd.scattered.origin, prd.scattered.direction, prd.scattered.color, false);
        } else {
            storePhoton(self, basePhotonIndex + bounce, owl::vec3f(0.0f), owl::vec3f(0.0f), owl::vec3f(0.0f), true);
        }
        
        updateScatteredRay(ray, prd);
    }
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)() {
    auto &prd = owl::getPRD<PerRayData>();
    const auto &self = owl::getProgramData<TrianglesGeomData>();

    const float diffuseProb = self.material->diffuse;
    const float specularProb = self.material->specular + diffuseProb;
    const float transmissionProb = self.material->transmission + specularProb;

    const float randomProb = prd.random();
//    if (randomProb < diffuseProb) {
      scatterDiffuse(prd, self);
//    } else if (randomProb < specularProb) {
//      scatterSpecular(prd, self);
//    } else if (randomProb < transmissionProb) {
//      scatterRefract(prd, self);
//    } else {
//      prd.event = ABSORBED;
//    }
}

OPTIX_MISS_PROGRAM(miss)() {
    auto &prd = owl::getPRD<PerRayData>();
    prd.event = MISS;
}
