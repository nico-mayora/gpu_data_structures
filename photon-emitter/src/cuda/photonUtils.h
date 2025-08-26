#pragma once
#include "photonEmitter.h"
#include <optix_device.h>

#define EPS 1e-3f
#define INFTY 1e10f
#define PI float(3.141592653)

inline __device__ owl::vec3f clampvec(owl::vec3f v, float f) {
    return owl::vec3f(owl::clamp(v.x, 0.0f, f), owl::clamp(v.y, 0.0f, f), owl::clamp(v.z, 0.0f, f));
}

inline __device__ bool nearZero(const owl::vec3f& v) {
    return v.x < EPS && v.y < EPS && v.z < EPS;
}

inline __device__ bool isZero(const owl::vec3f& v) {
    return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f;
}

inline __device__ float norm(owl::vec3f v) {
    return sqrtf(dot(v, v));
}

inline __device__ owl::vec3f multiplyColor(const owl::vec3f &a, const owl::vec3f &b) {
    return owl::vec3f(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ owl::vec3f randomPointInUnitSphere(Random &random) {
    const float u = random();
    const float v = random();
    const float theta = 2.0f * PI * u;
    const float phi = acosf(2.0f * v - 1.0f);

    return owl::vec3f(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));
}

inline __device__ void randomUnitVector(Random &random, owl::vec3f &vec) {
    do {
        vec.x = 2.0f * random() - 1.0f;
        vec.y = 2.0f * random() - 1.0f;
        vec.z = 2.0f * random() - 1.0f;
    } while (dot(vec, vec) >= 1.0f);
    vec = normalize(vec);
}

inline __device__ owl::vec3f cosineSampleHemisphere(const owl::vec3f &normal, Random &random) {
    return normalize(normal + randomPointInUnitSphere(random) * (1.0f - EPS));
}

inline __device__ owl::vec3f reflect(const owl::vec3f &incoming, const owl::vec3f &normal) {
    return incoming - 2.0f * dot(incoming, normal) * normal;
}

inline __device__ owl::vec3f reflectDiffuse(const owl::vec3f &normal, Random &random) {
    return cosineSampleHemisphere(normal, random);
}

inline __device__ owl::vec3f refract(const owl::vec3f &incoming, const owl::vec3f &normal, const float refractionIndex) {
    float cosTheta = -dot(incoming, normal);
    float mu;
    if(cosTheta > 0.0f) {
        mu = 1.0f / refractionIndex;
    } else {
        mu = refractionIndex;
        cosTheta = -cosTheta;
    }

    const float cosPhi = 1.0f - mu * mu * (1.0f - cosTheta * cosTheta);

    if (cosPhi >= 0.0f) {
        return mu * incoming + (mu * cosTheta - sqrtf(cosPhi)) * normal;
    } else {
        return reflect(incoming, normal);
    }
}

inline __device__ float schlickFresnelAprox(const float cos, const float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cos, 5.0f);
}

inline __device__ owl::vec3f getPrimitiveNormal(const TrianglesGeomData& self) {
    const unsigned int primID = optixGetPrimitiveIndex();
    const owl::vec3i index = self.index[primID];
    const owl::vec3f &A = self.vertex[index.x];
    const owl::vec3f &B = self.vertex[index.y];
    const owl::vec3f &C = self.vertex[index.z];

    return normalize(cross(B - A, C - A));
}

inline __device__ void storePhoton(const Light &self, int photonIndex, 
                                  const owl::vec3f &position, const owl::vec3f &direction, 
                                  const owl::vec3f &power, bool isDead = false) {
    Photon &photon = self.photons[photonIndex];
    if (isDead) {
        photon.isDead = true;
    } else {
        photon.coords[0] = position.x;
        photon.coords[1] = position.y;
        photon.coords[2] = position.z;
        photon.colour[0] = power.x;
        photon.colour[1] = power.y;
        photon.colour[2] = power.z;
        photon.power[0] = power.x;
        photon.power[1] = power.y;
        photon.power[2] = power.z;
        photon.dir[0] = direction.x;
        photon.dir[1] = direction.y;
        photon.dir[2] = direction.z;
        photon.isDead = false;
    }
}