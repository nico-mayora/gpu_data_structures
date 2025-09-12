#pragma once

#include "photonEmitter.cuh"

#define RANDVEC3F owl::vec3f(rnd(),rnd(),rnd())
#define INFTY 1e10
#define EPS 1e-3f
#define PI float(3.141592653)
typedef owl::LCG<> Random;

inline __device__ owl::vec3f clampvec(owl::vec3f v, float f) {
  return owl::vec3f(owl::clamp(v.x, f), owl::clamp(v.y, f), owl::clamp(v.z, f));
}

inline __device__ bool nearZero(const owl::vec3f& v) {
  return v.x < EPS && v.y < EPS && v.z < EPS;
}

inline __device__ bool isZero(const owl::vec3f& v) {
  return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}

inline __device__ float norm(owl::vec3f v) {
  return sqrtf(dot(v, v));
}

inline __device__ owl::vec3f randomPointInUnitSphere(Random &random) {
  const float u = random();
  const float v = random();
  const float theta = 2.f * PI * u;
  const float phi = acosf(2.f * v - 1.f);

  return owl::vec3f(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}

inline __device__ void randomUnitVector(Random &random, owl::vec3f &vec) {
  do {
    vec.x = 2.f*random() - 1.f;
    vec.y = 2.f*random() - 1.f;
    vec.z = 2.f*random() - 1.f;
  } while (dot(vec, vec) >= 1.f);
  vec = normalize(vec);
}

inline __device__ owl::vec3f cosineSampleHemisphere(const owl::vec3f &normal, Random &random) {
  return normalize(normal + randomPointInUnitSphere(random) * (1 - EPS));
}

inline __device__ owl::vec3f reflect(const owl::vec3f &incoming, const owl::vec3f &normal) {
  return incoming - 2.f * dot(incoming, normal) * normal;
}

inline __device__ owl::vec3f reflectDiffuse(const owl::vec3f &normal, Random &random) {
  return cosineSampleHemisphere(normal, random);
}

inline __device__ owl::vec3f refract(const owl::vec3f &incoming, const owl::vec3f &normal, const float refractionIndex) {
  float cosTheta = -dot(incoming, normal);
  float mu;
  if(cosTheta > 0.f) {
    mu = 1.f / refractionIndex;
  } else {
    mu = refractionIndex;
    cosTheta = -cosTheta;
  }

  const float cosPhi = 1.f - mu * mu * (1.f - cosTheta * cosTheta);

  if (cosPhi >= 0) {
    return mu * incoming + (mu * cosTheta - sqrtf(cosPhi)) * normal;
  } else {
    return reflect(incoming, normal);
  }
}

inline __device__ owl::vec3f getPrimitiveNormal(const TrianglesGeomData& self) {
  using namespace owl;
  const unsigned int primID = optixGetPrimitiveIndex();
  const vec3i index  = self.index[primID];
  const vec3f &A     = self.vertex[index.x];
  const vec3f &B     = self.vertex[index.y];
  const vec3f &C     = self.vertex[index.z];

  return normalize(cross(B-A,C-A));
}

inline __device__ owl::vec3f multiplyColor(const owl::vec3f &a, const owl::vec3f &b) {
  return owl::vec3f(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ float pIndex(const owl::vec3f &photon_color, const owl::vec3f &material_coeffs) {
  const auto pd_top = max(material_coeffs.x * photon_color.x,
                        max(material_coeffs.y * photon_color.y,
                            material_coeffs.z * photon_color.z));
  const auto pd_bottom = max(photon_color.x,
                         max(photon_color.y,
                             photon_color.z));
  return pd_top / pd_bottom;
}

inline __device__ owl::vec3f calculatePhotonColor(const owl::vec3f &photon_color, const owl::vec3f &material_coeffs) {
  const auto p_index = pIndex(photon_color, material_coeffs);

  return {photon_color.x * material_coeffs.x / p_index,
          photon_color.y * material_coeffs.y / p_index,
          photon_color.z * material_coeffs.z / p_index};
}

inline __device__
bool refract(const owl::vec3f& v,
             const owl::vec3f& n,
             const float ni_over_nt,
             owl::vec3f &refracted)
{
  const owl::vec3f uv = normalize(v);
  const float dt = dot(uv, n);
  const float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);
  if (discriminant > 0.f) {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
    return true;
  }

  return false;
}

inline __device__ float schlickFresnelAprox(const float cos, const float ior) {
  float r0 = (1. - ior) / (1. + ior);
  r0 = r0 * r0;
  return r0 + (1. - r0) * pow(1. - cos, 5);
}