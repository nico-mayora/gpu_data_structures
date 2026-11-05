#include "./photonEmitter.cuh"
#include "./helpers.cu"
#include "owl/common/math/vec.h"
#include "../../common/helpers.cuh"
#include "../../common/microfacet.cuh"

#define PHOTON_ATTENUATION_FACTOR 150
#define ATTENUATE_PHOTONS false

#include <optix_device.h>

using namespace owl;

inline __device__ void savePhoton(const PhotonMapperRGD &self, PhotonMapperPRD &prd) {
  const int photonIndex = atomicAdd(self.photonsCount, 1);

  const auto photon = &self.photons[photonIndex];
  photon->power[0] = photon->power[1] = photon->power[2] = 0.f;

  auto colour = prd.color / static_cast<float>(self.totalPhotons);
  photon->colour[0] = colour[0];
  photon->colour[1] = colour[1];
  photon->colour[2] = colour[2];

  auto coords = prd.scattered.origin;
  photon->coords[0] = coords[0];
  photon->coords[1] = coords[1];
  photon->coords[2] = coords[2];

  auto dir = prd.direction;
  photon->dir[0] = dir[0];
  photon->dir[1] = dir[1];
  photon->dir[2] = dir[2];
}

inline __device__ void updateScatteredRay(Ray &ray, PhotonMapperPRD &prd) {
  ray.origin = prd.scattered.origin;
  ray.direction = prd.scattered.direction;
  prd.direction = prd.scattered.direction;
  prd.color = prd.scattered.color;
}

inline __device__ void shootPhoton(const PhotonMapperRGD &self, Ray &ray, PhotonMapperPRD &prd) {
  for (int i = 0; i < self.maxDepth; i++) {
    owl::traceRay(self.world, ray, prd);

    if (prd.event == MISS) {
      break;
    }

    if (prd.hitDiffuse) {
      savePhoton(self, prd);
    }

    if (prd.event == ABSORBED) {
      break;
    }

    updateScatteredRay(ray, prd);
  }
}

inline __device__ void shootCausticsPhoton(const PhotonMapperRGD &self, Ray &ray, PhotonMapperPRD &prd) {
  bool hadSpecular = false;
  for (int i = 0; i < self.maxDepth; i++) {
    owl::traceRay(self.world, ray, prd);

    if (prd.event == MISS) {
      break;
    }

    if (prd.hitDiffuse) {
      if (hadSpecular) {
        savePhoton(self, prd);
      }
      // A coated (plastic) surface may have reflected the photon off its clear coat:
      // keep following it, the reflected light is a caustic for whatever it lands on.
      if (!prd.hitSpecular || prd.event != SCATTER_SPECULAR) break;
    }
    hadSpecular |= prd.hitSpecular;

    if (prd.event == ABSORBED) {
      break;
    }

    updateScatteredRay(ray, prd);
  }
}

// Orthonormal basis (t, b) spanning the plane perpendicular to unit vector n.
inline __device__ void orthoBasis(const vec3f &n, vec3f &t, vec3f &b) {
  if (fabsf(n.x) > fabsf(n.z)) t = normalize(vec3f(-n.y, n.x, 0.f));
  else                         t = normalize(vec3f(0.f, -n.z, n.y));
  b = cross(n, t);
}

OPTIX_RAYGEN_PROGRAM(pointLightRayGen)(){
  const auto &self = owl::getProgramData<PointLightRGD>();
  const vec2i id = owl::getLaunchIndex();

  PhotonMapperPRD prd;
  prd.random.init(id.x, id.y);
  prd.color = self.color;

  Ray ray;
  ray.tmin = EPS;

  if (self.lightType == LIGHT_DIRECTIONAL) {
    // Parallel rays along `direction`, launched from a disk (perpendicular to it,
    // sized to the scene bounding sphere) backed off behind the scene.
    vec3f t, b; orthoBasis(self.direction, t, b);
    const float r = self.diskRadius * sqrtf(prd.random());
    const float phi = 2.f * PI * prd.random();
    const vec3f offset = (cosf(phi) * t + sinf(phi) * b) * r;
    ray.origin = self.diskCenter - self.direction * (2.f * self.diskRadius) + offset;
    ray.direction = self.direction;
    prd.direction = self.direction;
  } else if (self.lightType == LIGHT_SPOT) {
    // Sample uniformly within the outer cone around `direction`.
    vec3f t, b; orthoBasis(self.direction, t, b);
    const float cosT = 1.f - prd.random() * (1.f - self.cosOuter);
    const float sinT = sqrtf(fmaxf(0.f, 1.f - cosT * cosT));
    const float phi = 2.f * PI * prd.random();
    const vec3f dir = normalize(t * (sinT * cosf(phi)) + b * (sinT * sinf(phi))
                                + self.direction * cosT);
    ray.origin = self.position;
    ray.direction = dir;
    prd.direction = dir;
    // Soften the penumbra: weight photon energy by the spot falloff at this angle.
    float cw = (self.cosInner > self.cosOuter)
             ? (cosT - self.cosOuter) / (self.cosInner - self.cosOuter) : 1.f;
    cw = fminf(fmaxf(cw, 0.f), 1.f);
    prd.color = self.color * (cw * cw * (3.f - 2.f * cw));
  } else if (self.lightType == LIGHT_AREA) {
    // Uniform point on the rectangle, cosine-distributed direction about its normal:
    // every photon then carries the same share of the emitter's flux.
    const float a = 2.f * prd.random() - 1.f;
    const float b = 2.f * prd.random() - 1.f;
    const vec3f dir = cosineSampleHemisphere(self.direction, prd.random);
    ray.origin = self.position + a * self.edgeU + b * self.edgeV;
    ray.direction = dir;
    prd.direction = dir;
  } else { // LIGHT_POINT
    const vec3f dir = randomPointInUnitSphere(prd.random);
    ray.origin = self.position;
    ray.direction = dir;
    prd.direction = dir;
  }

  if (self.causticsMode) {
    shootCausticsPhoton(self, ray, prd);
  } else {
    shootPhoton(self, ray, prd);
  }
}

// Unpolarised Fresnel reflectance; eta = n_incident / n_transmitted. 1 on total internal reflection.
inline __device__ float fresnelDielectric(const float cos_i, const float eta, float &cos_t) {
  const float sin2_t = eta * eta * fmaxf(0.f, 1.f - cos_i * cos_i);
  if (sin2_t >= 1.f) { cos_t = 0.f; return 1.f; }
  cos_t = sqrtf(1.f - sin2_t);
  const float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
  const float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
  return 0.5f * (rs * rs + rp * rp);
}

inline __device__ void absorb(PhotonMapperPRD &prd, const vec3f &hitPoint) {
  prd.event = ABSORBED;
  prd.scattered.origin = hitPoint;
}

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshClosestHit)(){
  auto &prd = owl::getPRD<PhotonMapperPRD>();
  const auto &self = owl::getProgramData<TrianglesGeomData>();
  const Material &mat = *self.material;

  const auto [u, v] = optixGetTriangleBarycentrics();
  const int primID = optixGetPrimitiveIndex();
  const vec3f surfaceAlbedo = get_albedo_at_hp(self, u, v, primID);
  const vec3f rayDir = optixGetWorldRayDirection();
  const vec3f rayOrg = optixGetWorldRayOrigin();
  const vec3f hitPoint = rayOrg + optixGetRayTmax() * rayDir;

  // Shading normal, flipped to face the incoming photon (surfaces are two-sided).
  const vec3f Ns = get_normal_at_hp(self, u, v, primID);
  const bool frontFace = dot(Ns, rayDir) < 0.f;
  const vec3f n = frontFace ? Ns : -Ns;

  prd.hitSpecular = false;
  prd.scattered.origin = hitPoint;

  // Masked surface letting the photon through untouched.
  if (mat.opacity < 1.f && prd.random() >= mat.opacity) {
    prd.hitDiffuse = false;
    prd.event = SCATTER_SPECULAR;
    prd.scattered.direction = rayDir;
    prd.scattered.color = prd.color;
    return;
  }

  // Photons are stored on diffuse surfaces, except emitters (they absorb, and photons
  // stored there would only leak into density estimates on nearby surfaces).
  const bool emissive = mat.emission.x > 0.f || mat.emission.y > 0.f || mat.emission.z > 0.f;
  prd.hitDiffuse = (mat.matType == LAMBERTIAN) && !emissive;

  // Clear coat (plastics): Fresnel-weighted glossy reflection before the diffuse base.
  // Decided before the albedo roulette: the coat reflects regardless of the base colour.
  // Conditional on not being reflected, the base's albedo is exactly get_albedo_at_hp's
  // coated albedo (it already divides out the average incident Fresnel loss).
  // Reflected photons count as specular, so light mirrored by e.g. a glossy floor lands in
  // the caustic map (the tracer's gathers only see diffuse radiance at the gather point).
  if (mat.matType == LAMBERTIAN && mat.coat_r0 > 0.f) {
    const vec3f wo = -rayDir;
    const float cos_i = fmaxf(0.f, dot(wo, n));
    const float F = mat.coat_r0 + (1.f - mat.coat_r0) * powf(1.f - cos_i, 5.f);
    if (prd.random() < F) {
      vec3f reflected = reflect(rayDir, n);
      if (mat.coat_roughness >= GGX_MIN_ALPHA) {
        vec3f h;
        const float u1 = prd.random(), u2 = prd.random();
        const float w = ggx_sample(n, wo, mat.coat_roughness, u1, u2, reflected, h);
        if (w <= 0.f || prd.random() >= w) { absorb(prd, hitPoint); return; }
      }
      prd.event = SCATTER_SPECULAR;
      prd.hitSpecular = true;
      prd.scattered.direction = reflected;
      prd.scattered.color = prd.color;   // coat reflection is untinted
      return;
    }
  }

  // Russian roulette on the surface albedo.
  if (prd.random() >= pIndex(prd.color, surfaceAlbedo, prd.debug)) {
    absorb(prd, hitPoint);
    return;
  }
  const vec3f survivingColor = calculatePhotonColor(prd.color, surfaceAlbedo, prd.debug);

  switch (mat.matType) {
    case LAMBERTIAN:
      prd.event = SCATTER_DIFFUSE;
      prd.scattered.direction = reflectDiffuse(n, prd.random);
      prd.scattered.color = survivingColor;
      return;
    case CONDUCTOR: {
      vec3f reflected = reflect(rayDir, n);
      vec3f color = survivingColor;
      if (mat.roughness >= GGX_MIN_ALPHA) {
        // GGX lobe; roulette on the sample weight (shadowing/masking losses).
        vec3f h;
        const float u1 = prd.random(), u2 = prd.random();
        const float w = ggx_sample(n, -rayDir, mat.roughness, u1, u2, reflected, h);
        if (w <= 0.f || prd.random() >= fminf(w, 1.f)) {
          absorb(prd, hitPoint);
          return;
        }
        if (w > 1.f) color *= w;
      }
      prd.event = SCATTER_SPECULAR;
      prd.hitSpecular = true;
      prd.scattered.direction = reflected;
      prd.scattered.color = color;
      return;
    }
    case DIELECTRIC: {
      const float cos_i = fminf(fmaxf(-dot(rayDir, n), 0.f), 1.f);
      const float ior = mat.ior > 0.f ? mat.ior : 1.5f;
      const float eta = (frontFace || mat.thin) ? 1.f / ior : ior;
      float cos_t;
      float F = fresnelDielectric(cos_i, eta, cos_t);
      if (mat.thin && F < 1.f) F = 2.f * F / (1.f + F);

      prd.hitSpecular = true;
      prd.scattered.color = survivingColor;
      if (prd.random() < F) {
        prd.event = SCATTER_SPECULAR;
        prd.scattered.direction = reflect(rayDir, n);
        return;
      }
      // Transmitted lobe, scaled by specular_transmittance via roulette.
      if (prd.random() >= mat.transmittance) {
        absorb(prd, hitPoint);
        return;
      }
      prd.event = SCATTER_REFRACT;
      prd.scattered.direction = mat.thin ? rayDir : normalize(eta * rayDir + (eta * cos_i - cos_t) * n);
      return;
    }
  }
}

OPTIX_MISS_PROGRAM(miss)(){
  auto &prd = owl::getPRD<PhotonMapperPRD>();
  prd.event = MISS;
  prd.hitDiffuse = false;
  prd.hitSpecular = false;
}