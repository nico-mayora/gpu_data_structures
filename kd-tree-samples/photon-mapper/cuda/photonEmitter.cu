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
  // Per-photon flux dPhi = Phi / N. The host folds the per-light emission factor (4*pi
  // for a point's full sphere, the cone solid angle for a spot, the disk area for a
  // directional) into prd.color's initial value, so prd.color already tracks total flux
  // Phi * throughput; here we only divide by the launch count N.
  photon->color = prd.color / static_cast<float>(self.totalPhotons);
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

// Orthonormal basis (t, b) spanning the plane perpendicular to unit vector n.
inline __device__ void orthoBasis(const vec3f &n, vec3f &t, vec3f &b) {
  if (fabsf(n.x) > fabsf(n.z)) t = normalize(vec3f(-n.y, n.x, 0.f));
  else                         t = normalize(vec3f(0.f, -n.z, n.y));
  b = cross(n, t);
}

// Store a volume photon at an in-medium scatter point. Like savePhoton, but the position
// is the scatter point (not a surface hit) and prd.direction is the photon's incoming
// travel direction (the omega_in fed to the phase function during the gather).
inline __device__ void saveVolumePhoton(const PhotonMapperRGD &self, PhotonMapperPRD &prd, const vec3f &pos) {
  const int idx = atomicAdd(self.photonsCount, 1);
  auto photon = &self.photons[idx];
  photon->color = prd.color / static_cast<float>(self.totalPhotons);
  photon->pos = pos;
  photon->dir = prd.direction;
}

// Importance-sample a scattered direction from the Henyey-Greenstein phase function,
// relative to the current travel direction `wo` (g>0 = forward scattering).
inline __device__ vec3f sampleHenyeyGreenstein(const vec3f &wo, const float g, Random &rnd) {
  float cosT;
  if (fabsf(g) < 1e-3f) {
    cosT = 1.f - 2.f * rnd();
  } else {
    const float s = (1.f - g * g) / (1.f + g - 2.f * g * rnd());
    cosT = (1.f + g * g - s * s) / (2.f * g);
  }
  const float sinT = sqrtf(fmaxf(0.f, 1.f - cosT * cosT));
  const float phi = 2.f * PI * rnd();
  vec3f t, b; orthoBasis(wo, t, b);
  return normalize(t * (sinT * cosf(phi)) + b * (sinT * sinf(phi)) + wo * cosT);
}

// Volume photon tracing through a global homogeneous medium. Free-flight sampling places
// the photon at a medium collision (deposit + HG scatter) or lets it reach a surface; the
// transmittance between events is handled implicitly by the exponential distance sampling.
inline __device__ void shootVolumePhoton(const PointLightRGD &self, Ray &ray, PhotonMapperPRD &prd) {
  const float sigma_t = self.sigmaT;
  const float albedo_mean = (self.mediumAlbedo.x + self.mediumAlbedo.y + self.mediumAlbedo.z) / 3.f;
  const float maxDist = 2.f * self.diskRadius;   // scene-extent cap for rays that miss geometry

  for (int i = 0; i < self.maxDepth; i++) {
    owl::traceRay(self.world, ray, prd);

    const float tHit = (prd.event == MISS) ? maxDist
                     : length(prd.scattered.origin - ray.origin);
    const float tScatter = -logf(1.f - prd.random()) / sigma_t;

    if (tScatter < tHit) {
      // Real collision in the medium: deposit, then Russian-roulette absorb vs scatter.
      const vec3f pos = ray.origin + tScatter * ray.direction;
      saveVolumePhoton(self, prd, pos);

      if (prd.random() >= albedo_mean) break;                       // absorbed
      prd.color = prd.color * (self.mediumAlbedo / albedo_mean);    // RGB, unbiased survival

      const vec3f newDir = sampleHenyeyGreenstein(ray.direction, self.mediumG, prd.random);
      ray.origin = pos;
      ray.direction = newDir;
      prd.direction = newDir;
      continue;
    }

    // Reached a surface before colliding in the medium.
    if (prd.event == MISS) break;
    if (prd.event == SCATTER_SPECULAR || prd.event == SCATTER_REFRACT) {
      updateScatteredRay(ray, prd);   // pass through / mirror, keep marching the medium
      continue;
    }
    break;  // diffuse / absorbed: the energy belongs to the surface maps, stop here
  }
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
  } else { // LIGHT_POINT
    const vec3f dir = randomPointInUnitSphere(prd.random);
    ray.origin = self.position;
    ray.direction = dir;
    prd.direction = dir;
  }

  if (self.volumeMode) {
    shootVolumePhoton(self, ray, prd);
  } else if (self.causticsMode) {
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