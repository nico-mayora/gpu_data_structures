#pragma once
// Isotropic GGX (Trowbridge-Reitz) microfacet helpers, shared by the path tracer and the
// photon emitter. Conventions: `n` is the shading normal on the side of `wo`, `wo` points
// away from the surface (towards the previous vertex), `alpha` is Mitsuba's roughness.

#include "owl/common/math/vec.h"

// Below this roughness a lobe is treated as a perfect mirror (delta distribution).
constexpr float GGX_MIN_ALPHA = 1e-3f;

inline __device__
float ggx_D(const float cos_m, const float alpha) {
    if (cos_m <= 0.f) return 0.f;
    const float a2 = alpha * alpha;
    const float d = cos_m * cos_m * (a2 - 1.f) + 1.f;
    return a2 / (3.14159265f * d * d);
}

// Smith masking for one direction.
inline __device__
float ggx_G1(const float cos_v, const float alpha) {
    if (cos_v <= 0.f) return 0.f;
    const float c2 = cos_v * cos_v;
    const float tan2 = fmaxf(0.f, 1.f - c2) / c2;
    return 2.f / (1.f + sqrtf(1.f + alpha * alpha * tan2));
}

inline __device__
void ggx_basis(const owl::vec3f &n, owl::vec3f &t, owl::vec3f &b) {
    t = (fabsf(n.x) > 0.5f) ? owl::vec3f(-n.y, n.x, 0.f) : owl::vec3f(0.f, -n.z, n.y);
    t = normalize(t);
    b = cross(n, t);
}

// Samples a microfacet normal from the distribution of normals visible from `wo`
// (Heitz 2018). Sample weights stay <= 1, unlike D*cos sampling at grazing angles.
inline __device__
owl::vec3f ggx_sample_m(const owl::vec3f &n, const owl::vec3f &wo, const float alpha,
                        const float u1, const float u2) {
    owl::vec3f t, b;
    ggx_basis(n, t, b);
    // View direction in the local frame, stretched to the hemisphere configuration.
    const owl::vec3f vh = normalize(owl::vec3f(alpha * dot(wo, t), alpha * dot(wo, b), fmaxf(dot(wo, n), 0.f)));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const owl::vec3f T1 = lensq > 0.f ? owl::vec3f(-vh.y, vh.x, 0.f) / sqrtf(lensq) : owl::vec3f(1.f, 0.f, 0.f);
    const owl::vec3f T2 = cross(vh, T1);
    const float r = sqrtf(u1);
    const float phi = 2.f * 3.14159265f * u2;
    const float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    const float s = 0.5f * (1.f + vh.z);
    t2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - t1 * t1)) + s * t2;
    const owl::vec3f nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.f, 1.f - t1 * t1 - t2 * t2)) * vh;
    const owl::vec3f m = normalize(owl::vec3f(alpha * nh.x, alpha * nh.y, fmaxf(1e-6f, nh.z)));
    return normalize(t * m.x + b * m.y + n * m.z);
}

// Solid-angle pdf of sampling `wi` by reflecting `wo` about a visible-normal sample:
// G1(wo) * D(m) / (4 cos_o).
inline __device__
float ggx_pdf(const owl::vec3f &n, const owl::vec3f &wo, const owl::vec3f &wi, const float alpha) {
    const owl::vec3f h = normalize(wo + wi);
    const float cos_m = dot(n, h);
    const float cos_o = dot(n, wo);
    if (cos_m <= 0.f || cos_o <= 0.f || dot(wo, h) <= 0.f) return 0.f;
    return ggx_G1(cos_o, alpha) * ggx_D(cos_m, alpha) / (4.f * cos_o);
}

// BRDF * cos(wi), without the Fresnel factor.
inline __device__
float ggx_eval_cos(const owl::vec3f &n, const owl::vec3f &wo, const owl::vec3f &wi, const float alpha) {
    const float cos_o = dot(n, wo);
    const float cos_i = dot(n, wi);
    if (cos_o <= 0.f || cos_i <= 0.f) return 0.f;
    const owl::vec3f h = normalize(wo + wi);
    return ggx_D(dot(n, h), alpha) * ggx_G1(cos_o, alpha) * ggx_G1(cos_i, alpha) / (4.f * cos_o);
}

// Samples a reflected direction. Returns BRDF*cos/pdf without Fresnel (0 if the sample is
// lost below the surface) and the half vector through `h`.
inline __device__
float ggx_sample(const owl::vec3f &n, const owl::vec3f &wo, const float alpha,
                 const float u1, const float u2, owl::vec3f &wi, owl::vec3f &h) {
    h = ggx_sample_m(n, wo, alpha, u1, u2);
    const float wo_h = dot(wo, h);
    if (wo_h <= 0.f) return 0.f;
    wi = 2.f * wo_h * h - wo;
    const float cos_o = dot(n, wo);
    const float cos_i = dot(n, wi);
    if (cos_i <= 0.f || cos_o <= 0.f) return 0.f;
    return ggx_G1(cos_i, alpha); // BRDF*cos/pdf for visible-normal sampling
}

inline __device__
float power_heuristic(const float a, const float b) {
    const float a2 = a * a, b2 = b * b;
    return (a2 + b2) > 0.f ? a2 / (a2 + b2) : 0.f;
}
