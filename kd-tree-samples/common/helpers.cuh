#pragma once

inline __device__
owl::vec3f get_normal_at_hp(const TrianglesGeomData& self, const float u, const float v, const int primID) {
    const owl::vec3i tri = self.index[primID];

    owl::vec3f N;
    if (self.faceted) {
        N = self.normal[primID];
    } else {
        N = normalize(
            (1.f - u - v) * self.normal[tri.x] +
                        u * self.normal[tri.y] +
                        v * self.normal[tri.z]
        );
    }
    return N;
}

// Diffuse albedo seen through a clear coat (internal reflections + light-side Fresnel).
inline __device__
owl::vec3f coated_albedo(const Material &m, const owl::vec3f &rho) {
    if (m.coat_r0 <= 0.f) return rho;
    const owl::vec3f denom = owl::vec3f(1.f) - (m.coat_nonlinear ? rho : owl::vec3f(1.f)) * m.coat_fdr_int;
    return rho * m.coat_diffuse_scale / denom;
}

inline __device__
owl::vec3f get_albedo_at_hp(const TrianglesGeomData& self, const float u, const float v, const int primID) {
    if (self.albedoTexture == 0 || self.texCoord == nullptr) {
        return coated_albedo(*self.material, self.material->albedo);
    }
    const owl::vec3i tri = self.index[primID];
    const owl::vec2f tc =
        (1.f - u - v) * self.texCoord[tri.x] +
                    u * self.texCoord[tri.y] +
                    v * self.texCoord[tri.z];
    // Texture is created with OWL_COLOR_SPACE_SRGB, so the sample is already linearised.
    const float4 t = tex2D<float4>(self.albedoTexture, tc.x, tc.y);
    return coated_albedo(*self.material, owl::vec3f(t.x, t.y, t.z));
}