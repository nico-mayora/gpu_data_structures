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

// Albedo at a hit point. When an albedo texture is bound, interpolate the UVs
// and sample it (sRGB texels approximately linearized); otherwise fall back to
// the material's flat albedo. Shared by the path tracer and the photon mapper
// so caustic/photon colours come from textures too.
inline __device__
owl::vec3f get_albedo_at_hp(const TrianglesGeomData& self, const float u, const float v, const int primID) {
    if (self.albedoTexture == 0 || self.texCoord == nullptr) {
        return self.material->albedo;
    }
    const owl::vec3i tri = self.index[primID];
    const owl::vec2f tc =
        (1.f - u - v) * self.texCoord[tri.x] +
                    u * self.texCoord[tri.y] +
                    v * self.texCoord[tri.z];
    // Texture is created with OWL_COLOR_SPACE_SRGB, so the sample is already linearized.
    const float4 t = tex2D<float4>(self.albedoTexture, tc.x, tc.y);
    return owl::vec3f(t.x, t.y, t.z);
}