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