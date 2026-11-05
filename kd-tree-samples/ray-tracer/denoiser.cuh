#pragma once
// Optional OptiX AI denoiser (HDR model with albedo + normal guides), applied to the
// accumulated image for display/screenshots only. Opt-in per scene via
// <default name="denoise" value="true"/>; never used in benchmark mode.

#include <cstdint>
#include <optix.h>
#include "owl/common/math/vec.h"

class Denoiser {
public:
    explicit Denoiser(OptixDeviceContext context);
    ~Denoiser();

    // Inputs are per-pixel sums over `spp` samples (device memory). Writes the denoised,
    // tonemapped RGBA8 image into `fb` (device-accessible).
    void run(const owl::vec3f *colourSum, const owl::vec3f *albedoSum, const owl::vec3f *normalSum,
             int spp, owl::vec2i size, float exposure, int tonemap, uint32_t *fb);

private:
    void resize(owl::vec2i size);

    OptixDeviceContext context = nullptr;
    OptixDenoiser denoiser = nullptr;
    owl::vec2i size = owl::vec2i(0);
    void *state = nullptr, *scratch = nullptr, *intensity = nullptr;
    size_t stateBytes = 0, scratchBytes = 0;
    owl::vec3f *colour = nullptr, *albedo = nullptr, *normal = nullptr, *output = nullptr;
};
