#include "denoiser.cuh"
#include "../common/tonemap.cuh"

#include <optix_stubs.h>
// OWL lives in its own DLL, so this executable needs its own OptiX function table.
#include <optix_function_table_definition.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#define OPTIX_CHECK(call)                                                                   \
    do {                                                                                    \
        const OptixResult res_ = (call);                                                    \
        if (res_ != OPTIX_SUCCESS) {                                                        \
            std::cerr << "OptiX error " << res_ << " (" << optixGetErrorName(res_) << ") at " \
                      << __FILE__ << ":" << __LINE__ << ": " #call << std::endl;            \
            std::exit(EXIT_FAILURE);                                                        \
        }                                                                                   \
    } while (0)

namespace {
__global__ void average_kernel(const owl::vec3f *colourSum, const owl::vec3f *albedoSum,
                               const owl::vec3f *normalSum, const float inv_spp, const int n,
                               owl::vec3f *colour, owl::vec3f *albedo, owl::vec3f *normal) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    colour[i] = colourSum[i] * inv_spp;
    albedo[i] = albedoSum[i] * inv_spp;
    normal[i] = normalSum[i] * inv_spp;
}

__global__ void tonemap_kernel(const owl::vec3f *hdr, const int n, const float exposure,
                               const int tonemap, uint32_t *fb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const owl::vec3f c = hdr[i];
    fb[i] = pack_rgba8(filter_colour(owl::vec3f(fmaxf(c.x, 0.f), fmaxf(c.y, 0.f), fmaxf(c.z, 0.f)),
                                     exposure, tonemap));
}

OptixImage2D float3_image(void *data, const owl::vec2i size) {
    OptixImage2D img = {};
    img.data = reinterpret_cast<CUdeviceptr>(data);
    img.width = size.x;
    img.height = size.y;
    img.pixelStrideInBytes = sizeof(owl::vec3f);
    img.rowStrideInBytes = size.x * sizeof(owl::vec3f);
    img.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    return img;
}
}

Denoiser::Denoiser(OptixDeviceContext ctx) : context(ctx) {
    OPTIX_CHECK(optixInit());
    OptixDenoiserOptions options = {};
    options.guideAlbedo = 1;
    options.guideNormal = 1;
    OPTIX_CHECK(optixDenoiserCreate(context, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &denoiser));
}

Denoiser::~Denoiser() {
    for (void *p : { state, scratch, intensity,
                     static_cast<void*>(colour), static_cast<void*>(albedo),
                     static_cast<void*>(normal), static_cast<void*>(output) })
        if (p) cudaFree(p);
    if (denoiser) optixDenoiserDestroy(denoiser);
}

void Denoiser::resize(const owl::vec2i newSize) {
    if (newSize == size) return;
    size = newSize;
    for (void **p : { &state, &scratch, reinterpret_cast<void**>(&colour), reinterpret_cast<void**>(&albedo),
                      reinterpret_cast<void**>(&normal), reinterpret_cast<void**>(&output) })
        if (*p) { cudaFree(*p); *p = nullptr; }

    OptixDenoiserSizes sizes = {};
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, size.x, size.y, &sizes));
    stateBytes = sizes.stateSizeInBytes;
    scratchBytes = std::max(sizes.withoutOverlapScratchSizeInBytes, sizes.computeIntensitySizeInBytes);
    cudaMalloc(&state, stateBytes);
    cudaMalloc(&scratch, scratchBytes);
    if (!intensity) cudaMalloc(&intensity, sizeof(float));
    const size_t px = size_t(size.x) * size.y * sizeof(owl::vec3f);
    cudaMalloc(reinterpret_cast<void**>(&colour), px);
    cudaMalloc(reinterpret_cast<void**>(&albedo), px);
    cudaMalloc(reinterpret_cast<void**>(&normal), px);
    cudaMalloc(reinterpret_cast<void**>(&output), px);
    OPTIX_CHECK(optixDenoiserSetup(denoiser, nullptr, size.x, size.y,
                                   reinterpret_cast<CUdeviceptr>(state), stateBytes,
                                   reinterpret_cast<CUdeviceptr>(scratch), scratchBytes));
}

void Denoiser::run(const owl::vec3f *colourSum, const owl::vec3f *albedoSum, const owl::vec3f *normalSum,
                   const int spp, const owl::vec2i fbSize, const float exposure, const int tonemap,
                   uint32_t *fb) {
    if (spp <= 0) return;
    resize(fbSize);
    const int n = size.x * size.y;
    constexpr int block = 256;
    average_kernel<<<(n + block - 1) / block, block>>>(colourSum, albedoSum, normalSum, 1.f / float(spp), n,
                                                     colour, albedo, normal);

    const OptixImage2D colourImg = float3_image(colour, size);
    OPTIX_CHECK(optixDenoiserComputeIntensity(denoiser, nullptr, &colourImg, reinterpret_cast<CUdeviceptr>(intensity),
                                              reinterpret_cast<CUdeviceptr>(scratch), scratchBytes));
    OptixDenoiserParams params = {};
    params.hdrIntensity = reinterpret_cast<CUdeviceptr>(intensity);
    params.blendFactor = 0.f;

    OptixDenoiserGuideLayer guide = {};
    guide.albedo = float3_image(albedo, size);
    guide.normal = float3_image(normal, size);

    OptixDenoiserLayer layer = {};
    layer.input = colourImg;
    layer.output = float3_image(output, size);

    OPTIX_CHECK(optixDenoiserInvoke(denoiser, nullptr, &params,
                                    reinterpret_cast<CUdeviceptr>(state), stateBytes,
                                    &guide, &layer, 1, 0, 0,
                                    reinterpret_cast<CUdeviceptr>(scratch), scratchBytes));
    tonemap_kernel<<<(n + block - 1) / block, block>>>(output, n, exposure, tonemap, fb);
    cudaDeviceSynchronize();
}
