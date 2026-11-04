#include "texture.cuh"

#include <iostream>
#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace {
std::unordered_map<OWLContext, std::unordered_map<std::string, OWLTexture>> g_cache;
}

OWLTexture load_albedo_texture(OWLContext context, const std::string &path) {
    auto &per_ctx = g_cache[context];
    if (const auto it = per_ctx.find(path); it != per_ctx.end()) {
        return it->second;
    }

    int w = 0, h = 0, channels = 0;
    stbi_set_flip_vertically_on_load(1);
    unsigned char *pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (pixels == nullptr) {
        std::cerr << "WARNING: failed to load texture '" << path << "': "
                  << stbi_failure_reason() << " (surface will use flat albedo)" << std::endl;
        per_ctx.emplace(path, nullptr);
        return nullptr;
    }

    const OWLTexture tex = owlTexture2DCreate(
        context,
        OWL_TEXEL_FORMAT_RGBA8,
        static_cast<uint32_t>(w), static_cast<uint32_t>(h),
        pixels,
        OWL_TEXTURE_LINEAR,
        OWL_TEXTURE_WRAP, OWL_TEXTURE_WRAP,
        OWL_COLOR_SPACE_SRGB);

    stbi_image_free(pixels);
    per_ctx.emplace(path, tex);
    return tex;
}
