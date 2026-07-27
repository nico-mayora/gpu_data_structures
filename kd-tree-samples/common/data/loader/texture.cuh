#pragma once
#include <string>
#include "owl/owl.h"

// Decode an image from disk and upload it as an OWL 2D texture (RGBA8, sRGB
// color space so samples are hardware-linearized). Results are cached per
// `context` by absolute path, so a texture shared across submeshes is decoded
// and uploaded once. Returns nullptr (and logs) when decoding fails; callers
// should then leave the geom's texture unbound (falls back to flat albedo).
OWLTexture load_albedo_texture(OWLContext context, const std::string &path);
