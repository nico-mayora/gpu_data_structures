#pragma once
#include <string>
#include <vector>

struct Mesh;
struct Material;

// One per `usemtl` group inside an OBJ. Single-material OBJs (no `usemtl`
// directive, or one group) yield a 1-entry vector with `usemtl_name = ""`.
struct ObjSubmesh {
    Mesh *mesh = nullptr;
    std::string usemtl_name; // empty when the source has no material assignment
    // Populated only when load_material_files=true and the .mtl provided an entry.
    // Translation is lossy (see obj.cu); caller is expected to override via XML when needed.
    Material *mtl_material = nullptr;
    // Absolute path to the .mtl's map_Kd albedo texture, resolved against the OBJ
    // directory. Empty when the material has no diffuse texture. Phase 1.3.
    std::string albedo_texture_path;
};

std::vector<ObjSubmesh> load_obj_submeshes(
    const std::string &obj_path,
    bool faceted,
    bool load_material_files);
