#pragma once
#include <string>
#include <vector>

struct Mesh;
struct Material;

struct ObjSubmesh {
    Mesh *mesh = nullptr;
    std::string usemtl_name;
    Material *mtl_material = nullptr;
    std::string albedo_texture_path;
};

std::vector<ObjSubmesh> load_obj_submeshes(
    const std::string &obj_path,
    bool faceted,
    bool load_material_files);
