#include "obj.cuh"
#include "tiny_obj_loader.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>

#include "../world.cuh"

namespace {

// Bundle a tinyobj face reference so we can group across multiple shapes by usemtl.
struct FaceRef {
    const tinyobj::shape_t *shape;
    size_t face_index;     // index into shape->mesh.num_face_vertices
    size_t index_offset;   // running offset into shape->mesh.indices
};

// Build a Mesh from a list of FaceRefs that all share the same usemtl. Vertices
// are compacted: only positions referenced by this group are copied, and indices
// are remapped to the local range.
Mesh *build_submesh(const tinyobj::attrib_t &attrib,
                    const std::vector<FaceRef> &faces,
                    bool faceted,
                    const std::string &obj_path,
                    const std::string &usemtl_name) {
    auto *mesh = new Mesh();
    mesh->faceted = faceted;
    const bool has_uvs = !attrib.texcoords.empty();

    // Remap original vertex_index → submesh-local index.
    std::unordered_map<int, int> remap;
    bool uv_seam_warned = false;

    for (const auto &fr : faces) {
        const int fv = fr.shape->mesh.num_face_vertices[fr.face_index];
        if (fv != 3) {
            std::cerr << "WARNING: skipping non-triangular face (fv=" << fv
                      << ") in '" << obj_path << "'" << std::endl;
            continue;
        }

        const tinyobj::index_t corners[3] = {
            fr.shape->mesh.indices[fr.index_offset + 0],
            fr.shape->mesh.indices[fr.index_offset + 1],
            fr.shape->mesh.indices[fr.index_offset + 2],
        };

        int tri_local[3];
        for (int c = 0; c < 3; c++) {
            const int orig = corners[c].vertex_index;
            auto it = remap.find(orig);
            int local;
            if (it == remap.end()) {
                local = static_cast<int>(mesh->vertices.size());
                remap.emplace(orig, local);
                mesh->vertices.emplace_back(
                    attrib.vertices[3 * orig + 0],
                    attrib.vertices[3 * orig + 1],
                    attrib.vertices[3 * orig + 2]
                );
                if (has_uvs) mesh->uvs.emplace_back(0.f, 0.f);
            } else {
                local = it->second;
            }
            tri_local[c] = local;

            if (has_uvs && corners[c].texcoord_index >= 0) {
                const owl::vec2f uv(
                    attrib.texcoords[2 * corners[c].texcoord_index + 0],
                    attrib.texcoords[2 * corners[c].texcoord_index + 1]
                );
                auto &slot = mesh->uvs[local];
                if (!uv_seam_warned && (slot.x != 0.f || slot.y != 0.f) &&
                    (slot.x != uv.x || slot.y != uv.y)) {
                    std::cerr << "WARNING: UV seam in '" << obj_path
                              << "' (usemtl='" << usemtl_name
                              << "', vertex " << orig
                              << " has multiple UVs); keeping last-written. "
                              << "Vertex deduplication per face-corner needed for clean seams."
                              << std::endl;
                    uv_seam_warned = true;
                }
                slot = uv;
            }
        }

        mesh->indices.emplace_back(tri_local[0], tri_local[1], tri_local[2]);

        if (faceted) {
            const owl::vec3f v0 = mesh->vertices[tri_local[0]];
            const owl::vec3f v1 = mesh->vertices[tri_local[1]];
            const owl::vec3f v2 = mesh->vertices[tri_local[2]];
            mesh->normals.push_back(normalize(cross(v1 - v0, v2 - v0)));
        }
    }

    if (!faceted) {
        mesh->normals.assign(mesh->vertices.size(), owl::vec3f(0.f));
        for (const auto &tri : mesh->indices) {
            const owl::vec3f v0 = mesh->vertices[tri.x];
            const owl::vec3f v1 = mesh->vertices[tri.y];
            const owl::vec3f v2 = mesh->vertices[tri.z];
            const owl::vec3f n = cross(v1 - v0, v2 - v0);
            mesh->normals[tri.x] = mesh->normals[tri.x] + n;
            mesh->normals[tri.y] = mesh->normals[tri.y] + n;
            mesh->normals[tri.z] = mesh->normals[tri.z] + n;
        }
        for (auto &n : mesh->normals) n = normalize(n);
    }

    return mesh;
}

// Lossy translation from a tinyobj .mtl entry into our Material enum.
// Heuristic — meant to get reasonable defaults; users override per-material via XML.
Material *translate_mtl(const tinyobj::material_t &mtl) {
    auto *m = new Material;
    m->albedo = owl::vec3f(mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2]);

    const float spec_max = std::max({ mtl.specular[0], mtl.specular[1], mtl.specular[2] });
    const float diff_max = std::max({ mtl.diffuse[0],  mtl.diffuse[1],  mtl.diffuse[2]  });
    const bool transparent = mtl.dissolve < 1.0f ||
                             mtl.illum == 4 || mtl.illum == 5 ||
                             mtl.illum == 7 || mtl.illum == 9;

    if (transparent) {
        m->matType = DIELECTRIC;
        m->diffuse = 0.f;
        m->specular = 0.f;
        m->ior = mtl.ior > 0.f ? mtl.ior : 1.5f;
    } else if (spec_max > diff_max && mtl.ior > 1.f) {
        m->matType = CONDUCTOR;
        m->diffuse = 0.f;
        m->specular = spec_max;
        m->ior = mtl.ior;
    } else {
        m->matType = LAMBERTIAN;
        m->diffuse = 1.f;
        m->specular = 0.f;
        m->ior = 0.f;
    }
    return m;
}

} // namespace

std::vector<ObjSubmesh> load_obj_submeshes(
    const std::string &obj_path,
    const bool faceted,
    const bool load_material_files) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mtl_materials;
    std::string warn, err;

    // Compute mtl_basedir explicitly: tinyobj's auto-derivation from the .obj path
    // misbehaves on Windows mixed-separator paths, leaving basedir empty and silently
    // failing to find the .mtl. Trailing separator required.
    std::string basedir;
    if (const auto pos = obj_path.find_last_of("/\\"); pos != std::string::npos) {
        basedir = obj_path.substr(0, pos + 1);
    }
    if (!LoadObj(&attrib, &shapes, &mtl_materials, &warn, &err, obj_path.c_str(),
                 basedir.empty() ? nullptr : basedir.c_str(), /*triangulate=*/true)) {
        throw std::runtime_error("Failed to load OBJ '" + obj_path + "': " + warn + err);
    }
    if (!warn.empty()) {
        std::cerr << "tinyobjloader warning on '" << obj_path << "': " << warn << std::endl;
    }
    std::cerr << "OBJ '" << obj_path << "': "
              << shapes.size() << " shapes, "
              << mtl_materials.size() << " .mtl materials" << std::endl;

    // Bucket faces by usemtl name. material_id == -1 lands in the "" bucket.
    // The bucket-order vector preserves the order in which usemtl groups first
    // appear, so the caller sees them in a stable, file-driven sequence.
    std::vector<std::string> bucket_order;
    std::unordered_map<std::string, std::vector<FaceRef>> buckets;

    for (const auto &shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            const int mid = shape.mesh.material_ids[f];
            std::string name = "";
            if (mid >= 0 && static_cast<size_t>(mid) < mtl_materials.size()) {
                name = mtl_materials[mid].name;
            }
            if (buckets.find(name) == buckets.end()) {
                bucket_order.push_back(name);
            }
            buckets[name].push_back({ &shape, f, index_offset });
            index_offset += shape.mesh.num_face_vertices[f];
        }
    }

    std::cerr << "OBJ '" << obj_path << "': "
              << bucket_order.size() << " usemtl buckets" << std::endl;

    std::vector<ObjSubmesh> out;
    out.reserve(bucket_order.size());
    for (const auto &name : bucket_order) {
        Mesh *m = build_submesh(attrib, buckets[name], faceted, obj_path, name);
        out.push_back({ m, name, /*mtl_material=*/nullptr });
    }

    if (load_material_files) {
        std::unordered_map<std::string, Material *> by_name;
        for (const auto &mtl : mtl_materials) {
            by_name.emplace(mtl.name, translate_mtl(mtl));
            if (!mtl.diffuse_texname.empty()) {
                std::cerr << "INFO: '" << mtl.name << "' references map_Kd='"
                          << mtl.diffuse_texname << "'; texture support lands in Phase 1.3"
                          << std::endl;
            }
        }
        for (auto &sub : out) {
            if (sub.usemtl_name.empty()) continue;
            if (const auto it = by_name.find(sub.usemtl_name); it != by_name.end()) {
                sub.mtl_material = it->second;
            }
        }
    }

    return out;
}
