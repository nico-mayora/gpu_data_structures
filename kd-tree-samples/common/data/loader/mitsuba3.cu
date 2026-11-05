#include "mitsuba3.cuh"
#include "obj.cuh"
#include "../world.cuh"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
constexpr auto DEFAULT_SCENES_FOLDER = "scenes";

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::string scenesFolderFromEnvFile() {
    std::ifstream env(".env");
    std::string line;
    while (std::getline(env, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos || trim(line.substr(0, eq)) != "SCENES_DIR") continue;
        auto value = trim(line.substr(eq + 1));
        if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') && value.back() == value.front())
            value = value.substr(1, value.size() - 2);
        return value;
    }
    return "";
}
}

Mitsuba3Loader::Mitsuba3Loader(const std::string& scene_name) {
    const std::string envFolder = scenesFolderFromEnvFile();
    const std::string scenesFolder = !envFolder.empty() ? envFolder : DEFAULT_SCENES_FOLDER;

    namespace fs = std::filesystem;
    const fs::path scenesPath(scenesFolder);
    if (!fs::is_directory(scenesPath)) {
        std::cerr << "ERROR: scenes folder not found: " << fs::absolute(scenesPath).string() << "\n";
        std::exit(EXIT_FAILURE);
    }
    if (!fs::is_directory(scenesPath / scene_name)) {
        std::cerr << "ERROR: scene '" << scene_name << "' not found in " << fs::absolute(scenesPath).string() << "\n";
        std::exit(EXIT_FAILURE);
    }

    sceneDir = scenesFolder + '/' + scene_name;
    const auto scene_path = sceneDir + "/scene_v3.xml";
    std::cout << "Loading scene from: " << scene_path << std::endl;
    const tinyxml2::XMLError status = sceneDesc.LoadFile(scene_path.c_str());

    if (status != tinyxml2::XMLError::XML_SUCCESS) {
        std::cout << "TinyXML2 error: " << status << '\n';
        std::exit(EXIT_FAILURE);
    }

    world = new World;
    world->cam = new Camera;
}

World *Mitsuba3Loader::load() {
    const auto root = sceneDesc.RootElement();
    for (auto elem = root->FirstChildElement();
         elem;
         elem = elem->NextSiblingElement()) {
        const std::string name = elem->Name();

        if (name == "default") {
            memoizeDefaultValue(elem);
        } else if (name == "integrator") {
            loadIntegrator(elem);
        } else if (name == "sensor") {
            loadSensor(elem);
        } else if (name == "bsdf") {
            loadMaterial(elem);
        } else if (name == "shape") {
            loadShape(elem);
        } else if (name == "emitter") {
            loadLight(elem);
        } else {
            std::cerr << "WARNING: Skipping unknown scene element <" << name << ">" << std::endl;
        }
    }

    // Renderer-only properties piggyback on Mitsuba's <default> block (see header).
    if (const auto it = defaultValues.find("diffuse_scattered"); it != defaultValues.end()) {
        world->cam->image.num_diffuse_scattered = resolveValue<int>(it->second);
    } else {
        std::cerr << "WARNING: scene missing <default name=\"diffuse_scattered\">; defaulting to 8" << std::endl;
        world->cam->image.num_diffuse_scattered = 8;
    }

    // Artistic gain on the final-gather (indirect) term. 1.0 = physically correct;
    // raise it to exaggerate colour bleeding on scenes where it is geometrically faint.
    if (const auto it = defaultValues.find("indirect_intensity"); it != defaultValues.end()) {
        world->cam->image.indirect_intensity = resolveValue<float>(it->second);
    } else {
        world->cam->image.indirect_intensity = 1.0f;
    }

    if (const auto it = defaultValues.find("caustic_intensity"); it != defaultValues.end()) {
        world->cam->image.caustic_intensity = resolveValue<float>(it->second);
    } else {
        world->cam->image.caustic_intensity = 1.0f;
    }

    if (const auto it = defaultValues.find("exposure"); it != defaultValues.end()) {
        world->cam->image.exposure = resolveValue<float>(it->second);
    }
    if (const auto it = defaultValues.find("denoise"); it != defaultValues.end()) {
        world->cam->image.denoise = it->second == "true";
    }
    if (const auto it = defaultValues.find("tonemap"); it != defaultValues.end()) {
        if (it->second == "filmic") world->cam->image.tonemap = TONEMAP_FILMIC;
        else if (it->second == "hable") world->cam->image.tonemap = TONEMAP_HABLE;
        else std::cerr << "WARNING: unknown tonemap '" << it->second << "'; using hable" << std::endl;
    }
    if (const auto it = defaultValues.find("global_gather_radius"); it != defaultValues.end()) {
        world->cam->image.global_gather_radius = resolveValue<float>(it->second);
    }
    if (const auto it = defaultValues.find("area_light_samples"); it != defaultValues.end()) {
        world->cam->image.area_light_samples = resolveValue<int>(it->second);
    }

    if (const auto it = defaultValues.find("casted_diffuse_photons"); it != defaultValues.end()) {
        world->casted_diffuse_photons = resolveValue<int>(it->second);
    }
    if (const auto it = defaultValues.find("photon_max_depth"); it != defaultValues.end()) {
        world->photon_max_depth = resolveValue<int>(it->second);
    }
    if (const auto it = defaultValues.find("casted_caustic_photons"); it != defaultValues.end()) {
        world->casted_caustic_photons = resolveValue<int>(it->second);
    }

    if (const auto it = defaultValues.find("sky_colour"); it != defaultValues.end()) {
        world->sky_colour = parseVec3f(it->second);
    }

    return world;
}

void Mitsuba3Loader::loadLight(const tinyxml2::XMLElement *light) {
    const std::string emitter_type = light->Attribute("type") ? light->Attribute("type") : "";

    auto findChild = [light](const char *elem, const char *nm) -> const tinyxml2::XMLElement* {
        for (auto c = light->FirstChildElement(elem); c; c = c->NextSiblingElement(elem))
            if (c->Attribute("name") && std::string(nm) == c->Attribute("name")) return c;
        return nullptr;
    };

    auto *l = new Light;
    l->direction = owl::vec3f(0.f);
    l->position = owl::vec3f(0.f);
    l->cos_inner = l->cos_outer = 0.f;

    if (emitter_type == "point") {
        const auto *intensity_elem = findChild("rgb", "intensity");
        const auto *position_elem = light->FirstChildElement("point");
        if (!intensity_elem || !position_elem) {
            std::cerr << "ERROR: point emitter missing intensity or position" << std::endl;
            std::abort();
        }
        l->type = LIGHT_POINT;
        l->power = parseVec3f(intensity_elem->Attribute("value"));
        l->position = owl::vec3f(
            resolveValue<float>(position_elem->Attribute("x")),
            resolveValue<float>(position_elem->Attribute("y")),
            resolveValue<float>(position_elem->Attribute("z")));
    } else if (emitter_type == "spot") {
        const auto *intensity_elem = findChild("rgb", "intensity");
        const auto *transform_elem = light->FirstChildElement("transform");
        if (!intensity_elem || !transform_elem) {
            std::cerr << "ERROR: spot emitter missing intensity or transform" << std::endl;
            std::abort();
        }
        const Mat4f tf = load_transform(transform_elem);
        l->type = LIGHT_SPOT;
        l->power = parseVec3f(intensity_elem->Attribute("value"));
        l->position = owl::vec3f(tf * owl::vec4f(0, 0, 0, 1));
        l->direction = normalize(owl::vec3f(tf * owl::vec4f(0, 0, 1, 0)));
        const auto *cutoff = findChild("float", "cutoff_angle");
        const auto *beam   = findChild("float", "beam_width");
        const float cutoff_deg = cutoff ? resolveValue<float>(cutoff->Attribute("value")) : 20.f;
        const float beam_deg   = beam   ? resolveValue<float>(beam->Attribute("value"))   : cutoff_deg * 0.75f;
        constexpr float DEG2RAD = 0.01745329252f;
        l->cos_outer = std::cos(cutoff_deg * DEG2RAD);
        l->cos_inner = std::cos(beam_deg * DEG2RAD);
    } else if (emitter_type == "directional") {
        // Mitsuba directional: `direction` (propagation dir) + `irradiance`.
        const auto *irr_elem = findChild("rgb", "irradiance");
        const auto *dir_elem = findChild("vector", "direction");
        if (!irr_elem || !dir_elem) {
            std::cerr << "ERROR: directional emitter missing irradiance or direction" << std::endl;
            std::abort();
        }
        l->type = LIGHT_DIRECTIONAL;
        l->power = parseVec3f(irr_elem->Attribute("value"));
        l->direction = normalize(parseVec3f(dir_elem->Attribute("value")));
    } else {
        std::cerr << "WARNING: emitter type '" << emitter_type
                  << "' not supported; skipping" << std::endl;
        delete l;
        return;
    }
    world->lights.push_back(l);
}

static const tinyxml2::XMLElement *find_named_child(
    const tinyxml2::XMLElement *parent, const char *element_name, const std::string &prop_name) {
    for (auto e = parent->FirstChildElement(element_name); e; e = e->NextSiblingElement(element_name)) {
        if (e->Attribute("name") && prop_name == e->Attribute("name")) return e;
    }
    return nullptr;
}

// Reads a colour-ish property that may be given as <rgb> or <float>.
static bool read_spectrum(const tinyxml2::XMLElement *parent, const char *prop, owl::vec3f &out) {
    if (const auto rgb = find_named_child(parent, "rgb", prop)) {
        out = parseVec3f(rgb->Attribute("value"));
        return true;
    }
    if (const auto f = find_named_child(parent, "float", prop)) {
        out = owl::vec3f(std::stof(f->Attribute("value")));
        return true;
    }
    return false;
}

Material *Mitsuba3Loader::resolveSubmeshMaterial(
    const ObjSubmesh &sub, const tinyxml2::XMLElement *shape) {
    if (!sub.usemtl_name.empty()) {
        if (const auto it = materials.find(sub.usemtl_name); it != materials.end()) {
            return it->second;
        }
    }
    if (sub.mtl_material) return sub.mtl_material;
    if (const auto ref = shape->FirstChildElement("ref")) {
        if (const char *id = ref->Attribute("id")) {
            if (const auto it = materials.find(id); it != materials.end()) {
                return it->second;
            }
        }
    }
    std::cerr << "WARNING: no material resolved for submesh '" << sub.usemtl_name
              << "'; using default Lambertian" << std::endl;
    auto *fallback = new Material;
    fallback->matType = LAMBERTIAN;
    fallback->albedo = owl::vec3f(0.8f);
    fallback->diffuse = 1.f;
    fallback->specular = 0.f;
    fallback->ior = 0.f;
    return fallback;
}

void Mitsuba3Loader::loadShape(const tinyxml2::XMLElement *shape) {
    const std::string type = shape->Attribute("type");
    const auto tf = load_transform(shape->FirstChildElement("transform"));
    const std::string shape_id = shape->Attribute("id") ? shape->Attribute("id") : "<unnamed>";

    // Material: an inline <bsdf> takes precedence over a <ref>.
    Material *shape_material = nullptr;
    if (const auto inline_bsdf = shape->FirstChildElement("bsdf")) {
        shape_material = parseMaterial(inline_bsdf, shape_id);
    } else if (const auto ref = shape->FirstChildElement("ref")) {
        if (const char *id = ref->Attribute("id")) {
            if (const auto it = materials.find(id); it != materials.end()) shape_material = it->second;
        }
    }

    owl::vec3f emitted_radiance(0.f);
    if (const auto emitter = shape->FirstChildElement("emitter")) {
        const std::string emitter_type = emitter->Attribute("type") ? emitter->Attribute("type") : "";
        if (emitter_type == "area") {
            if (!read_spectrum(emitter, "radiance", emitted_radiance)) {
                std::cerr << "WARNING: area emitter on shape '" << shape_id << "' has no radiance" << std::endl;
            }
        } else {
            std::cerr << "WARNING: unsupported emitter type '" << emitter_type
                      << "' inside shape; skipping" << std::endl;
        }
    }
    const bool is_emitter = emitted_radiance.x > 0.f || emitted_radiance.y > 0.f || emitted_radiance.z > 0.f;
    if (is_emitter) {
        // Emitters get their own material copy so the emission doesn't leak to other shapes.
        auto *m = new Material(shape_material ? *shape_material : Material{});
        if (!shape_material) {
            m->matType = LAMBERTIAN;
            m->albedo = owl::vec3f(0.f);
            m->diffuse = 1.f;
            m->specular = 0.f;
            m->ior = 0.f;
        }
        m->emission = emitted_radiance;
        shape_material = m;

        if (type == "rectangle") {
            // Sampled area light. Mitsuba rectangles span [-1,1]^2 in local xy, emitting along +z
            // (normals transform with the inverse transpose, i.e. cross(edge_u, edge_v) / det).
            auto *l = new Light;
            l->type = LIGHT_AREA;
            l->position = owl::vec3f(tf * owl::vec4f(0, 0, 0, 1));
            l->edge_u = owl::vec3f(tf * owl::vec4f(1, 0, 0, 0));
            l->edge_v = owl::vec3f(tf * owl::vec4f(0, 1, 0, 0));
            const owl::vec3f edge_w = owl::vec3f(tf * owl::vec4f(0, 0, 1, 0));
            const owl::vec3f n = cross(l->edge_u, l->edge_v);
            l->direction = normalize(dot(n, edge_w) < 0.f ? -n : n);
            l->area = 4.f * length(n);
            l->power = emitted_radiance;
            l->cos_inner = l->cos_outer = 0.f;
            world->lights.push_back(l);
            m->emitter_area = l->area;
        } else {
            std::cerr << "WARNING: area emitter on '" << type << "' shape '" << shape_id
                      << "' is visible but not sampled as a light (only rectangles are)" << std::endl;
        }
    }

    if (type == "rectangle" || type == "cube") {
        Mesh *mesh = (type == "rectangle") ? Mesh::makeBaseRectangle() : Mesh::makeBaseCube();
        mesh->applyTransform(tf);
        if (!shape_material) {
            std::cerr << "ERROR: shape '" << shape_id << "' of type '" << type
                      << "' has no material" << std::endl;
            std::abort();
        }
        auto *model = new Model{ mesh, shape_material, xmlTexturePath(shape_material), xmlBumpPath(shape_material) };
        world->models.emplace_back(model);
        return;
    }

    if (type != "obj") {
        std::cerr << "ERROR: Unknown shape type: " << type << std::endl;
        std::abort();
    }

    std::string obj_file_path = sceneDir + "/";
    if (const auto file_name_elem = find_named_child(shape, "string", "filename")) {
        obj_file_path += file_name_elem->Attribute("value");
    }

    bool faceted = false;
    if (const auto faceted_elem = find_named_child(shape, "boolean", "face_normals")) {
        faceted = std::string(faceted_elem->Attribute("value")) == "true";
    }

    bool load_material_files = false;
    if (const auto it = defaultValues.find("load_material_files"); it != defaultValues.end()) {
        load_material_files = it->second == "true";
    }
    if (const auto lmf_elem = find_named_child(shape, "boolean", "load_material_files")) {
        load_material_files = std::string(lmf_elem->Attribute("value")) == "true";
    }

    auto submeshes = load_obj_submeshes(obj_file_path, faceted, load_material_files);
    const bool shape_overrides = is_emitter || shape->FirstChildElement("bsdf");
    for (auto &sub : submeshes) {
        sub.mesh->applyTransform(tf);
        Material *material = shape_overrides ? shape_material : resolveSubmeshMaterial(sub, shape);
        const std::string &tex = sub.albedo_texture_path.empty()
            ? xmlTexturePath(material) : sub.albedo_texture_path;
        world->models.emplace_back(new Model{ sub.mesh, material, tex, xmlBumpPath(material) });
    }
}

const std::string &Mitsuba3Loader::xmlBumpPath(const Material *material) const {
    static const std::string kNoTexture;
    const auto it = materialBumpTextures.find(material);
    return it != materialBumpTextures.end() ? it->second : kNoTexture;
}

const std::string &Mitsuba3Loader::xmlTexturePath(const Material *material) const {
    static const std::string kNoTexture;
    const auto it = materialTextures.find(material);
    return it != materialTextures.end() ? it->second : kNoTexture;
}

float Mitsuba3Loader::getDiffuseCoeff(const Material* mat, const tinyxml2::XMLElement *inner_bsdf=nullptr) {
    if (mat->matType == LAMBERTIAN) {
        return 1.f;
    }
    if (mat->matType == DIELECTRIC) {
        return 0.f;
    }
    if (mat->matType == CONDUCTOR) {
        return 0.f;
    }
    std::cerr << "ERROR: Unknown material type in getDiffuseCoeff" << std::endl;
    std::abort();
}

float Mitsuba3Loader::getSpecularCoeff(const Material* mat, const tinyxml2::XMLElement *inner_bsdf=nullptr) {
    if (mat->matType == LAMBERTIAN) {
        return 0.f;
    }
    if (mat->matType == DIELECTRIC) {
        return 0.f;
    }
    if (mat->matType == CONDUCTOR) {
      if (const auto specular = find_named_child(inner_bsdf, "float", "specular")) {
        return resolveValue<float>(specular->Attribute("value"));
      }
      return 1.f;
    }
    std::cerr << "ERROR: Unknown material type in getSpecularCoeff" << std::endl;
    std::abort();
}

float Mitsuba3Loader::getTransmissionCoeff(const Material* mat, const tinyxml2::XMLElement *inner_bsdf=nullptr) {
    if (mat->matType == LAMBERTIAN) {
        return 0.f;
    }
    if (mat->matType == DIELECTRIC) {
        // We assume ext_ior to always be 1.0
        if (const auto ior = find_named_child(inner_bsdf, "float", "int_ior")) {
            return resolveValue<float>(ior->Attribute("value"));
        }
        return 1.5f;   // Mitsuba's dielectric default (bk7 glass)
    }
    if (mat->matType == CONDUCTOR) {
        return 0.f;
    }
    std::cerr << "ERROR: Unknown material type in getIor" << std::endl;
    std::abort();
}

// Exact unpolarised Fresnel reflectance; eta = n_transmitted / n_incident.
static float fresnel_dielectric_host(const float cos_i, const float eta) {
    const float sin2_t = (1.f - cos_i * cos_i) / (eta * eta);
    if (sin2_t >= 1.f) return 1.f;
    const float cos_t = std::sqrt(1.f - sin2_t);
    const float rs = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    const float rp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    return 0.5f * (rs * rs + rp * rp);
}

// Cosine-weighted hemispherical average of the Fresnel reflectance.
static float fresnel_diffuse_reflectance(const float eta) {
    constexpr int N = 2048;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        const float mu2 = (float(i) + 0.5f) / float(N); // uniform in cos^2 => cosine-weighted
        sum += fresnel_dielectric_host(std::sqrt(mu2), eta);
    }
    return float(sum / N);
}

static const tinyxml2::XMLElement *unwrap_twosided(const tinyxml2::XMLElement *bsdf) {
    while (bsdf) {
        const auto type = bsdf->Attribute("type");
        if (!type || (std::string(type) != "twosided" && std::string(type) != "mask")) break;
        const auto inner = bsdf->FirstChildElement("bsdf");
        if (!inner) break;
        bsdf = inner;
    }
    return bsdf;
}

owl::vec3f Mitsuba3Loader::conductorAlbedo(const tinyxml2::XMLElement *inner) {
    owl::vec3f albedo(1.f);
    if (const auto refl = find_named_child(inner, "rgb", "specular_reflectance")) {
        albedo = parseVec3f(refl->Attribute("value"));
    }
    const auto eta_elem = find_named_child(inner, "rgb", "eta");
    const auto k_elem = find_named_child(inner, "rgb", "k");
    if (eta_elem && k_elem) {
        const owl::vec3f eta = parseVec3f(eta_elem->Attribute("value"));
        const owl::vec3f k = parseVec3f(k_elem->Attribute("value"));
        const auto f0 = [](const float e, const float kk) {
            return ((e - 1.f) * (e - 1.f) + kk * kk) / ((e + 1.f) * (e + 1.f) + kk * kk);
        };
        albedo *= owl::vec3f(f0(eta.x, k.x), f0(eta.y, k.y), f0(eta.z, k.z));
    }
    return albedo;
}

void Mitsuba3Loader::loadReflectance(Material *material, const std::string &name,
                                     const tinyxml2::XMLElement *inner, const char *prop) {
    if (const auto rgb = find_named_child(inner, "rgb", prop)) {
        material->albedo = parseVec3f(rgb->Attribute("value"));
        return;
    }
    material->albedo = owl::vec3f(0.8f);
    if (const auto texture = find_named_child(inner, "texture", prop)) {
        if (const auto file_elem = find_named_child(texture, "string", "filename")) {
            materialTextures.emplace(material, sceneDir + "/" + file_elem->Attribute("value"));
        } else {
            std::cerr << "WARNING: BSDF '" << name << "' has a texture " << prop
                      << " without a filename" << std::endl;
        }
    } else {
        std::cerr << "WARNING: BSDF '" << name << "' has no " << prop
                  << "; using default Lambertian gray" << std::endl;
    }
}

void Mitsuba3Loader::loadMaterial(const tinyxml2::XMLElement *bsdf) {
    if (const auto type_attr = bsdf->Attribute("type"); type_attr && std::string(type_attr) == "bumpmap") {
        if (const auto wrapped = bsdf->FirstChildElement("bsdf")) {
            loadMaterial(wrapped);
            const char *wid = wrapped->Attribute("id");
            const auto it = wid ? materials.find(wid) : materials.end();
            const auto map = find_named_child(bsdf, "texture", "map");
            const auto file = map ? find_named_child(map, "string", "filename") : nullptr;
            if (it != materials.end() && file) {
                materialBumpTextures[it->second] = sceneDir + "/" + file->Attribute("value");
                if (const auto scale = find_named_child(bsdf, "float", "scale"))
                    it->second->bump_scale = resolveValue<float>(scale->Attribute("value"));
            } else {
                std::cerr << "WARNING: bumpmap without a usable bitmap 'map' dropped" << std::endl;
            }
        }
        return;
    }
    if (!bsdf->Attribute("id")) {
        std::cerr << "WARNING: skipping <bsdf type=\"" << (bsdf->Attribute("type") ? bsdf->Attribute("type") : "?")
                  << "\"> without id; it cannot be referenced" << std::endl;
        return;
    }
    const std::string name = bsdf->Attribute("id");
    materials.emplace(name, parseMaterial(bsdf, name));
}

Material *Mitsuba3Loader::parseMaterial(const tinyxml2::XMLElement *bsdf, const std::string &name) {
    // Mask: remember the opacity, then parse the wrapped BSDF as usual.
    float opacity = 1.f;
    for (auto b = bsdf; b; b = b->FirstChildElement("bsdf")) {
        const auto t = b->Attribute("type");
        if (!t || (std::string(t) != "twosided" && std::string(t) != "mask")) break;
        if (std::string(t) == "mask") {
            owl::vec3f op(1.f);
            if (read_spectrum(b, "opacity", op)) opacity = (op.x + op.y + op.z) / 3.f;
        }
    }

    const auto inner = unwrap_twosided(bsdf);
    auto material = new Material;
    material->opacity = opacity;

    const std::string type = inner->Attribute("type");

    if (type == "diffuse") {
        material->matType = LAMBERTIAN;
        loadReflectance(material, name, inner, "reflectance");
        material->diffuse = getDiffuseCoeff(material, inner);
        material->specular = getSpecularCoeff(material, inner);
        material->ior = getTransmissionCoeff(material, inner);
    } else if (type == "roughplastic" || type == "plastic") {
        // Diffuse base under a dielectric clear coat. The coat is handled as a
        // Fresnel-weighted (optionally fuzzed) mirror lobe on top of the Lambertian.
        material->matType = LAMBERTIAN;
        loadReflectance(material, name, inner, "diffuse_reflectance");
        material->diffuse = 1.f;
        material->specular = 0.f;
        material->ior = 0.f;
        float int_ior = 1.49f; // Mitsuba default (polypropylene)
        if (const auto e = find_named_child(inner, "float", "int_ior")) int_ior = resolveValue<float>(e->Attribute("value"));
        float ext_ior = 1.f;
        if (const auto e = find_named_child(inner, "float", "ext_ior")) ext_ior = resolveValue<float>(e->Attribute("value"));
        const float eta = int_ior / ext_ior;
        material->coat_r0 = ((eta - 1.f) / (eta + 1.f)) * ((eta - 1.f) / (eta + 1.f));
        // Light enters through the coat (avg. transmission 1 - Fdr_ext), bounces between the
        // base and the coat (1 / (1 - rho * Fdr_int)) and is spread over a smaller solid angle
        // on exit (1 / eta^2). The exit-side (1 - F) is applied per view direction by the tracer.
        material->coat_diffuse_scale = (1.f - fresnel_diffuse_reflectance(eta)) / (eta * eta);
        material->coat_fdr_int = fresnel_diffuse_reflectance(1.f / eta);
        material->coat_nonlinear = true; // Tungsten's behaviour; Mitsuba defaults to false
        if (const auto nl = find_named_child(inner, "boolean", "nonlinear"))
            material->coat_nonlinear = std::string(nl->Attribute("value")) == "true";
        if (type == "roughplastic") {
            material->coat_roughness = 0.1f; // Mitsuba default alpha
            if (const auto a = find_named_child(inner, "float", "alpha")) material->coat_roughness = resolveValue<float>(a->Attribute("value"));
        }
    } else if (type == "dielectric" || type == "thindielectric") {
        material->matType = DIELECTRIC;
        material->albedo = 1.f;
        material->diffuse = getDiffuseCoeff(material, inner);
        material->specular = getSpecularCoeff(material, inner);
        material->ior = getTransmissionCoeff(material, inner);
        material->thin = (type == "thindielectric");
        owl::vec3f trans(1.f);
        if (read_spectrum(inner, "specular_transmittance", trans)) {
            material->transmittance = (trans.x + trans.y + trans.z) / 3.f;
        }
    } else if (type == "conductor") {
        material->matType = CONDUCTOR;
        material->albedo = conductorAlbedo(inner);
        material->diffuse = getDiffuseCoeff(material, inner);
        material->ior = getTransmissionCoeff(material, inner);
        material->specular = getSpecularCoeff(material, inner);
    } else if (type == "roughconductor") {
        material->matType = CONDUCTOR;
        material->albedo = conductorAlbedo(inner);
        material->diffuse = 0.f;
        material->ior = 0.f;
        material->specular = 1.f;
        if (const auto alpha = find_named_child(inner, "float", "alpha")) {
            material->roughness = resolveValue<float>(alpha->Attribute("value"));
        }
    } else {
        std::cerr << "WARNING: BSDF '" << name << "' uses unsupported type '" << type
                  << "'; using default Lambertian" << std::endl;
        material->matType = LAMBERTIAN;
        material->albedo = owl::vec3f(0.8f);
        material->diffuse = 1.f;
        material->specular = 0.f;
        material->ior = 0.f;
    }

    return material;
}


void Mitsuba3Loader::loadSensor(const tinyxml2::XMLElement *sensor) {
    for (auto elem = sensor->FirstChildElement();
         elem;
         elem = elem->NextSiblingElement())
    {
        // Handle properties straight inside the `sensor` element.
        const auto name_cstr = elem->Attribute("name");
        if (std::string name = (name_cstr != nullptr) ? name_cstr : ""; name == "fov") {
            world->cam->image.fov = resolveValue<float>(elem->Attribute("value"));
        } else if (name == "to_world") {
            const auto tf = load_transform(elem);
            world->cam->lookFrom = owl::vec3f(tf * owl::vec4f(0, 0, 0, 1));
            world->cam->up = owl::vec3f(tf * owl::vec4f(0, 1, 0, 0));
            const auto forward = owl::vec3f(tf * owl::vec4f(0, 0, 1, 0));
            world->cam->lookAt = world->cam->lookFrom + forward;
        }

        // Handle sampler and film
        std::string elem_name = elem->Name();
        if (elem_name == "sampler") {
            for (auto child = elem->FirstChildElement("integer"); child;
                 child = child->NextSiblingElement("integer")) {
                if (child->Attribute("name") && !strcmp(child->Attribute("name"), "sample_count")) {
                    world->cam->image.pixel_samples = resolveValue<int>(child->Attribute("value"));
                }
            }
            // diffuse_scattered is non-Mitsuba; read from <extras> after the main parse instead.
            continue;
        }

        if (elem_name == "film") {
            for (auto film_elem = elem->FirstChildElement("integer");
                film_elem;
                film_elem = film_elem->NextSiblingElement("integer")) {
                if (!strcmp(film_elem->Attribute("name"), "width")) {
                    world->cam->image.resolution.x = resolveValue<int>(film_elem->Attribute("value"));
                    continue;
                }

                if (!strcmp(film_elem->Attribute("name"), "height")) {
                    world->cam->image.resolution.y = resolveValue<int>(film_elem->Attribute("value"));
                }
            }
        }
    }

    // The viewer takes a vertical FOV; Mitsuba's `fov` is measured along `fov_axis`
    // (default "x"). Convert so the framing matches Mitsuba for any aspect ratio.
    std::string axis = "x";
    if (const auto axis_elem = find_named_child(sensor, "string", "fov_axis"))
        axis = axis_elem->Attribute("value");
    const owl::vec2i res = world->cam->image.resolution;
    const float aspect = (res.y > 0) ? float(res.x) / float(res.y) : 1.f;
    constexpr float DEG2RAD = 0.01745329252f;
    float tan_half = std::tan(0.5f * world->cam->image.fov * DEG2RAD);
    if (axis == "x" || (axis == "smaller" && aspect < 1.f) || (axis == "larger" && aspect > 1.f)) {
        tan_half /= aspect;
    } else if (axis == "diagonal") {
        tan_half /= std::sqrt(aspect * aspect + 1.f);
    } else if (axis != "y" && axis != "smaller" && axis != "larger") {
        std::cerr << "WARNING: unknown fov_axis '" << axis << "'; treating fov as vertical" << std::endl;
    }
    world->cam->image.fov = 2.f * std::atan(tan_half) / DEG2RAD;
}

Mat4f load_transform(const tinyxml2::XMLElement* transform) {
    if (const auto lookat = transform->FirstChildElement("lookat")) {
        const owl::vec3f origin = parseVec3f(lookat->Attribute("origin"));
        const owl::vec3f target = parseVec3f(lookat->Attribute("target"));
        const owl::vec3f up     = parseVec3f(lookat->Attribute("up"));

        const owl::vec3f dir    = normalize(target - origin);
        const owl::vec3f left   = normalize(cross(up, dir));
        const owl::vec3f new_up = cross(dir, left);

        return Mat4f(std::array<float, 16>{
            left.x, new_up.x, dir.x, origin.x,
            left.y, new_up.y, dir.y, origin.y,
            left.z, new_up.z, dir.z, origin.z,
            0.f,    0.f,      0.f,   1.f
        });
    }

    const auto matrix_element = transform->FirstChildElement("matrix");
    return Mat4f(matrix_element->Attribute("value"));
}

void Mitsuba3Loader::loadIntegrator(const tinyxml2::XMLElement *integrator) {
    for (auto elem = integrator->FirstChildElement("integer");
         elem;
         elem = elem->NextSiblingElement("integer"))
    {
        if (strcmp("max_depth", elem->Attribute("name")) != 0) {
            continue;
        }
        world->cam->image.depth = resolveValue<int>(elem->Attribute("value"));
    }
}

void Mitsuba3Loader::memoizeDefaultValue(const tinyxml2::XMLElement* defaultElem) {
    std::string key = defaultElem->Attribute("name");
    std::string value = defaultElem->Attribute("value");
    defaultValues.emplace(key, value);
}
