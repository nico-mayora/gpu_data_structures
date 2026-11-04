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

    if (const auto it = defaultValues.find("casted_diffuse_photons"); it != defaultValues.end()) {
        world->casted_diffuse_photons = resolveValue<int>(it->second);
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

    if (const auto emitter = shape->FirstChildElement("emitter")) {
        const std::string emitter_type = emitter->Attribute("type") ? emitter->Attribute("type") : "";
        if (emitter_type == "area") {
            std::cerr << "WARNING: area emitter on shape '"
                      << (shape->Attribute("id") ? shape->Attribute("id") : "<unnamed>")
                      << "' skipped" << std::endl;
        } else {
            std::cerr << "WARNING: unsupported emitter type '" << emitter_type
                      << "' inside shape; skipping" << std::endl;
        }
    }

    if (type == "rectangle" || type == "cube") {
        Mesh *mesh = (type == "rectangle") ? Mesh::makeBaseRectangle() : Mesh::makeBaseCube();
        mesh->applyTransform(tf);
        const auto ref = shape->FirstChildElement("ref");
        if (!ref) {
            std::cerr << "ERROR: shape '" << (shape->Attribute("id") ? shape->Attribute("id") : "<unnamed>")
                      << "' of type '" << type << "' has no material <ref>" << std::endl;
            std::abort();
        }
        const std::string mat_id = ref->Attribute("id");
        Material *material = materials.at(mat_id);
        auto *model = new Model{ mesh, material, xmlTexturePath(material) };
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
    for (auto &sub : submeshes) {
        sub.mesh->applyTransform(tf);
        Material *material = resolveSubmeshMaterial(sub, shape);
        const std::string &tex = sub.albedo_texture_path.empty()
            ? xmlTexturePath(material) : sub.albedo_texture_path;
        world->models.emplace_back(new Model{ sub.mesh, material, tex });
    }
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
            std::cerr << "WARNING: bumpmap wrapper dropped; loading wrapped BSDF '"
                      << (wrapped->Attribute("id") ? wrapped->Attribute("id") : "<unnamed>")
                      << "' without bump mapping" << std::endl;
            loadMaterial(wrapped);
        }
        return;
    }
    if (!bsdf->Attribute("id")) {
        std::cerr << "WARNING: skipping <bsdf type=\"" << (bsdf->Attribute("type") ? bsdf->Attribute("type") : "?")
                  << "\"> without id; it cannot be referenced" << std::endl;
        return;
    }

    const auto inner = unwrap_twosided(bsdf);
    auto material = new Material;

    std::string name = bsdf->Attribute("id");
    const std::string type = inner->Attribute("type");

    if (type == "diffuse") {
        material->matType = LAMBERTIAN;
        loadReflectance(material, name, inner, "reflectance");
        material->diffuse = getDiffuseCoeff(material, inner);
        material->specular = getSpecularCoeff(material, inner);
        material->ior = getTransmissionCoeff(material, inner);
    } else if (type == "roughplastic" || type == "plastic") {
        material->matType = LAMBERTIAN;
        loadReflectance(material, name, inner, "diffuse_reflectance");
        material->diffuse = 1.f;
        material->specular = 0.f;
        material->ior = 0.f;
        std::cerr << "WARNING: " << type << " '" << name
                  << "' collapsed to LAMBERTIAN; specular coat dropped" << std::endl;
    } else if (type == "dielectric") {
        material->matType = DIELECTRIC;
        material->albedo = 1.f;
        material->diffuse = getDiffuseCoeff(material, inner);
        material->specular = getSpecularCoeff(material, inner);
        material->ior = getTransmissionCoeff(material, inner);
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

    materials.emplace(name, material);
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
