#include "mitsuba3.cuh"
#include "obj.cuh"
#include "../world.cuh"

Mitsuba3Loader::Mitsuba3Loader(const std::string& scene_name) {
    sceneDir = scenesFolder + '\\' + scene_name;
    auto scene_path = sceneDir + "\\scene_v3.xml";
#ifdef __linux__
    std::ranges::replace(scene_path, '\\', '/');
#endif
    std::cout << "Loading scene from: " << scene_path << std::endl;
    sceneDesc.LoadFile(scene_path.c_str());

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

    // Artistic gain on the caustic term. 1.0 = physically correct; raise it to make
    // sparse/faint caustics pop without re-emitting more caustic photons.
    if (const auto it = defaultValues.find("caustic_intensity"); it != defaultValues.end()) {
        world->cam->image.caustic_intensity = resolveValue<float>(it->second);
    } else {
        world->cam->image.caustic_intensity = 1.0f;
    }

    // Photon-emitter budget (emitter-only; the path tracer ignores these). Keep the
    // World defaults when the scene omits them.
    if (const auto it = defaultValues.find("casted_diffuse_photons"); it != defaultValues.end()) {
        world->casted_diffuse_photons = resolveValue<int>(it->second);
    }
    if (const auto it = defaultValues.find("casted_caustic_photons"); it != defaultValues.end()) {
        world->casted_caustic_photons = resolveValue<int>(it->second);
    }
    if (const auto it = defaultValues.find("casted_volume_photons"); it != defaultValues.end()) {
        world->casted_volume_photons = resolveValue<int>(it->second);
    }

    // Global homogeneous medium (renderer-only, Mitsuba-inert defaults). sigma_t <= 0 = vacuum.
    if (const auto it = defaultValues.find("medium_sigma_t"); it != defaultValues.end()) {
        world->medium.sigma_t = resolveValue<float>(it->second);
    }
    if (const auto it = defaultValues.find("medium_albedo"); it != defaultValues.end()) {
        world->medium.albedo = parseVec3f(it->second);
    }
    if (const auto it = defaultValues.find("medium_g"); it != defaultValues.end()) {
        world->medium.g = resolveValue<float>(it->second);
    }

    // Scene bounding sphere (volume-march cap; also mirrors the emitter's own bounds).
    owl::vec3f lo(1e30f), hi(-1e30f);
    for (const auto* m : world->models)
        for (const auto& v : m->mesh->vertices) {
            lo.x = fminf(lo.x, v.x); lo.y = fminf(lo.y, v.y); lo.z = fminf(lo.z, v.z);
            hi.x = fmaxf(hi.x, v.x); hi.y = fmaxf(hi.y, v.y); hi.z = fmaxf(hi.z, v.z);
        }
    if (hi.x >= lo.x) {
        world->scene_center = 0.5f * (lo + hi);
        world->scene_radius = 0.5f * length(hi - lo);
    }
    if (world->scene_radius <= 0.f) world->scene_radius = 1.f;

    return world;
}

void Mitsuba3Loader::loadLight(const tinyxml2::XMLElement *light) {
    const std::string emitter_type = light->Attribute("type") ? light->Attribute("type") : "";

    // Find a child by element name + `name` attribute (e.g. <float name="cutoff_angle">).
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
        // Mitsuba spot: positioned + oriented by to_world (emits along local +Z), with
        // intensity, cutoff_angle (outer half-angle, deg) and beam_width (inner, deg).
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

// Look up an `<{element_name} name="{prop_name}" value="..."/>` child by name attribute.
static const tinyxml2::XMLElement *find_named_child(
    const tinyxml2::XMLElement *parent, const char *element_name, const std::string &prop_name) {
    for (auto e = parent->FirstChildElement(element_name); e; e = e->NextSiblingElement(element_name)) {
        if (e->Attribute("name") && prop_name == e->Attribute("name")) return e;
    }
    return nullptr;
}

// Resolve the material for one submesh of an OBJ. Priority:
//   1. XML <bsdf> whose id matches the submesh's usemtl name
//   2. .mtl-derived material attached to the submesh
//   3. The shape's <ref id="..."> fallback, if any
//   4. Default Lambertian gray with a warning
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
                      << "' skipped; area lights land in Phase 2 (shape kept, light dropped)" << std::endl;
        } else {
            std::cerr << "WARNING: unsupported emitter type '" << emitter_type
                      << "' inside shape; skipping" << std::endl;
        }
    }

    if (type == "rectangle" || type == "cube") {
        Mesh *mesh = (type == "rectangle") ? Mesh::makeBaseRectangle() : Mesh::makeBaseCube();
        mesh->applyTransform(tf);
        const std::string mat_id = shape->FirstChildElement("ref")->Attribute("id");
        auto *model = new Model{ mesh, materials.at(mat_id) };
        world->models.emplace_back(model);
        return;
    }

    if (type != "obj") {
        std::cerr << "ERROR: Unknown shape type: " << type << std::endl;
        std::abort();
    }

    std::string obj_file_path = sceneDir + "\\";
    if (const auto file_name_elem = find_named_child(shape, "string", "filename")) {
        obj_file_path += file_name_elem->Attribute("value");
    }

    bool faceted = false;
    if (const auto faceted_elem = find_named_child(shape, "boolean", "face_normals")) {
        faceted = std::string(faceted_elem->Attribute("value")) == "true";
    }

    // Per-shape override beats scene-level default. The default lives in <default>
    // (not a per-shape <boolean>) because Mitsuba 3.8's `obj` plugin rejects
    // unrecognized boolean properties — defaults are inert and dual-load-safe.
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
        world->models.emplace_back(new Model{ sub.mesh, material, sub.albedo_texture_path });
    }
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
      const auto specular = inner_bsdf->FirstChildElement("float");
      assert(std::string(specular->Attribute("name")) == "specular");
      return resolveValue<float>(specular->Attribute("value"));
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
        const auto ior = inner_bsdf->FirstChildElement("float");
        assert(std::string(ior->Attribute("name")) == "int_ior");
        return resolveValue<float>(ior->Attribute("value"));
    }
    if (mat->matType == CONDUCTOR) {
        return 0.f;
    }
    std::cerr << "ERROR: Unknown material type in getIor" << std::endl;
    std::abort();
}

// Returns the effective leaf BSDF element to read material properties from.
// Mitsuba's `twosided` is a modifier that wraps a real BSDF; everything else
// (diffuse, dielectric, conductor, roughconductor) is itself a leaf.
static const tinyxml2::XMLElement *unwrap_twosided(const tinyxml2::XMLElement *bsdf) {
    if (const auto type = bsdf->Attribute("type"); type && std::string(type) == "twosided") {
        return bsdf->FirstChildElement("bsdf");
    }
    return bsdf;
}

void Mitsuba3Loader::loadMaterial(const tinyxml2::XMLElement *bsdf) {
    const auto inner = unwrap_twosided(bsdf);
    auto material = new Material;

    std::string name = bsdf->Attribute("id");
    const std::string type = inner->Attribute("type");

    if (type == "diffuse") {
        material->matType = LAMBERTIAN;
        const auto reflectance = inner->FirstChildElement("rgb");
        assert(std::string(reflectance->Attribute("name")) == "reflectance");
        material->albedo = parseVec3f(reflectance->Attribute("value"));
        material->diffuse = getDiffuseCoeff(material, inner);
        material->specular = getSpecularCoeff(material, inner);
        material->ior = getTransmissionCoeff(material, inner);
    } else if (type == "dielectric") {
        material->matType = DIELECTRIC;
        material->albedo = 1.f;
        material->diffuse = getDiffuseCoeff(material, inner);
        material->specular = getSpecularCoeff(material, inner);
        material->ior = getTransmissionCoeff(material, inner);
    } else if (type == "conductor") {
        material->matType = CONDUCTOR;
        material->albedo = 1.f;
        material->diffuse = getDiffuseCoeff(material, inner);
        material->ior = getTransmissionCoeff(material, inner);
        material->specular = getSpecularCoeff(material, inner);
    } else if (type == "roughconductor") {
        // Collapse to smooth CONDUCTOR; alpha (roughness) is dropped until a microfacet pass exists.
        material->matType = CONDUCTOR;
        material->albedo = 1.f;
        material->diffuse = 0.f;
        material->ior = 0.f;
        material->specular = 1.f;
        const char *alpha_str = nullptr;
        for (auto f = inner->FirstChildElement("float"); f; f = f->NextSiblingElement("float")) {
            if (f->Attribute("name") && std::string(f->Attribute("name")) == "alpha") {
                alpha_str = f->Attribute("value");
                break;
            }
        }
        std::cerr << "WARNING: roughconductor '" << name << "' collapsed to smooth CONDUCTOR; "
                  << "dropped alpha=" << (alpha_str ? alpha_str : "(unset)")
                  << " until microfacet pass lands" << std::endl;
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
    // Mitsuba allows either an explicit <matrix> or a <lookat origin/target/up>.
    if (const auto lookat = transform->FirstChildElement("lookat")) {
        const owl::vec3f origin = parseVec3f(lookat->Attribute("origin"));
        const owl::vec3f target = parseVec3f(lookat->Attribute("target"));
        const owl::vec3f up     = parseVec3f(lookat->Attribute("up"));

        // Mitsuba's look_at convention: camera looks along local +Z toward the
        // target, columns = [left, new_up, dir, origin].
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
