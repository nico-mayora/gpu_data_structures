#include "mitsuba3.h"
#include "../world.h"

Mitsuba3Loader::Mitsuba3Loader(const std::string& scene_name) {
    sceneDir = scenesFolder + '/' + scene_name;
    const auto scene_path = sceneDir + "/scene_v3.xml";
    std::cout << "Loading scene from: " << scene_path << std::endl;
    if (const auto status = sceneDesc.LoadFile(scene_path.c_str()); status != 0) {
        std::cout << "Error loading scene: " << status << "\n";
        return;
    }
    std::cout << "Loaded scene successfully!\n";

    world = new World;
    world->cam = new Camera;
}

World *Mitsuba3Loader::load() {
    auto root = sceneDesc.RootElement();
    for (auto node = root->FirstChild();
         node;
         node = node->NextSibling())
    {
        std::string name = node->Value();

        if (name == "default") {
            memoizeDefaultValue(node->ToElement());
            continue;
        }
        if (name == "integrator") {
            loadIntegrator(node->ToElement());
            continue;
        }
        if (name == "sensor") {
            loadSensor(node->ToElement());
            continue;
        }
        if (name == "bsdf") {
            loadMaterial(node->ToElement());
            continue;
        }
        if (name == "shape") {
            loadShape(node->ToElement());
            continue;
        }

        std::cerr << "ERROR: Unknown element: " << name << std::endl;
        std::abort();
    }

    return world;
}

void Mitsuba3Loader::loadShape(const tinyxml2::XMLElement *shape) {
    const std::string type = shape->Attribute("type");
    Mesh *mesh;
    if (type == "rectangle") {
        mesh = Mesh::makeBaseRectangle();
    } else if (type == "cube") {
        mesh = Mesh::makeBaseCube();
    } else {
        std::cerr << "ERROR: Unknown shape type: " << type << std::endl;
        std::abort();
    }

    const auto tf = load_transform(shape->FirstChildElement("transform"));
    mesh->applyTransform(tf);

    if (const auto emitter = shape->FirstChildElement("emitter")) {
        auto *ls = new LightSource;
        ls->mesh = mesh;
        ls->radiance = parseVec3f(emitter->FirstChildElement("rgb")->Attribute("value"));
        world->light_sources.emplace_back(ls);
        return;
    }

    const std::string mat_id = shape->FirstChildElement("ref")->Attribute("id");
    auto model = new Model;
    model->mesh = mesh;
    model->material = materials.at(mat_id);
    world->models.emplace_back(model);
}

void Mitsuba3Loader::loadMaterial(const tinyxml2::XMLElement *bsdf) {
    const auto inner_bsdf = bsdf->FirstChildElement("bsdf");
    auto material = new Material;

    std::string name = bsdf->Attribute("id");
    std::string type = inner_bsdf->Attribute("type");

    if (type == "diffuse") {
        material->matType = LAMBERTIAN;
        const auto reflectance = inner_bsdf->FirstChildElement("rgb");
        assert(std::string(reflectance->Attribute("name")) == "reflectance");
        material->albedo = parseVec3f(reflectance->Attribute("value"));
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
        if (std::string name = name_cstr ? name_cstr : ""; name == "fov") {
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
            auto child = elem->FirstChildElement("integer");
            assert(!strcmp(child->Attribute("name"), "sample_count"));
            world->cam->image.pixel_samples = resolveValue<int>(child->Attribute("value"));

            auto sibling = child->NextSiblingElement("integer");
            assert(!strcmp(sibling->Attribute("name"), "light_samples"));
            world->cam->image.light_samples = resolveValue<int>(sibling->Attribute("value"));
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
