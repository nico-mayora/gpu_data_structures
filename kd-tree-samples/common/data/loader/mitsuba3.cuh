#pragma once
#include <charconv>
#include <unordered_map>

#include "tinyxml2.h"
#include "obj.cuh"
#include "../pt-math.cuh"
#include "../world.cuh"

class Mitsuba3Loader {
    std::string sceneDir;
    tinyxml2::XMLDocument sceneDesc;
    std::unordered_map<std::string, std::string> defaultValues;
    std::unordered_map<std::string, Material*> materials;

    std::unordered_map<const Material*, std::string> materialTextures;
    std::unordered_map<const Material*, std::string> materialBumpTextures;

    World *world;

    template<typename T>
    T resolveValue(std::string value);

    void memoizeDefaultValue(const tinyxml2::XMLElement *defaultElem);
    void loadIntegrator(const tinyxml2::XMLElement *integrator);
    void loadLight(const tinyxml2::XMLElement *light);
    void loadSensor(const tinyxml2::XMLElement *sensor);
    void loadMaterial(const tinyxml2::XMLElement *bsdf);
    Material *parseMaterial(const tinyxml2::XMLElement *bsdf, const std::string &name);
    void loadReflectance(Material *material, const std::string &name,
                         const tinyxml2::XMLElement *inner, const char *prop);
    owl::vec3f conductorAlbedo(const tinyxml2::XMLElement *inner);
    void loadShape(const tinyxml2::XMLElement *shape);
    Material *resolveSubmeshMaterial(const ObjSubmesh &sub, const tinyxml2::XMLElement *shape);
    const std::string &xmlTexturePath(const Material *material) const;
    const std::string &xmlBumpPath(const Material *material) const;

    float getDiffuseCoeff(const Material* mat, const tinyxml2::XMLElement *bsdf);
    float getSpecularCoeff(const Material* mat, const tinyxml2::XMLElement *bsdf);
    float getTransmissionCoeff(const Material* mat, const tinyxml2::XMLElement *bsdf);
public:
    explicit Mitsuba3Loader(const std::string& scene_name);

    [[nodiscard]] World *load();
};

/* Takes a (i) literal value inside a string or (ii) a "$value" and:
 * (i) Returns the literal cast into type T.
 * (ii) Looks up the "value" key in the defaults set and casts that.
*/
template<typename T>
T Mitsuba3Loader::resolveValue(std::string value) {
    if (value.at(0) == '$') {
        const std::string key = value.substr(1);
        try {
            value = defaultValues.at(key);
        } catch (std::out_of_range&) {
            std::cerr << "ERROR: Uninitialised value: " << key << std::endl;
            std::abort();
        }
    }
    T val;
    std::from_chars(value.data(), value.data() + value.size(), val);
    return val;
}

Mat4f load_transform(const tinyxml2::XMLElement *transform);