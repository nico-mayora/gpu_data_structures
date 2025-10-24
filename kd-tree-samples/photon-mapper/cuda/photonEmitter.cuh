#pragma once

#include "owl/include/owl/owl.h"
#include "owl/include/owl/common/math/vec.h"
#include "owl/include/owl/common/math/random.h"
#include <vector>
#include "../world.cuh"

struct Photon
{
    owl::vec3f pos;
    owl::vec3f dir;
    int power;
    owl::vec3f color;
};

struct PhotonMapperRGD
{
    Photon *photons;
    int *photonsCount;
    OptixTraversableHandle world;
    int maxDepth;
    bool causticsMode;
};

struct PointLightRGD: public PhotonMapperRGD
{
    owl::vec3f position;
    owl::vec3f color;
    float intensity;
};

enum RayEvent
{
    MISS = 0,
    ABSORBED = 1,
    SCATTER_DIFFUSE = 2,
    SCATTER_SPECULAR = 4,
    SCATTER_REFRACT = 8,
};

struct PhotonMapperPRD
{
    owl::LCG<> random;
    owl::vec3f color;
    owl::vec3f direction;
    RayEvent event;
    struct {
        owl::vec3f origin;
        owl::vec3f direction;
        owl::vec3f color;
    } scattered;
    bool debug;
};

//struct Material {
//    owl::vec3f albedo;
//    float diffuse;
//    float specular;
//    float transmission;
//    float refraction_idx;
//};

/* variables for the triangle mesh geometry */
struct TrianglesGeomData
{
    Material *material;
    owl::vec3i *index;
    owl::vec3f *vertex;
    owl::vec3f *normal;
};

/* The vectors need to be (trivially) transformed into regular arrays
   before being passed into OptiX */
//struct Mesh {
//    std::string name;
//    std::vector<owl::vec3f> vertices;
//    std::vector<owl::vec3i> indices;
//    std::shared_ptr<Material> material;
//};

enum LightType {
    POINT_LIGHT,
    SQUARE_LIGHT,
};

struct LightSource {
    LightType source_type;
    owl::vec3f pos;
    double power;
    owl::vec3f rgb;
    /* for emission surface */
    owl::vec3f normal;
    double side_length;

    /* calculated values */
    int num_photons;
};

/* This holds all the state required for the path tracer to function.
 * As we use the STL, this is code in C++ land that needs a bit of
 * glue to transform to data that can be held in the GPU.
 */
//struct World {
//    std::vector<LightSource> light_sources;
//    std::vector<Mesh> meshes;
//};

struct GeometryData {
    std::vector<OWLGeom> geometry;
    OWLGeomType trianglesGeomType;
    OWLGroup trianglesGroup;
    OWLGroup worldGroup;
};

struct Program {
    OWLContext owlContext;
    OWLModule owlModule;
    OWLRayGen rayGen;

//    std::unique_ptr<World> world;
    World* world;
    GeometryData geometryData;

    OWLBuffer photonsBuffer;
    OWLBuffer photonsCount;
    OWLBuffer causticsPhotonsBuffer;
    OWLBuffer causticsPhotonsCount;

    int maxDepth;
    int castedCausticsPhotons;
    int castedDiffusePhotons;
    int photonsPerWatt;
    int causticsPhotonsPerWatt;
};