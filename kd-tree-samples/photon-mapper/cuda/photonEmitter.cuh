#pragma once

#include "owl/include/owl/owl.h"
#include "owl/include/owl/common/math/vec.h"
#include "owl/include/owl/common/math/random.h"
#include <vector>
#include "../../common/data/world.cuh"

struct PhotonMapperRGD
{
    EmittedPhoton *photons;
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