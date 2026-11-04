#pragma once

#include "owl/include/owl/owl.h"
#include "owl/include/owl/common/math/vec.h"
#include "owl/include/owl/common/math/random.h"
#include <vector>
#include "../../common/data/world.cuh"

struct PhotonMapperRGD
{
    Photon *photons;
    int *photonsCount;
    int totalPhotons;
    OptixTraversableHandle world;
    int maxDepth;
    bool causticsMode;
};

struct PointLightRGD: public PhotonMapperRGD
{
    owl::vec3f position;
    owl::vec3f color;       // total emitted flux Phi (power x emission factor), RGB
    int lightType;          // LightType (point / spot / directional)
    owl::vec3f direction;   // spot/directional propagation axis (unit)
    float cosOuter;         // spot outer cone (cos)
    float cosInner;         // spot inner cone (cos)
    owl::vec3f diskCenter;  // directional: scene bounding-sphere center
    float diskRadius;       // directional: scene bounding-sphere radius
};

struct PhotonMapperPRD
{
    owl::LCG<> random;
    owl::vec3f color;
    owl::vec3f direction;
    RayEvent event;
    bool hitDiffuse;
    struct {
        owl::vec3f origin;
        owl::vec3f direction;
        owl::vec3f color;
    } scattered;
    bool debug;
};

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

    World* world;
    GeometryData geometryData;

    OWLBuffer photonsBuffer;
    OWLBuffer photonsCount;
    OWLBuffer causticsPhotonsBuffer;
    OWLBuffer causticsPhotonsCount;

    int maxDepth;
    int castedCausticsPhotons;
    int castedDiffusePhotons;
    float photonsPerWatt;
    float causticsPhotonsPerWatt;

    // Scene bounding sphere (for directional-light disk emission). Computed once at load.
    owl::vec3f sceneCenter;
    float sceneRadius;
};