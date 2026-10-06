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
    int totalPhotons;
    OptixTraversableHandle world;
    int maxDepth;
    bool causticsMode;
};

struct PointLightRGD: public PhotonMapperRGD
{
    owl::vec3f position;
    owl::vec3f color;       // total emitted flux Phi (power x emission factor), RGB
    float intensity;        // legacy, unused
    // Phase 2.2: light-type-aware emission.
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
    struct {
        owl::vec3f origin;
        owl::vec3f direction;
        owl::vec3f color;
    } scattered;
    bool debug;
};

// (Legacy LightType/LightSource removed — unused; the live light type is `Light`
//  in common/data/world.cuh, introduced in Phase 2.1.)

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

    World* world;
    GeometryData geometryData;

    OWLBuffer photonsBuffer;
    OWLBuffer photonsCount;
    OWLBuffer causticsPhotonsBuffer;
    OWLBuffer causticsPhotonsCount;

    int maxDepth;
    int castedCausticsPhotons;
    int castedDiffusePhotons;
    // Photons-per-watt must be float: casted/totalWatts truncates to 0 as an int
    // whenever the casted count is below the total wattage (e.g. a small caustic
    // budget against Sponza's ~675000 W), which zeroes the launch width.
    float photonsPerWatt;
    float causticsPhotonsPerWatt;

    // Scene bounding sphere (for directional-light disk emission). Computed once at load.
    owl::vec3f sceneCenter;
    float sceneRadius;
};