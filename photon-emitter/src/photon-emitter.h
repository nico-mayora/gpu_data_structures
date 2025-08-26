#pragma once

#include <vector>
#include "owl/owl_host.h"
#include "owlViewer/OWLViewer.h"
#include "world.h"
#include "cuda/photonEmitter.h"

struct PhotonEmitter {
    explicit PhotonEmitter(const World *world);
    void emit(const LightSource* light, int numPhotons, int maxDepth = 10);
    std::vector<Photon> getPhotons() const;
    void clearPhotons();

    bool sbtDirty = true;
    OWLRayGen rayGen   { nullptr };
    OWLContext context { nullptr };
    
private:
    OWLBuffer photonsBuffer   { nullptr };
    int maxPhotons = 1000000;
    int totalStoredPhotons = 0;
};
