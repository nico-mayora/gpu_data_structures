#pragma once

#include "owl/owl_host.h"
#include "owlViewer/OWLViewer.h"
#include "world.h"

struct Viewer : owl::viewer::OWLViewer {
    explicit Viewer(const World *world);
    void render() override;
    void resize(const owl::vec2i &newSize) override;
    void cameraChanged() override;

    bool sbtDirty = true;
    OWLRayGen rayGen   { nullptr };
    OWLContext context { nullptr };
};