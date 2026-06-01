#pragma once

#include "owl/owl_host.h"
#include "owlViewer/OWLViewer.h"
#include "../common/data/world.cuh"

struct Viewer : owl::viewer::OWLViewer {
    explicit Viewer(const World *world, std::string scene_name = "frame");
    ~Viewer();
    void render() override;
    void draw() override;   // base blit + ImGui HUD overlay
    void resize(const owl::vec2i &newSize) override;
    void cameraChanged() override;
    void key(char key, const owl::vec2i &where) override; // 'P' screenshot, 'H' toggle HUD

    std::string sceneName;     // used in screenshot filenames
    bool sbtDirty = true;
    OWLRayGen rayGen   { nullptr };
    OWLContext context { nullptr };

    // ImGui HUD state (Phase 3.3). Stats are read-only.
    bool hudVisible = true;
    float lastFrameMs = 0.f;
    std::string lastScreenshotPath;
    int numPhotons = 0;
    int numCaustic = 0;

    // Progressive accumulation state. accumBuffer holds linear radiance summed across
    // launches; accumID is the launch count since the last reset (camera move / resize);
    // targetSpp is how many launches to accumulate before idling (from the scene's spp).
    OWLBuffer accumBuffer { nullptr };
    int accumID  = 0;
    int targetSpp = 1;
};