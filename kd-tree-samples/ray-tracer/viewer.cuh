#pragma once

#include "owl/owl_host.h"
#include "owlViewer/OWLViewer.h"
#include "../common/data/world.cuh"

struct Viewer : owl::viewer::OWLViewer {
    explicit Viewer(const World *world, std::string scene_name = "frame", bool benchmark = false, bool visible = true);
    ~Viewer();
    void render() override;
    void draw() override;
    void resize(const owl::vec2i &newSize) override;
    void cameraChanged() override;
    void key(char key, const owl::vec2i &where) override; // 'P' screenshot, 'H' toggle HUD

    bool visible = true;
    bool benchmarkMode = false;
    std::vector<float> benchmarkTimes;

    std::string sceneName;     // used in screenshot filenames
    bool sbtDirty = true;
    OWLRayGen rayGen   { nullptr };
    OWLContext context { nullptr };

    bool hudVisible = true;
    float lastFrameMs = 0.f;
    std::string lastScreenshotPath;
    int numPhotons = 0;
    int numCaustic = 0;

    OWLBuffer accumBuffer { nullptr };
    int accumID  = 0;
    int targetSpp = 1;

private:
    void takeScreenshot();
};