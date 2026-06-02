#include "viewer.cuh"
#include "../common/data/world.cuh"
#include "../common/data/loader/texture.cuh"
#include "cuda/pathTracer.cuh"

#include <ctime>
#include <filesystem>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

extern "C" char pathTracer_ptx[];

Viewer::Viewer(const World *world, std::string scene_name) : sceneName(std::move(scene_name)) {
    context = owlContextCreate(nullptr, 1);
    owlContextSetRayTypeCount(context, RAY_TYPES_COUNT);
    OWLModule module = owlModuleCreate(context, pathTracer_ptx);

    OWLVarDecl triangles_geom_vars[] = {
        { "material", OWL_RAW_POINTER, OWL_OFFSETOF(TrianglesGeomData,material)},
        { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
        { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
        { "normal",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,normal)},
        { "texCoord", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,texCoord)},
        { "albedoTexture", OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData,albedoTexture)},
        { "faceted", OWL_BOOL, OWL_OFFSETOF(TrianglesGeomData, faceted)},
        { nullptr /* Sentinel to mark end-of-list */}
    };

    OWLGeomType triangles_geom_type
        = owlGeomTypeCreate(context,
                          OWL_TRIANGLES,
                          sizeof(TrianglesGeomData),
                          triangles_geom_vars,-1);
    owlGeomTypeSetClosestHit(triangles_geom_type,PRIMARY,
                             module,"TriangleMesh");
    owlGeomTypeSetClosestHit(triangles_geom_type, SHADOW, module,"shadow");

    std::cout << "Building geometries...\n";

    // Upload meshes to GPU
    std::vector<OWLGeom> geometries;
    for (const auto model : world->models) {
        const auto mesh = model->mesh;
        OWLBuffer vertex_buffer
            = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->vertices.size(), mesh->vertices.data());
        OWLBuffer normal_buffer
            = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->normals.size(), mesh->normals.data());
        OWLBuffer index_buffer
            = owlDeviceBufferCreate(context,OWL_INT3,mesh->indices.size(), mesh->indices.data());

        OWLGeom triangles_geom
            = owlGeomCreate(context, triangles_geom_type);

        owlTrianglesSetVertices(triangles_geom,vertex_buffer,
                                mesh->vertices.size(),sizeof(owl::vec3f),0);
        owlTrianglesSetIndices(triangles_geom,index_buffer,
                               mesh->indices.size(),sizeof(owl::vec3i),0);

        owlGeomSetBuffer(triangles_geom,"vertex", vertex_buffer);
        owlGeomSetBuffer(triangles_geom,"index", index_buffer);
        owlGeomSetBuffer(triangles_geom,"normal", normal_buffer);
        owlGeomSet1b(triangles_geom, "faceted", mesh->faceted);

        // UVs + albedo texture (Phase 1.3). Both stay unbound (null / 0) when absent,
        // and the closest-hit falls back to the material's flat albedo.
        if (!mesh->uvs.empty()) {
            OWLBuffer uv_buffer
                = owlDeviceBufferCreate(context, OWL_FLOAT2, mesh->uvs.size(), mesh->uvs.data());
            owlGeomSetBuffer(triangles_geom, "texCoord", uv_buffer);
        }
        if (!model->albedo_texture_path.empty()) {
            if (const OWLTexture tex = load_albedo_texture(context, model->albedo_texture_path)) {
                owlGeomSetTexture(triangles_geom, "albedoTexture", tex);
            }
        }

        // Copy material to device memory.
        Material *mat_ptr;
        cudaMalloc(reinterpret_cast<void**>(&mat_ptr),sizeof(Material));
        cudaMemcpy(mat_ptr, model->material, sizeof(Material), cudaMemcpyHostToDevice);
        owlGeomSetPointer(triangles_geom, "material", mat_ptr);

        geometries.emplace_back(triangles_geom);
    }

    OWLGroup triangles_group
        = owlTrianglesGeomGroupCreate(context,geometries.size(),geometries.data());
    owlGroupBuildAccel(triangles_group);
    OWLGroup owl_world
        = owlInstanceGroupCreate(context,1);
    owlInstanceGroupSetChild(owl_world,0,triangles_group);
    owlGroupBuildAccel(owl_world);

    // Miss program
    OWLVarDecl missProgVars[] =
    {
        { "sky_colour", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, sky_colour)},
        { /* sentinel to mark end of list */ }
    };
    OWLMissProg missProg
      = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                          missProgVars,-1);
    owlMissProgSet3f(missProg,"sky_colour",owl3f{.1f,.01f,.2f});

    owlMissProgCreate(context, module,"shadow",0,nullptr,-1);

    OWLVarDecl rayGenVars[] = {
        { "fbPtr",         OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,fbPtr)},
        { "accumBuffer", OWL_BUFPTR, OWL_OFFSETOF(RayGenData,accumBuffer)},
        { "accumID", OWL_INT, OWL_OFFSETOF(RayGenData,accumID)},
        { "depth", OWL_INT, OWL_OFFSETOF(RayGenData,depth)},
        { "pixel_samples", OWL_INT, OWL_OFFSETOF(RayGenData,pixel_samples)},
        { "num_diffuse_scattered", OWL_INT, OWL_OFFSETOF(RayGenData,num_diffuse_scattered)},
        { "indirect_intensity", OWL_FLOAT, OWL_OFFSETOF(RayGenData,indirect_intensity)},
        { "caustic_intensity", OWL_FLOAT, OWL_OFFSETOF(RayGenData,caustic_intensity)},
        { "photon_map", OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,photon_map)},
        { "photon_coords", OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,photon_coords)},
        { "num_photons", OWL_INT, OWL_OFFSETOF(RayGenData,num_photons)},
        { "caustic_map", OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,caustic_map)},
        { "caustic_coords", OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,caustic_coords)},
        { "num_caustic", OWL_INT, OWL_OFFSETOF(RayGenData,num_caustic)},
        { "resolution", OWL_INT2, OWL_OFFSETOF(RayGenData,resolution)},
        { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
        { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.pos)},
        { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_00)},
        { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_dv)},
        { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_du)},
        { "lights", OWL_BUFPTR, OWL_OFFSETOF(RayGenData,lights)},
        { "num_lights", OWL_INT, OWL_OFFSETOF(RayGenData,num_lights)},
        { /* sentinel to mark end of list */ },
    };

    rayGen
        = owlRayGenCreate(context,module,"ptRayGen", sizeof(RayGenData), rayGenVars,-1);
    owlRayGenSetGroup(rayGen,"world", owl_world);

    // Flatten the vector<Light*> into a contiguous device buffer of Light records.
    std::vector<Light> lights_flat;
    lights_flat.reserve(world->lights.size());
    for (const auto* l : world->lights) lights_flat.push_back(*l);
    auto lights_buf = owlDeviceBufferCreate(
        context, OWL_USER_TYPE(Light), lights_flat.size(), lights_flat.data());
    owlRayGenSetBuffer(rayGen, "lights", lights_buf);
    owlRayGenSet1i(rayGen, "num_lights", static_cast<int>(lights_flat.size()));

    // Initialise Viewer camera with params from scene description.
    camera.setOrientation(world->cam->lookFrom,
                          world->cam->lookAt,
                          world->cam->up,
                          world->cam->image.fov);

    // Set RayGen constant attributes
    owlRayGenSet1i(rayGen, "pixel_samples", world->cam->image.pixel_samples);
    // Progressive renderer accumulates one launch per displayed frame until it reaches
    // the scene's spp ("samples to converge"), then idles. cameraChanged() resets it.
    targetSpp = world->cam->image.pixel_samples > 0 ? world->cam->image.pixel_samples : 1;
    owlRayGenSet1i(rayGen, "num_diffuse_scattered", world->cam->image.num_diffuse_scattered);
    owlRayGenSet1f(rayGen, "indirect_intensity", world->cam->image.indirect_intensity);
    owlRayGenSet1f(rayGen, "caustic_intensity", world->cam->image.caustic_intensity);
    owlRayGenSetPointer(rayGen, "photon_map", world->photon_map);
    owlRayGenSetPointer(rayGen, "photon_coords", world->photon_coords);
    owlRayGenSet1i(rayGen, "num_photons", world->num_photons);
    owlRayGenSetPointer(rayGen, "caustic_map", world->caustic_map);
    owlRayGenSetPointer(rayGen, "caustic_coords", world->caustic_coords);
    owlRayGenSet1i(rayGen, "num_caustic", world->num_caustic);
    owlRayGenSet1i(rayGen, "depth", world->cam->image.depth);
    owlRayGenSet2i(rayGen, "resolution", reinterpret_cast<const owl2i&>(world->cam->image.resolution));
    setWindowSize(world->cam->image.resolution);

    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);

    // HUD stats (read-only): photon counts captured once at load.
    numPhotons = world->num_photons;
    numCaustic = world->num_caustic;

    // Dear ImGui init. The OWLViewer base ctor has already created the GLFW window
    // (`handle`). install_callbacks=false so ImGui doesn't replace OWLViewer's input
    // callbacks (the HUD is read-only, so it doesn't need mouse/keyboard routing).
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    glfwMakeContextCurrent(handle);
    ImGui_ImplGlfw_InitForOpenGL(handle, /*install_callbacks=*/false);
    ImGui_ImplOpenGL3_Init("#version 130");
}

Viewer::~Viewer()
{
    // OWLViewer::showAndRun() destroys the window and calls glfwTerminate() before we
    // get here, so the GLFW/GL backends are already torn down — calling their Shutdown()
    // would hit "GLFW library is not initialized". Only the CPU-side context needs freeing;
    // the backend resources are reclaimed as the process exits.
    ImGui::DestroyContext();
}

void Viewer::render()
{
    // Converged: the accumulation has reached the target sample count, so there is
    // nothing new to compute. fbPtr already holds the result; let the viewer re-blit it.
    if (accumID >= targetSpp) return;

    // accumID is a per-launch uniform, so the SBT must be rebuilt before each launch.
    owlRayGenSet1i(rayGen, "accumID", accumID);
    owlBuildSBT(context);
    sbtDirty = false;

    const auto start = std::chrono::high_resolution_clock::now();
    owlRayGenLaunch2D(rayGen, fbSize.x, fbSize.y);
    cudaDeviceSynchronize();
    const auto end = std::chrono::high_resolution_clock::now();

    accumID++;
    lastFrameMs = std::chrono::duration<float, std::milli>(end - start).count();
}

void Viewer::draw()
{
    // Blit the path-traced framebuffer first (makes the GL context current too), then
    // overlay the ImGui HUD on top, before showAndRun swaps buffers.
    OWLViewer::draw();

    if (!hudVisible) return;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("Stats", nullptr,
                 ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNav);
    const bool converged = accumID >= targetSpp;
    ImGui::Text("Sample:   %d / %d%s", accumID, targetSpp, converged ? "  (converged)" : "");
    ImGui::Text("Last sample took:    %.2f ms", lastFrameMs);
    ImGui::Text("Total photons:  %d global, %d caustic", numPhotons, numCaustic);
    ImGui::Text("Gathered photons:  %d global, %d caustic", K_GLOBAL_PHOTONS, K_CAUSTIC_PHOTONS);
    const owl::vec3f from = camera.getFrom();
    const owl::vec3f at = camera.getAt();
    ImGui::Text("Camera Position:  %.1f %.1f %.1f", from.x, from.y, from.z);
    ImGui::Text("Pointed at:   %.1f %.1f %.1f", at.x, at.y, at.z);
    ImGui::Text("Last rendered path: %s", lastScreenshotPath.empty() ? "(none)" : lastScreenshotPath.c_str());
    ImGui::Separator();
    ImGui::TextDisabled("[P] screenshot   [H] toggle HUD");
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Viewer::resize(const owl::vec2i &newSize)
{
    OWLViewer::resize(newSize);   // reallocates fbPointer, updates fbSize

    // Match the accumulation buffer to the new framebuffer size.
    const size_t n = static_cast<size_t>(newSize.x) * newSize.y;
    if (!accumBuffer) {
        accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, n, nullptr);
    } else {
        owlBufferResize(accumBuffer, n);
    }
    owlRayGenSetBuffer(rayGen, "accumBuffer", accumBuffer);

    cameraChanged();   // updates camera vars + fbPtr/resolution and resets accumID
}

void Viewer::cameraChanged()
{
    const owl::vec3f lookFrom = camera.getFrom();
    const owl::vec3f lookAt = camera.getAt();
    const owl::vec3f lookUp = camera.getUp();

    // Frustum half-extent is tan(fovy/2): it grows with FOV (wider view). OWL's
    // getCosFovy() returns cos(fovy), which shrinks as FOV grows and flips sign
    // past 90 deg — so a larger FOV would paradoxically zoom in.
    const float fovyRad = camera.getFovyInDegrees() * float(M_PI) / 180.f;
    const float tanHalfFovy = tanf(0.5f * fovyRad);
    // ----------- compute variable values  ------------------
    owl::vec3f camera_pos = lookFrom;
    owl::vec3f camera_d00
      = normalize(lookAt-lookFrom);
    float aspect = fbSize.x / float(fbSize.y);
    owl::vec3f camera_ddu
      = tanHalfFovy * aspect * normalize(cross(camera_d00,lookUp));
    owl::vec3f camera_ddv
      = tanHalfFovy * normalize(cross(camera_ddu,camera_d00));
    camera_d00 -= 0.5f * camera_ddu;
    camera_d00 -= 0.5f * camera_ddv;

    // ----------- set variables  ----------------------------
    owlRayGenSet1ul   (rayGen,"fbPtr",        reinterpret_cast<uint64_t>(fbPointer));
    owlRayGenSet2i    (rayGen,"resolution",   reinterpret_cast<const owl2i&>(fbSize));
    owlRayGenSet3f    (rayGen,"camera.pos",   reinterpret_cast<const owl3f&>(camera_pos));
    owlRayGenSet3f    (rayGen,"camera.dir_00",reinterpret_cast<const owl3f&>(camera_d00));
    owlRayGenSet3f    (rayGen,"camera.dir_du",reinterpret_cast<const owl3f&>(camera_ddu));
    owlRayGenSet3f    (rayGen,"camera.dir_dv",reinterpret_cast<const owl3f&>(camera_ddv));

    // Camera (or framebuffer) changed: restart accumulation from scratch so stale
    // radiance from the previous viewpoint isn't blended in.
    accumID = 0;
    sbtDirty = true;
}

void Viewer::key(char key, const owl::vec2i &where)
{
    // 'P' saves the current (tonemapped, accumulated) frame to screenshots/<scene>_<ts>.png.
    // OWLViewer::screenShot reads fbPointer, which already holds exactly what's on screen.
    if (key == 'p' || key == 'P') {
        std::error_code ec;
        std::filesystem::create_directories("screenshots", ec);

        char ts[32];
        const std::time_t t = std::time(nullptr);
        std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));

        const std::string path = "screenshots/" + sceneName + "_" + ts + ".png";
        screenShot(path);
        lastScreenshotPath = path;
        return;
    }
    if (key == 'h' || key == 'H') {
        hudVisible = !hudVisible;
        return;
    }
    // Defer everything else (camera controls, etc.) to the base viewer.
    OWLViewer::key(key, where);
}