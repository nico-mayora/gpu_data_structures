#include "photon-emitter.h"

#include "world.h"
#include "cuda/photonEmitter.h"
#include <cuda_runtime.h>

extern "C" char photonEmitter_ptx[];

PhotonEmitter::PhotonEmitter(const World *world) {
    context = owlContextCreate(nullptr, 1);
    owlContextSetRayTypeCount(context, RAY_TYPES_COUNT);
    OWLModule module = owlModuleCreate(context, photonEmitter_ptx);

    std::vector<Photon> initialPhotons(maxPhotons);
//    for (auto& photon : initialPhotons) {
//        photon.isDead = true;
//        photon.coords[0] = photon.coords[1] = photon.coords[2] = 0.0f;
//        photon.colour[0] = photon.colour[1] = photon.colour[2] = 0.0f;
//        photon.power[0] = photon.power[1] = photon.power[2] = 0.0f;
//        photon.dir[0] = photon.dir[1] = photon.dir[2] = 0.0f;
//    }
    
    photonsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(Photon), maxPhotons, initialPhotons.data());

    OWLVarDecl triangles_geom_vars[] = {
        { "material", OWL_RAW_POINTER, OWL_OFFSETOF(TrianglesGeomData,material)},
        { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
        { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
        { "normal",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,normal)},
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

    std::cout << "Building geometries...\n";

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

    OWLVarDecl missProgVars[] =
    {
        { "sky_colour", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, sky_colour)},
        { /* sentinel to mark end of list */ }
    };
    OWLMissProg missProg
      = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                          missProgVars,-1);
    owlMissProgSet3f(missProg,"sky_colour",owl3f{.1f,.3f,.5f});

    OWLVarDecl rayGenVars[] = {
        { "photons",         OWL_BUFPTR, OWL_OFFSETOF(Light,photons)},
        { "world",         OWL_GROUP,  OWL_OFFSETOF(Light,world)},
        { "maxDepth",       OWL_INT, OWL_OFFSETOF(Light,maxDepth)},
        { "position",       OWL_FLOAT3, OWL_OFFSETOF(Light,position)},
        { "colour",         OWL_FLOAT3, OWL_OFFSETOF(Light,colour)},
        { "power",          OWL_FLOAT,  OWL_OFFSETOF(Light,power)},
        { "type",           OWL_INT,    OWL_OFFSETOF(Light,type)},
        { /* sentinel to mark end of list */ }
    };

    rayGen = owlRayGenCreate(context, module, "photonEmitter", sizeof(Light), rayGenVars, -1);
    owlRayGenSetGroup(rayGen, "world", owl_world);
    owlRayGenSetBuffer(rayGen, "photons", photonsBuffer);
}

void PhotonEmitter::emit(const LightSource* light, int numPhotons, int maxDepth)
{
//    if (sbtDirty) {
//        owlBuildSBT(context);
//        sbtDirty = false;
//    }
    
    owl::vec3f ls_centre = 0.f;
    for (const auto &v : light->mesh->vertices) {
        ls_centre += v;
    }
    ls_centre /= static_cast<float>(light->mesh->vertices.size());
    std::cout << "Light source center calculated at (" << ls_centre.x << ", " << ls_centre.y << ", " << ls_centre.z << ")\n";
    std::cout << "Light radiance: (" << light->radiance.x << ", " << light->radiance.y << ", " << light->radiance.z << ")\n";
    std::cout << "Emitting " << numPhotons << " photons with max depth " << maxDepth << "\n";

    owlRayGenSet3f(rayGen, "position", reinterpret_cast<const owl3f&>(ls_centre));
    owlRayGenSet3f(rayGen, "colour", reinterpret_cast<const owl3f&>(light->radiance));
    owlRayGenSet1f(rayGen, "power", 1.0f);
    owlRayGenSet1i(rayGen, "maxDepth", maxDepth);
    owlRayGenSet1i(rayGen, "type", 0); // Point light
    

    std::cout << "Emitting " << numPhotons << " photons from light at (" 
              << ls_centre.x << ", " << ls_centre.y << ", " << ls_centre.z 
              << ") storing " << (numPhotons * maxDepth) << " total photons" << std::endl;

    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);

    owlRayGenLaunch2D(rayGen, numPhotons, 1);

    int newTotalPhotons = totalStoredPhotons + (numPhotons * maxDepth);
    if (newTotalPhotons > maxPhotons) {
        std::cerr << "Error: Photon buffer overflow! Requested: " << newTotalPhotons
                  << ", Available: " << maxPhotons << std::endl;
        return;
    }
    totalStoredPhotons = newTotalPhotons;
}

// NEEDS SMALL REFACTOR (SPLIT FILTER AND SAVE PHOTONS OUTSIDE OF GPU IN EMIT CALL, TO CONCAT SIDDERENT RUNS)
std::vector<Photon> PhotonEmitter::getPhotons() const
{
    if (totalStoredPhotons <= 0) {
        std::cout << "No photons stored." << std::endl;
        return {};
    }

    std::vector<Photon> allPhotons(totalStoredPhotons);

    const void* devicePtr = owlBufferGetPointer(photonsBuffer, 0);
    if (devicePtr == nullptr) {
        std::cerr << "Error: Failed to get photons buffer pointer" << std::endl;
        return {};
    }

    cudaError_t err = cudaMemcpy(allPhotons.data(), devicePtr, totalStoredPhotons * sizeof(Photon), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error copying photons: " << cudaGetErrorString(err) << std::endl;
        return {};
    }

    std::vector<Photon> livePhotons;
    livePhotons.reserve(totalStoredPhotons);

    for (const auto& photon : allPhotons) {
        if (!photon.isDead) {
            livePhotons.push_back(photon);
        }
    }

    std::cout << "Retrieved " << livePhotons.size() << " live photons out of "
              << totalStoredPhotons << " total stored photons." << std::endl;

    return livePhotons;
}

void PhotonEmitter::clearPhotons() {
    totalStoredPhotons = 0;
}
