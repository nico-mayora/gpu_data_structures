#include <iostream>
#include <fstream>
#include <iomanip>
#include "owl/owl.h"
#include "./cuda/photonEmitter.cuh"
#include "../common/data/loader/mitsuba3.cuh"
#include "../common/data/photon/photon-file-manager.cuh"

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char photonEmitter_ptx[];

void setupPointLightRayGenProgram(Program &program) {
  OWLVarDecl rayGenVars[] = {
          { "photons",OWL_BUFPTR,OWL_OFFSETOF(PointLightRGD,photons)},
          { "photonsCount",OWL_BUFPTR,OWL_OFFSETOF(PointLightRGD,photonsCount)},
          { "maxDepth",OWL_INT,OWL_OFFSETOF(PointLightRGD, maxDepth)},
          {"causticsMode", OWL_BOOL, OWL_OFFSETOF(PointLightRGD, causticsMode)},
          { "world",OWL_GROUP,OWL_OFFSETOF(PointLightRGD,world)},
          { "position",OWL_FLOAT3,OWL_OFFSETOF(PointLightRGD,position)},
          { "color",OWL_FLOAT3,OWL_OFFSETOF(PointLightRGD,color)},
          { "intensity",OWL_FLOAT,OWL_OFFSETOF(PointLightRGD,intensity)},
          { /* sentinel to mark end of list */ }
  };

  program.rayGen = owlRayGenCreate(program.owlContext,program.owlModule,"pointLightRayGen",
                                   sizeof(PointLightRGD),
                                   rayGenVars,-1);

  owlRayGenSetGroup(program.rayGen,"world",program.geometryData.worldGroup);
  owlRayGenSet1i(program.rayGen,"maxDepth",program.maxDepth);
}

GeometryData loadGeometry(OWLContext &owlContext, World* world){
  GeometryData data;

  OWLVarDecl trianglesGeomVars[] = {
          { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
          { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
          { "normal", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,normal)},
          { "material", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,material)},
          { nullptr /* Sentinel to mark end-of-list */}
  };

  data.trianglesGeomType = owlGeomTypeCreate(owlContext,
                                             OWL_TRIANGLES,
                                             sizeof(TrianglesGeomData),
                                             trianglesGeomVars,-1);

//  const int numMeshes = static_cast<int>(world->meshes.size());

  for (const auto model : world->models) {
    auto mesh = model->mesh;
    auto vertices = mesh->vertices;
    auto indices = mesh->indices;
    auto material = model->material;

    std::vector<Material> mats_vec = { *material };

    OWLBuffer vertexBuffer
            = owlDeviceBufferCreate(owlContext,OWL_FLOAT3,vertices.size(), vertices.data());
    OWLBuffer indexBuffer
            = owlDeviceBufferCreate(owlContext,OWL_INT3,indices.size(), indices.data());
    OWLBuffer normal_buffer
            = owlDeviceBufferCreate(owlContext, OWL_FLOAT3, mesh->normals.size(), mesh->normals.data());
    OWLBuffer materialBuffer
            = owlDeviceBufferCreate(owlContext,OWL_USER_TYPE(Material),1, mats_vec.data());

    OWLGeom trianglesGeom
            = owlGeomCreate(owlContext,data.trianglesGeomType);

    owlTrianglesSetVertices(trianglesGeom,vertexBuffer,
                            vertices.size(),sizeof(owl::vec3f),0);
    owlTrianglesSetIndices(trianglesGeom,indexBuffer,
                           indices.size(),sizeof(owl::vec3i),0);


    owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
    owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);
    owlGeomSetBuffer(trianglesGeom,"normal", normal_buffer);
    owlGeomSetBuffer(trianglesGeom,"material", materialBuffer);

    std::cout << "All info about mesh: " << "\n";
    std::cout << " #vertices: " << vertices.size() << "\n";
    std::cout << " #triangles: " << indices.size() << "\n";
    std::cout << " #normals: " << mesh->normals.size() << "\n";
    for (int i = 0; i < mesh->normals.size(); i++) {
      std::cout << "normal[" << i << "]: " << mesh->normals[i].x << " "
                << mesh->normals[i].y << " "
                << mesh->normals[i].z << "\n";
    }
    std::cout << " #material: " << material->albedo.x << " "
              << material->albedo.y << " "
              << material->albedo.z << "\n";
    std::cout << "---------------------------------------\n";

    data.geometry.push_back(trianglesGeom);
  }

  data.trianglesGroup = owlTrianglesGeomGroupCreate(owlContext,data.geometry.size(),data.geometry.data());
  owlGroupBuildAccel(data.trianglesGroup);

  data.worldGroup = owlInstanceGroupCreate(owlContext,1);
  owlInstanceGroupSetChild(data.worldGroup,0,data.trianglesGroup);
  owlGroupBuildAccel(data.worldGroup);

  return data;
}

void runPointLightRayGen(Program &program, const PointLight* light, bool causticsMode) {
  owlRayGenSet1b(program.rayGen,"causticsMode",causticsMode);
  owlRayGenSet3f(program.rayGen,"position",reinterpret_cast<const owl3f&>(light->position));
  owlRayGenSet3f(program.rayGen,"color",reinterpret_cast<const owl3f&>(light->power));
  owlRayGenSet1f(program.rayGen,"intensity",1);

  int initialPhotons;
  if (causticsMode) {
    owlRayGenSetBuffer(program.rayGen,"photons",program.causticsPhotonsBuffer);
    owlRayGenSetBuffer(program.rayGen,"photonsCount",program.causticsPhotonsCount);
    initialPhotons = program.causticsPhotonsPerWatt * (light->power.x + light->power.y + light->power.z);
  } else {
    owlRayGenSetBuffer(program.rayGen,"photons",program.photonsBuffer);
    owlRayGenSetBuffer(program.rayGen,"photonsCount",program.photonsCount);
    initialPhotons = program.photonsPerWatt * (light->power.x + light->power.y + light->power.z);
  }

  owlBuildSBT(program.owlContext);
  owlRayGenLaunch2D(program.rayGen,initialPhotons,1);
}

void initPhotonBuffers(Program &program) {
  program.photonsBuffer = owlHostPinnedBufferCreate(program.owlContext, OWL_USER_TYPE(EmittedPhoton), program.castedDiffusePhotons * program.maxDepth);
  program.photonsCount = owlHostPinnedBufferCreate(program.owlContext, OWL_INT, 1);
  owlBufferClear(program.photonsCount);

  program.causticsPhotonsBuffer = owlHostPinnedBufferCreate(program.owlContext, OWL_USER_TYPE(EmittedPhoton), program.castedCausticsPhotons * program.maxDepth);
  program.causticsPhotonsCount = owlHostPinnedBufferCreate(program.owlContext, OWL_INT, 1);
  owlBufferClear(program.causticsPhotonsCount);
}

void computePhotonsPerWatt(Program &program) {
  auto light = program.world->scene_light;
  auto totalWatts = light->power.x + light->power.y + light->power.z;

  program.photonsPerWatt = program.castedDiffusePhotons / totalWatts;
  program.causticsPhotonsPerWatt = program.castedCausticsPhotons / totalWatts;
}

void runNormal(Program &program, const std::string &output_filename) {
  LOG("launching normal photons ...")

  runPointLightRayGen(program, program.world->scene_light, false);

  LOG("done with launch, writing normal photons ...")
  auto *fb = static_cast<const EmittedPhoton*>(owlBufferGetPointer(program.photonsBuffer, 0));
  auto count = *(int*)owlBufferGetPointer(program.photonsCount, 0);

  LOG("normal photon count: " << count)
  PhotonFileManager::savePhotonsToFile(fb, count, output_filename);
}

void runCaustics(Program &program, const std::string &output_filename) {
  LOG("launching caustic photons ...")

  runPointLightRayGen(program, program.world->scene_light, true);

  LOG("done with launch, writing caustic photons ...")
  auto *fb = static_cast<const EmittedPhoton*>(owlBufferGetPointer(program.causticsPhotonsBuffer, 0));
  auto count = *(int*)owlBufferGetPointer(program.causticsPhotonsCount, 0);

  LOG("caustic photon count: " << count)
  PhotonFileManager::savePhotonsToFile(fb, count, output_filename);
}

int main(int ac, char **av)
{
  LOG("Starting up...");

  Program program;
  program.owlContext = owlContextCreate(nullptr,1);
  program.owlModule = owlModuleCreate(program.owlContext, photonEmitter_ptx);
  owlContextSetRayTypeCount(program.owlContext, 1);

  LOG("Loading Config file...")

  const auto loader = new Mitsuba3Loader("cornell-box");
  program.world = loader->load();

  auto normal_photons_filename = "normal_photons.txt";
  auto caustic_photons_filename = "caustic_photons.txt";
  program.castedDiffusePhotons = 100'000;
  program.castedCausticsPhotons = 100'000;
  program.maxDepth = 10;

  LOG_OK("Loaded world.")

  program.geometryData = loadGeometry(program.owlContext, program.world);

  owlGeomTypeSetClosestHit(program.geometryData.trianglesGeomType, 0, program.owlModule,"triangleMeshClosestHit");
  owlMissProgCreate(program.owlContext, program.owlModule, "miss", 0, nullptr, -1);

  computePhotonsPerWatt(program);
  initPhotonBuffers(program);

  setupPointLightRayGenProgram(program);

  owlBuildPrograms(program.owlContext);
  owlBuildPipeline(program.owlContext);

  runNormal(program, normal_photons_filename);
  runCaustics(program, caustic_photons_filename);

  LOG("destroying devicegroup ...");
  owlContextDestroy(program.owlContext);

  LOG_OK("seems all went OK; app is done, this should be the last output ...");
  return 0;
}
