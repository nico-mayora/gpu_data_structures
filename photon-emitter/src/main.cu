#include <iostream>
#include <fstream>
#include <iomanip>
#include "owl/owl.h"
#include "./cuda/photonEmitter.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../externals/stb/stb_image_write.h"
#include "loader/mitsuba3.cuh"
#include "writer/photon-file-manager.h"

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char photonEmitter_ptx[];

void writeAlivePhotons(const Photon* photons, int count, const std::string& filename) {
  std::ofstream outFile(filename);

  if (!outFile.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  outFile << std::fixed << std::setprecision(6);

  for (int i = 0; i < count; i++) {
    auto photon = photons[i];
    outFile << photon.pos.x << " " << photon.pos.y << " " << photon.pos.z << " "
            << photon.dir.x << " " << photon.dir.y << " " << photon.dir.z << " "
            << photon.color.x << " " << photon.color.y << " " << photon.color.z << "\n";
  }

  outFile.close();
}

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
    owlGeomSetBuffer(trianglesGeom,"material", materialBuffer);

    std::cout << "All info about mesh: " << "\n";
    std::cout << " #vertices: " << vertices.size() << "\n";
    std::cout << " #triangles: " << indices.size() << "\n";
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

  if (causticsMode) {
    owlRayGenSetBuffer(program.rayGen,"photons",program.causticsPhotonsBuffer);
    owlRayGenSetBuffer(program.rayGen,"photonsCount",program.causticsPhotonsCount);
  } else {
    owlRayGenSetBuffer(program.rayGen,"photons",program.photonsBuffer);
    owlRayGenSetBuffer(program.rayGen,"photonsCount",program.photonsCount);
  }

  const int initialPhotons = program.photonsPerWatt * (light->power.x + light->power.y + light->power.z);

  owlBuildSBT(program.owlContext);
  owlRayGenLaunch2D(program.rayGen,initialPhotons,1);
}

void initPhotonBuffers(Program &program) {
  program.photonsBuffer = owlHostPinnedBufferCreate(program.owlContext, OWL_USER_TYPE(Photon), program.castedDiffusePhotons * program.maxDepth);
  program.photonsCount = owlHostPinnedBufferCreate(program.owlContext, OWL_INT, 1);
  owlBufferClear(program.photonsCount);
}

void computePhotonsPerWatt(Program &program) {
  auto light = program.world->scene_light;
  auto totalWatts = light->power.x + light->power.y + light->power.z;

  program.photonsPerWatt = program.castedDiffusePhotons / totalWatts;
}

void runNormal(Program &program, const std::string &output_filename) {
  LOG("launching normal photons ...")

  runPointLightRayGen(program, program.world->scene_light, false);

  LOG("done with launch, writing photons ...")
  auto *fb = static_cast<const Photon*>(owlBufferGetPointer(program.photonsBuffer, 0));
  auto count = *(int*)owlBufferGetPointer(program.photonsCount, 0);

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

  auto photons_filename = "photons.txt";
  program.castedDiffusePhotons = 1'000'000;
  program.castedCausticsPhotons = 100;
  program.maxDepth = 2;

  LOG_OK("Loaded world.")

  program.geometryData = loadGeometry(program.owlContext, program.world);

  owlGeomTypeSetClosestHit(program.geometryData.trianglesGeomType, 0, program.owlModule,"triangleMeshClosestHit");
  owlMissProgCreate(program.owlContext, program.owlModule, "miss", 0, nullptr, -1);

  computePhotonsPerWatt(program);
  initPhotonBuffers(program);

  setupPointLightRayGenProgram(program);

  owlBuildPrograms(program.owlContext);
  owlBuildPipeline(program.owlContext);

  LOG("launching ...")

  runNormal(program, photons_filename);

  LOG("destroying devicegroup ...");
  owlContextDestroy(program.owlContext);

  LOG_OK("seems all went OK; app is done, this should be the last output ...");
  return 0;
}
