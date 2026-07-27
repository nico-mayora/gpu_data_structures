#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include "owl/owl.h"
#include "./cuda/photonEmitter.cuh"
#include "../common/data/loader/mitsuba3.cuh"
#include "../common/data/loader/texture.cuh"
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
          { "totalPhotons",OWL_INT,OWL_OFFSETOF(PointLightRGD,totalPhotons)},
          { "maxDepth",OWL_INT,OWL_OFFSETOF(PointLightRGD, maxDepth)},
          {"causticsMode", OWL_BOOL, OWL_OFFSETOF(PointLightRGD, causticsMode)},
          { "world",OWL_GROUP,OWL_OFFSETOF(PointLightRGD,world)},
          { "position",OWL_FLOAT3,OWL_OFFSETOF(PointLightRGD,position)},
          { "color",OWL_FLOAT3,OWL_OFFSETOF(PointLightRGD,color)},
          { "intensity",OWL_FLOAT,OWL_OFFSETOF(PointLightRGD,intensity)},
          { "lightType",OWL_INT,OWL_OFFSETOF(PointLightRGD,lightType)},
          { "direction",OWL_FLOAT3,OWL_OFFSETOF(PointLightRGD,direction)},
          { "cosOuter",OWL_FLOAT,OWL_OFFSETOF(PointLightRGD,cosOuter)},
          { "cosInner",OWL_FLOAT,OWL_OFFSETOF(PointLightRGD,cosInner)},
          { "diskCenter",OWL_FLOAT3,OWL_OFFSETOF(PointLightRGD,diskCenter)},
          { "diskRadius",OWL_FLOAT,OWL_OFFSETOF(PointLightRGD,diskRadius)},
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
          { "texCoord", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,texCoord)},
          { "albedoTexture", OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData,albedoTexture)},
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

    // UVs + albedo texture (Phase 1.3); photons must pick up textured albedo so
    // caustic/global colours match the textured surfaces. Unbound when absent.
    if (!mesh->uvs.empty()) {
      OWLBuffer uv_buffer
              = owlDeviceBufferCreate(owlContext, OWL_FLOAT2, mesh->uvs.size(), mesh->uvs.data());
      owlGeomSetBuffer(trianglesGeom, "texCoord", uv_buffer);
    }
    if (!model->albedo_texture_path.empty()) {
      if (const OWLTexture tex = load_albedo_texture(owlContext, model->albedo_texture_path)) {
        owlGeomSetTexture(trianglesGeom, "albedoTexture", tex);
      }
    }

    // std::cout << "All info about mesh: " << "\n";
    // std::cout << " #vertices: " << vertices.size() << "\n";
    // std::cout << " #triangles: " << indices.size() << "\n";
    // std::cout << " #normals: " << mesh->normals.size() << "\n";
    // for (int i = 0; i < mesh->normals.size(); i++) {
    //   std::cout << "normal[" << i << "]: " << mesh->normals[i].x << " "
    //             << mesh->normals[i].y << " "
    //             << mesh->normals[i].z << "\n";
    // }
    // std::cout << " #material: " << material->albedo.x << " "
    //           << material->albedo.y << " "
    //           << material->albedo.z << "\n";
    // std::cout << "---------------------------------------\n";

    data.geometry.push_back(trianglesGeom);
  }

  data.trianglesGroup = owlTrianglesGeomGroupCreate(owlContext,data.geometry.size(),data.geometry.data());
  owlGroupBuildAccel(data.trianglesGroup);

  data.worldGroup = owlInstanceGroupCreate(owlContext,1);
  owlInstanceGroupSetChild(data.worldGroup,0,data.trianglesGroup);
  owlGroupBuildAccel(data.worldGroup);

  return data;
}

static float emissionFactor(const Program &program, const PointLight *light);
static float lightFluxScalar(const Program &program, const PointLight *light);

void runPointLightRayGen(Program &program, const PointLight* light, bool causticsMode) {
  const float factor = emissionFactor(program, light);
  const owl::vec3f flux = light->power * factor;     // total emitted flux Phi (RGB)

  // This light's share of the budget, proportional to its emitted flux.
  const float perFlux = causticsMode ? program.causticsPhotonsPerWatt : program.photonsPerWatt;
  const int initialPhotons = static_cast<int>(std::lround(perFlux * lightFluxScalar(program, light)));
  if (initialPhotons < 1) return;   // negligible share -> skip (also avoids a 0-width launch)

  owlRayGenSet1b(program.rayGen,"causticsMode",causticsMode);
  owlRayGenSet3f(program.rayGen,"position",reinterpret_cast<const owl3f&>(light->position));
  owlRayGenSet3f(program.rayGen,"color",reinterpret_cast<const owl3f&>(flux));
  owlRayGenSet1f(program.rayGen,"intensity",1);
  owlRayGenSet1i(program.rayGen,"lightType",static_cast<int>(light->type));
  owlRayGenSet3f(program.rayGen,"direction",reinterpret_cast<const owl3f&>(light->direction));
  owlRayGenSet1f(program.rayGen,"cosOuter",light->cos_outer);
  owlRayGenSet1f(program.rayGen,"cosInner",light->cos_inner);
  owlRayGenSet3f(program.rayGen,"diskCenter",reinterpret_cast<const owl3f&>(program.sceneCenter));
  owlRayGenSet1f(program.rayGen,"diskRadius",program.sceneRadius);

  if (causticsMode) {
    owlRayGenSetBuffer(program.rayGen,"photons",program.causticsPhotonsBuffer);
    owlRayGenSetBuffer(program.rayGen,"photonsCount",program.causticsPhotonsCount);
  } else {
    owlRayGenSetBuffer(program.rayGen,"photons",program.photonsBuffer);
    owlRayGenSetBuffer(program.rayGen,"photonsCount",program.photonsCount);
  }
  // savePhoton stores Phi/totalPhotons, so totalPhotons must be THIS light's launch
  // count (not the global budget) for the per-photon flux to be correct.
  owlRayGenSet1i(program.rayGen, "totalPhotons", initialPhotons);

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

// Per-light emission factor: converts `power` into total emitted flux Phi.
//   point       Phi = 4*pi * I             (full sphere)
//   spot        Phi = 2*pi*(1-cosOuter)*I  (outer cone solid angle)
//   directional Phi = pi * R^2 * E         (disk area; E is irradiance)
static float emissionFactor(const Program &program, const PointLight *light) {
  constexpr float PI_F = 3.14159265358979f;
  switch (light->type) {
    case LIGHT_SPOT:        return 2.f * PI_F * (1.f - light->cos_outer);
    case LIGHT_DIRECTIONAL: return PI_F * program.sceneRadius * program.sceneRadius;
    default:                return 4.f * PI_F; // point
  }
}

// Scalar emitted flux of a light (sum of its RGB flux). Used to split the photon budget
// across lights in consistent units (W), which raw `power` is NOT — a directional's
// irradiance (~units) and a point's intensity (~1e5) are different quantities.
static float lightFluxScalar(const Program &program, const PointLight *light) {
  return emissionFactor(program, light) * (light->power.x + light->power.y + light->power.z);
}

void computeSceneBounds(Program &program) {
  owl::vec3f lo(1e30f), hi(-1e30f);
  for (const auto* model : program.world->models)
    for (const auto& v : model->mesh->vertices) {
      lo.x = fminf(lo.x, v.x); lo.y = fminf(lo.y, v.y); lo.z = fminf(lo.z, v.z);
      hi.x = fmaxf(hi.x, v.x); hi.y = fmaxf(hi.y, v.y); hi.z = fmaxf(hi.z, v.z);
    }
  if (hi.x < lo.x) { lo = owl::vec3f(0.f); hi = owl::vec3f(0.f); } // no geometry
  program.sceneCenter = 0.5f * (lo + hi);
  program.sceneRadius = 0.5f * length(hi - lo);
  if (program.sceneRadius <= 0.f) program.sceneRadius = 1.f;
}

void computePhotonsPerWatt(Program &program) {
  // Budget split is proportional to each light's emitted flux Phi (consistent W across
  // light types), so `photonsPerWatt` here is really "photons per unit flux".
  float totalFlux = 0.f;
  for (const auto* light : program.world->lights)
    totalFlux += lightFluxScalar(program, light);

  program.photonsPerWatt = totalFlux > 0.f ? program.castedDiffusePhotons / totalFlux : 0.f;
  program.causticsPhotonsPerWatt = totalFlux > 0.f ? program.castedCausticsPhotons / totalFlux : 0.f;
}

void runNormal(Program &program, const std::string &output_filename) {
  LOG("launching normal photons ...")

  // One launch per light; each appends into the shared buffer (photonsCount is an
  // atomic counter, cleared once in initPhotonBuffers), so totals accumulate.
  for (const auto* light : program.world->lights)
    runPointLightRayGen(program, light, false);

  LOG("done with launch, building + writing normal photon kd-tree ...")
  auto *fb = static_cast<const EmittedPhoton*>(owlBufferGetPointer(program.photonsBuffer, 0));
  auto count = *(int*)owlBufferGetPointer(program.photonsCount, 0);

  LOG("normal photon count: " << count)
  PhotonFileManager::saveKdTreeToFile(fb, count, output_filename, PhotonFileFormat::BINARY);
}

void runCaustics(Program &program, const std::string &output_filename) {
  LOG("launching caustic photons ...")

  for (const auto* light : program.world->lights)
    runPointLightRayGen(program, light, true);

  LOG("done with launch, building + writing caustic photon kd-tree ...")
  auto *fb = static_cast<const EmittedPhoton*>(owlBufferGetPointer(program.causticsPhotonsBuffer, 0));
  auto count = *(int*)owlBufferGetPointer(program.causticsPhotonsCount, 0);

  LOG("caustic photon count: " << count)
  PhotonFileManager::saveKdTreeToFile(fb, count, output_filename, PhotonFileFormat::BINARY);
}

int main(int ac, char **av)
{
  LOG("Starting up...");

  Program program;
  program.owlContext = owlContextCreate(nullptr,1);
  program.owlModule = owlModuleCreate(program.owlContext, photonEmitter_ptx);
  owlContextSetRayTypeCount(program.owlContext, 1);

  LOG("Loading Config file...")

  const std::string scene_name = (ac > 1) ? av[1] : "sponza";
  LOG("Scene: " << scene_name)

  const auto t_load_start = std::chrono::steady_clock::now();
  const auto loader = new Mitsuba3Loader(scene_name);
  program.world = loader->load();
  const auto t_load_end = std::chrono::steady_clock::now();

  auto normal_photons_filename = "normal_photons.kdt";
  auto caustic_photons_filename = "caustic_photons.kdt";
  program.castedDiffusePhotons = program.world->casted_diffuse_photons;
  program.castedCausticsPhotons = program.world->casted_caustic_photons;
  program.maxDepth = 10;

  LOG_OK("Loaded world in "
    << std::chrono::duration_cast<std::chrono::milliseconds>(t_load_end - t_load_start).count()
    << " ms (" << program.world->models.size() << " models)")

  const auto t_bvh_start = std::chrono::steady_clock::now();
  program.geometryData = loadGeometry(program.owlContext, program.world);
  const auto t_bvh_end = std::chrono::steady_clock::now();
  LOG("BVH built in "
    << std::chrono::duration_cast<std::chrono::milliseconds>(t_bvh_end - t_bvh_start).count()
    << " ms")

  owlGeomTypeSetClosestHit(program.geometryData.trianglesGeomType, 0, program.owlModule,"triangleMeshClosestHit");
  owlMissProgCreate(program.owlContext, program.owlModule, "miss", 0, nullptr, -1);

  computeSceneBounds(program);
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
