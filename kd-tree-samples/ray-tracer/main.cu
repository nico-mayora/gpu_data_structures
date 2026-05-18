#include <iostream>

#include "viewer.cuh"
#include "cuda/pathTracer.cuh"
#include "../common/data/loader/mitsuba3.cuh"
#include "../common/data/photon/photon-file-manager.cuh"

int main()
{
    std::cout << "Start!\n";
    // TODO: pass scene name as argv
    const auto loader = new Mitsuba3Loader("cornell-box");
    const auto world = loader->load();
    PhotonFileManager::loadKdTreeFromFile("normal_photons.txt",
                                          world->photon_map,
                                          world->num_photons,
                                          PhotonFileFormat::TEXT);
    PhotonFileManager::loadKdTreeFromFile("caustic_photons.txt",
                                      world->caustic_map,
                                      world->num_caustic,
                                      PhotonFileFormat::TEXT);

    const int parallelThreads = world->cam->image.resolution.x * world->cam->image.resolution.y;

    cudaMalloc(reinterpret_cast<void**>(&world->heapPhotonAddr), sizeof(uint64_t) * parallelThreads * K_GLOBAL_PHOTONS);
    cudaMalloc(reinterpret_cast<void**>(&world->heapCausticAddr), sizeof(uint64_t) * parallelThreads * K_CAUSTIC_PHOTONS);

    Viewer viewer(world);
    viewer.enableFlyMode();

    std::cout << "Launching...\n";
    viewer.showAndRun();
}
