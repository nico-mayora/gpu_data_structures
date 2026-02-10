#include <iostream>

#include "viewer.cuh"
#include "../common/data/loader/mitsuba3.cuh"
#include "../common/data/photon/photon-file-manager.cuh"

// This needs to be the largest number of K-photons between K_GLOBAL and K_CAUSTIC
#define K_PHOTONS 500

int main()
{
    std::cout << "start!\n";
    // TODO: pass scene name as argv
    const auto loader = new Mitsuba3Loader("water-caustic");
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

    const size_t heap_size = parallelThreads * K_PHOTONS;
    cudaMalloc(reinterpret_cast<void**>(&world->heap_indices), sizeof(size_t) * heap_size);
    cudaMalloc(reinterpret_cast<void**>(&world->heap_distances), sizeof(float) * heap_size);

    Viewer viewer(world);
    viewer.enableFlyMode();

    std::cout << "Launching...\n";
    viewer.showAndRun();
}
