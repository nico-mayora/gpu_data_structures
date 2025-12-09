#include <iostream>

#include "viewer.cuh"
#include "../common/data/loader/mitsuba3.cuh"
#include "../common/data/photon/photon-file-manager.cuh"

#define K_PHOTONS 1000

int main()
{
    std::cout << "start!\n";
    // TODO: pass scene name as argv
    const auto loader = new Mitsuba3Loader("cornell-box");
    const auto world = loader->load();
    PhotonFileManager::loadKdTreeFromFile("cornell-box-photons.txt",
                                          world->photon_map,
                                          world->num_photons,
                                          PhotonFileFormat::TEXT);

    const int parallelThreads = world->cam->image.resolution.x *
                            world->cam->image.resolution.y;

    const size_t heap_size = parallelThreads * K_PHOTONS;
    cudaMalloc(reinterpret_cast<void**>(&world->heap_indices), sizeof(size_t) * heap_size);
    cudaMalloc(reinterpret_cast<void**>(&world->heap_distances), sizeof(float) * heap_size);


    auto lf = world->cam->lookFrom;
    std::cout << "cam: " << lf.x << " " << lf.y << " " << lf.z << '\n';

    Viewer viewer(world);
    viewer.enableFlyMode();

    std::cout << "Launching...\n";
    viewer.showAndRun();
}
