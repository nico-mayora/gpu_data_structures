#include <iostream>

#include "viewer.cuh"
#include "../common/data/loader/mitsuba3.cuh"
#include "../common/data/photon/photon-file-manager.cuh"

int main()
{
    std::cout << "Start!\n";
    // TODO: pass scene name as argv
    const auto loader = new Mitsuba3Loader("sponza");
    const auto world = loader->load();
    PhotonFileManager::loadKdTreeFromFile("normal_photons.txt",
                                          world->photon_map,
                                          world->photon_coords,
                                          world->num_photons,
                                          PhotonFileFormat::TEXT);
    PhotonFileManager::loadKdTreeFromFile("caustic_photons.txt",
                                      world->caustic_map,
                                      world->caustic_coords,
                                      world->num_caustic,
                                      PhotonFileFormat::TEXT);

    Viewer viewer(world);
    viewer.enableFlyMode();

    std::cout << "Launching...\n";
    viewer.showAndRun();
}
