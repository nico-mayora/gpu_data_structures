#include <iostream>

#include "viewer.cuh"
#include "../common/data/loader/mitsuba3.cuh"
#include "../common/data/photon/photon-file-manager.cuh"

int main(int ac, char **av)
{
    std::cout << "Start!\n";
    const std::string scene_name = (ac > 1) ? av[1] : "sponza";
    std::cout << "Scene: " << scene_name << "\n";
    const auto loader = new Mitsuba3Loader(scene_name);
    const auto world = loader->load();
    PhotonFileManager::loadKdTreeFromFile("normal_photons.kdt",
                                          world->photon_map,
                                          world->photon_coords,
                                          world->num_photons,
                                          PhotonFileFormat::BINARY);
    PhotonFileManager::loadKdTreeFromFile("caustic_photons.kdt",
                                      world->caustic_map,
                                      world->caustic_coords,
                                      world->num_caustic,
                                      PhotonFileFormat::BINARY);

    Viewer viewer(world, scene_name);
    viewer.enableFlyMode();

    std::cout << "Launching...\n";
    viewer.showAndRun();
}
