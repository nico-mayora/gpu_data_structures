#include <iostream>

#include "viewer.cuh"
#include "loader/mitsuba3.cuh"
#include "writer/photon-file-manager.cuh"

void main()
{
    std::cout << "start!\n";
    // TODO: pass scene name as argv
    const auto loader = new Mitsuba3Loader("cornell-box");
    const auto world = loader->load();
    PhotonFileManager::loadKdTreeFromFile("cornell-box-photons.txt",
                                          world->photon_map,
                                          world->num_photons,
                                          PhotonFileFormat::TEXT);

    auto lf = world->cam->lookFrom;
    std::cout << "cam: " << lf.x << " " << lf.y << " " << lf.z << '\n';

    Viewer viewer(world);
    viewer.enableFlyMode();

    std::cout << "Launching...\n";
    viewer.showAndRun();
}
