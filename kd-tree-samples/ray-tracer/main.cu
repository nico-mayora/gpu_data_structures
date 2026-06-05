#include <iostream>
#include <string>

#include "viewer.cuh"
#include "../common/data/loader/mitsuba3.cuh"
#include "../common/data/photon/photon-file-manager.cuh"

int main(int ac, char **av)
{
    std::cout << "Start!\n";
    const std::string scene_name = (ac > 1) ? av[1] : "sponza";
    std::cout << "Scene: " << scene_name << "\n";

    int normal_photons = 0;
    int caustic_photons = 0;
    bool benchmark = false;
    bool visible = true;

    if (ac > 2) normal_photons = std::stoi(av[2]);
    if (ac > 3) caustic_photons = std::stoi(av[3]);
    // Remaining args are flags: "benchmark" and/or "visible"
    for (int i = 4; i < ac; ++i) {
        std::string arg(av[i]);
        if (arg == "benchmark") benchmark = true;
        else if (arg == "visible") visible = true;
        else if (arg == "nowindow") visible = false;
    }
    // Default: benchmark hides window unless "visible" is explicitly passed
    if (benchmark) {
        bool explicit_visible = false;
        for (int i = 4; i < ac; ++i) {
            if (std::string(av[i]) == "visible") { explicit_visible = true; break; }
        }
        if (!explicit_visible) visible = false;
    }

    const auto loader = new Mitsuba3Loader(scene_name);
    const auto world = loader->load();

    std::string normal_file = "photon_maps/" + scene_name + "_normal_" + std::to_string(normal_photons) + ".kdt";
    std::string caustic_file = "photon_maps/" + scene_name + "_caustic_" + std::to_string(caustic_photons) + ".kdt";

    std::cout << "Loading normal photons from: " << normal_file << "\n";
    std::cout << "Loading caustic photons from: " << caustic_file << "\n";

    PhotonFileManager::loadKdTreeFromFile(normal_file.c_str(),
                                          world->photon_map,
                                          world->photon_coords,
                                          world->num_photons,
                                          PhotonFileFormat::BINARY);
    PhotonFileManager::loadKdTreeFromFile(caustic_file.c_str(),
                                      world->caustic_map,
                                      world->caustic_coords,
                                      world->num_caustic,
                                      PhotonFileFormat::BINARY);

    std::cout << "Benchmark: " << (benchmark ? "ON" : "OFF")
              << ", Visible: " << (visible ? "ON" : "OFF") << "\n";

    Viewer viewer(world, scene_name, benchmark, visible);

    if (benchmark) {
        viewer.showAndRun([&viewer]() {
            return viewer.benchmarkTimes.size() < 10;
        });
        if (viewer.benchmarkTimes.size() >= 2) {
            float sum = 0.f;
            for (size_t i = 1; i < viewer.benchmarkTimes.size(); ++i)
                sum += viewer.benchmarkTimes[i];
            float avg = sum / float(viewer.benchmarkTimes.size() - 1);
            std::cout << "BENCHMARK_RESULT: " << avg << " ms" << std::endl;
        } else {
            std::cout << "BENCHMARK_RESULT: 0 ms" << std::endl;
        }
    } else {
        viewer.enableFlyMode();
        std::cout << "Launching...\n";
        viewer.showAndRun();
    }
}
