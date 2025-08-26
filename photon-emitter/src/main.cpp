#include <iostream>
#include <chrono>

#include "photon-emitter.h"
#include "photon-file-manager.h"
#include "loader/mitsuba3.h"

int main()
{
    // TODO: pass scene name as argv
    const auto loader = new Mitsuba3Loader("cornell-box");
    const auto world = loader->load();

    PhotonEmitter emitter(world);
    emitter.clearPhotons();
    
    const int GLOBAL_MAX_PHOTONS = 1000;
    const int MAX_DEPTH = 10;
    
    float totalPower = 0.0f;
    for (const auto* light : world->light_sources) {
        // Use luminance approximation for power: 0.299*R + 0.587*G + 0.114*B
        float lightPower = 0.299f * light->radiance.x + 0.587f * light->radiance.y + 0.114f * light->radiance.z;
        totalPower += lightPower;
    }
    
    if (totalPower <= 0.0f) {
        std::cerr << "Error: No lights found or total light power is zero!" << std::endl;
        return -1;
    }
    
    std::cout << "Found " << world->light_sources.size() << " lights with total power: " << totalPower << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < world->light_sources.size(); ++i) {
        const auto* light = world->light_sources[i];
        
        float lightPower = 0.299f * light->radiance.x + 0.587f * light->radiance.y + 0.114f * light->radiance.z;
        int photonsForThisLight = static_cast<int>((lightPower / totalPower) * GLOBAL_MAX_PHOTONS);
        
        std::cout << "Light " << (i+1) << ": Power=" << lightPower << " (" << (100.0f * lightPower / totalPower) << "%), " << "Photons=" << photonsForThisLight << std::endl;
        
        emitter.emit(light, photonsForThisLight, MAX_DEPTH);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Photon mapping completed in " << duration.count() << " ms.\n";

    std::vector<Photon> photons = emitter.getPhotons();
    if (!photons.empty()) {
        std::string baseFilename = "photon_map_" + std::to_string(photons.size()) + ".txt";
        PhotonFileManager::savePhotonsToFile(photons, baseFilename, PhotonFileFormat::TEXT);
    } else {
        std::cout << "No photons were generated.\n";
    }
    
    std::cout << "Program completed successfully.\n";
    return 0;
}
