#include <iostream>

#include "viewer.cuh"
#include "../common/data/loader/mitsuba3.cuh"
#include "../common/data/photon/photon-file-manager.cuh"

// Extract float3 positions from a device Photon array (after tree build).
// Copies back to host, extracts positions, uploads to device.
static float3* extract_float3_positions(Photon *d_photons, int count) {
    auto *h_photons = new Photon[count];
    cudaMemcpy(h_photons, d_photons, sizeof(Photon) * count, cudaMemcpyDeviceToHost);

    auto *h_positions = new float3[count];
    for (int i = 0; i < count; i++)
        h_positions[i] = make_float3(h_photons[i].coords[0],
                                      h_photons[i].coords[1],
                                      h_photons[i].coords[2]);

    float3 *d_positions = nullptr;
    cudaMalloc(&d_positions, sizeof(float3) * count);
    cudaMemcpy(d_positions, h_positions, sizeof(float3) * count, cudaMemcpyHostToDevice);

    delete[] h_photons;
    delete[] h_positions;
    return d_positions;
}

int main()
{
    std::cout << "Start!\n";
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

    // Extract float3 position arrays for cudaKDTree queries.
    // The Photon arrays are already reordered as KD-trees by the builder,
    // so the extracted float3 arrays are valid implicit KD-trees.
    world->photon_positions = extract_float3_positions(world->photon_map, world->num_photons);
    world->caustic_positions = extract_float3_positions(world->caustic_map, world->num_caustic);

    Viewer viewer(world);
    viewer.enableFlyMode();

    std::cout << "Launching...\n";
    viewer.showAndRun();
}
