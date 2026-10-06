#include "photon-file-manager.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <cstdint>
#include <cstring>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "../../kdtree/builder.cuh"

namespace {
    // Binary file layout: this header, immediately followed by `count` Photon records
    // written verbatim (the already-built, left-balanced kd-tree buffer).
    struct KdTreeFileHeader {
        char     magic[4];        // "KDPM"
        uint32_t version;         // bump on layout change
        uint32_t count;           // number of Photon records that follow
        uint32_t photon_stride;   // sizeof(Photon); guards against struct-layout drift
    };

    constexpr char     KDTREE_MAGIC[4] = {'K', 'D', 'P', 'M'};
    constexpr uint32_t KDTREE_VERSION  = 1;

    Photon emittedToPhoton(const EmittedPhoton& e) {
        Photon p;
        p.coords[0] = e.pos.x; p.coords[1] = e.pos.y; p.coords[2] = e.pos.z;
        p.colour[0] = e.color.x; p.colour[1] = e.color.y; p.colour[2] = e.color.z;
        // EmittedPhoton::power is a vestigial int; mirror the legacy text mapping.
        p.power[0] = p.power[1] = p.power[2] = static_cast<float>(e.power);
        p.dir[0] = e.dir.x; p.dir[1] = e.dir.y; p.dir[2] = e.dir.z;
        return p;
    }
}

struct ExtractCoords {
    __device__ PhotonCoord operator()(const Photon &p) const {
        PhotonCoord c;
        c.coords[0] = p.coords[0];
        c.coords[1] = p.coords[1];
        c.coords[2] = p.coords[2];
        return c;
    }
};

std::vector<Photon> PhotonFileManager::buildTreeHost(const std::vector<EmittedPhoton>& emitted) {
    std::vector<Photon> photons;
    photons.reserve(emitted.size());
    for (const auto& e : emitted) photons.push_back(emittedToPhoton(e));

    const int count = static_cast<int>(photons.size());
    if (count == 0) return photons;

    // Build the left-balanced kd-tree in place on the device, then read it back.
    Photon* d_photons = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_photons), sizeof(Photon) * count);
    cudaMemcpy(d_photons, photons.data(), sizeof(Photon) * count, cudaMemcpyHostToDevice);

    build_kd_tree<Photon>(d_photons, count);

    cudaMemcpy(photons.data(), d_photons, sizeof(Photon) * count, cudaMemcpyDeviceToHost);
    cudaFree(d_photons);

    return photons;
}

bool PhotonFileManager::saveKdTreeToFile(const EmittedPhoton* photons, int count,
                                         const std::string& filename,
                                         PhotonFileFormat format) {
    if (count <= 0 || photons == nullptr) {
        std::cerr << "Warning: No photons to save." << std::endl;
        return false;
    }

    const std::vector<EmittedPhoton> photonVec(photons, photons + count);

    switch (format) {
        case PhotonFileFormat::BINARY: {
            std::cout << "Building kd-tree from " << count << " photons ..." << std::endl;
            const std::vector<Photon> tree = buildTreeHost(photonVec);
            return writeBinary(tree, filename);
        }
        case PhotonFileFormat::TEXT:
            return writeText(photonVec, filename);
        default:
            std::cerr << "Error: Unknown photon file format." << std::endl;
            return false;
    }
}

bool PhotonFileManager::loadKdTreeFromFile(const std::string &filename,
                                           Photon *&photon_ptr,
                                           PhotonCoord *&coord_ptr,
                                           int &photon_count,
                                           PhotonFileFormat format) {
    photon_ptr = nullptr;
    coord_ptr = nullptr;
    photon_count = 0;

    if (!std::filesystem::exists(filename)) {
        std::cerr << "Error: File does not exist: " << filename << std::endl;
        return false;
    }

    try {
        std::vector<Photon> photons;
        bool already_built = false;

        switch (format) {
            case PhotonFileFormat::BINARY:
                photons = readBinary(filename);   // already a built kd-tree
                already_built = true;
                break;
            case PhotonFileFormat::TEXT:
                photons = readText(filename);     // raw photons, build below
                break;
            default:
                std::cerr << "Error: Unknown photon file format." << std::endl;
                return false;
        }

        photon_count = static_cast<int>(photons.size());
        if (photon_count == 0) {
            std::cerr << "Warning: loaded 0 photons from " << filename << std::endl;
            return true;
        }

        cudaMalloc(reinterpret_cast<void**>(&photon_ptr), sizeof(Photon) * photon_count);
        cudaMemcpy(photon_ptr, photons.data(), sizeof(Photon) * photon_count, cudaMemcpyHostToDevice);

        // Text dumps store the raw (unbuilt) photons; binary dumps are the built tree.
        if (!already_built)
            build_kd_tree<Photon>(photon_ptr, photon_count);

        // Extract the coords-only parallel array. The kd-tree traversal hits these
        // per node — 12 bytes vs 48 bytes is ~4x less bandwidth on the hot path.
        cudaMalloc(reinterpret_cast<void**>(&coord_ptr), sizeof(PhotonCoord) * photon_count);
        thrust::transform(thrust::device, photon_ptr, photon_ptr + photon_count,
                          coord_ptr, ExtractCoords());
    } catch (const std::exception& e) {
        std::cerr << "Error loading KD-Tree from file: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool PhotonFileManager::writeBinary(const std::vector<Photon>& tree, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    KdTreeFileHeader header{};
    std::memcpy(header.magic, KDTREE_MAGIC, sizeof(header.magic));
    header.version = KDTREE_VERSION;
    header.count = static_cast<uint32_t>(tree.size());
    header.photon_stride = static_cast<uint32_t>(sizeof(Photon));

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(tree.data()),
               static_cast<std::streamsize>(sizeof(Photon) * tree.size()));

    if (!file) {
        std::cerr << "Error: Failed while writing binary kd-tree: " << filename << std::endl;
        return false;
    }
    std::cout << "Saved kd-tree (" << tree.size() << " photons) to binary file: "
              << filename << std::endl;
    return true;
}

std::vector<Photon> PhotonFileManager::readBinary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file for reading: " + filename);

    KdTreeFileHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file)
        throw std::runtime_error("Truncated kd-tree header in: " + filename);

    if (std::memcmp(header.magic, KDTREE_MAGIC, sizeof(header.magic)) != 0)
        throw std::runtime_error("Bad magic (not a kd-tree dump): " + filename);
    if (header.version != KDTREE_VERSION)
        throw std::runtime_error("Unsupported kd-tree file version in: " + filename);
    if (header.photon_stride != sizeof(Photon))
        throw std::runtime_error("Photon struct layout mismatch in: " + filename);

    std::vector<Photon> photons(header.count);
    file.read(reinterpret_cast<char*>(photons.data()),
              static_cast<std::streamsize>(sizeof(Photon) * header.count));
    if (!file)
        throw std::runtime_error("Truncated kd-tree body in: " + filename);

    std::cout << "Loaded prebuilt kd-tree (" << photons.size() << " photons) from: "
              << filename << std::endl;
    return photons;
}

bool PhotonFileManager::writeText(const std::vector<EmittedPhoton>& photons, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    file << "# PHOTON_MAP_V1" << std::endl;
    file << "# PhotonCount: " << photons.size() << std::endl;
    file << "# Format: x y z r g b power_r power_g power_b dir_x dir_y dir_z" << std::endl;

    for (const auto& photon : photons) {
        file << photon.pos[0] << " " << photon.pos[1] << " " << photon.pos[2] << " "
             << photon.color[0] << " " << photon.color[1] << " " << photon.color[2] << " "
             << photon.power << " " << photon.power << " " << photon.power << " "
             << photon.dir[0] << " " << photon.dir[1] << " " << photon.dir[2] << std::endl;
    }

    std::cout << "Saved " << photons.size() << " raw photons to text file (debug): "
              << filename << std::endl;
    return true;
}

std::vector<Photon> PhotonFileManager::readText(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file for reading: " + filename);

    std::vector<Photon> photons;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue; // Skip header/comments

        std::istringstream iss(line);
        Photon photon;

        if (iss >> photon.coords[0] >> photon.coords[1] >> photon.coords[2] >>
                  photon.colour[0] >> photon.colour[1] >> photon.colour[2] >>
                  photon.power[0] >> photon.power[1] >> photon.power[2] >>
                  photon.dir[0] >> photon.dir[1] >> photon.dir[2]) {
            photons.push_back(photon);
        }
    }

    std::cout << "Loaded " << photons.size() << " raw photons from text file (debug): "
              << filename << std::endl;
    return photons;
}
