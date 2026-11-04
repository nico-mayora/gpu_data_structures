#pragma once

#include <vector>
#include <string>
#include "../world.cuh"

enum class PhotonFileFormat {
    BINARY,
    TEXT
};

class PhotonFileManager {
public:
    static bool saveKdTreeToFile(const Photon* photons, int count,
                                 const std::string& filename,
                                 PhotonFileFormat format = PhotonFileFormat::BINARY);

    static bool loadKdTreeFromFile(const std::string& filename,
                                   Photon*& photon_ptr, PhotonCoord*& coord_ptr,
                                   int& photon_count,
                                   PhotonFileFormat format = PhotonFileFormat::BINARY);

private:
    static std::vector<Photon> buildTreeHost(const std::vector<Photon>& emitted);

    static bool writeBinary(const std::vector<Photon>& tree, const std::string& filename);
    static std::vector<Photon> readBinary(const std::string& filename);

    static bool writeText(const std::vector<Photon>& photons, const std::string& filename);
    static std::vector<Photon> readText(const std::string& filename);
};
