#pragma once

#include <vector>
#include <string>
#include "../cuda/photonEmitter.cuh"

enum class PhotonFileFormat {
    BINARY,
    TEXT
};

class PhotonFileManager {
public:
    static bool savePhotonsToFile(const Photon* photons, int count,
                                  const std::string& filename,
                                  PhotonFileFormat format = PhotonFileFormat::TEXT);
    static bool savePhotonsToFile(const std::vector<Photon>& photons, 
                                  const std::string& filename, 
                                  PhotonFileFormat format = PhotonFileFormat::TEXT);
    
    static std::vector<Photon> loadPhotonsFromFile(const std::string& filename, 
                                                   PhotonFileFormat format = PhotonFileFormat::TEXT);
private:
    static bool saveBinary(const std::vector<Photon>& photons, const std::string& filename);
    static bool saveText(const std::vector<Photon>& photons, const std::string& filename);
    
    static std::vector<Photon> loadBinary(const std::string& filename);
    static std::vector<Photon> loadText(const std::string& filename);
};