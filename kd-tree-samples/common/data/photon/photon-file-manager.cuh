#pragma once

#include <vector>
#include <string>
#include "../world.cuh"

enum class PhotonFileFormat {
    BINARY,
    TEXT
};

class   PhotonFileManager {
public:
    static bool savePhotonsToFile(const EmittedPhoton* photons, int count,
                                  const std::string& filename,
                                  PhotonFileFormat format = PhotonFileFormat::TEXT);
    static bool savePhotonsToFile(const std::vector<EmittedPhoton>& photons,
                                  const std::string& filename, 
                                  PhotonFileFormat format = PhotonFileFormat::TEXT);
    
    static std::vector<Photon> loadPhotonsFromFile(const std::string& filename, 
                                                   PhotonFileFormat format = PhotonFileFormat::TEXT);
    static bool loadKdTreeFromFile(const std::string&, Photon*&, int&, PhotonFileFormat);
private:
    static bool saveBinary(const std::vector<EmittedPhoton>& photons, const std::string& filename);
    static bool saveText(const std::vector<EmittedPhoton>& photons, const std::string& filename);
    
    static std::vector<Photon> loadBinary(const std::string& filename);
    static std::vector<Photon> loadText(const std::string& filename);
};