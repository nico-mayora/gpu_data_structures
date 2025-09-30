#include "photon-file-manager.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <stdexcept>

#include "../cuda/kdtree/builder.cuh"

bool PhotonFileManager::savePhotonsToFile(const Photon* photons, int count,
                                          const std::string& filename,
                                          PhotonFileFormat format) {
    if (count <= 0 || photons == nullptr) {
        std::cerr << "Warning: No photons to save." << std::endl;
        return false;
    }

    std::cout << "Saving " << count << " photons to: " << filename << std::endl;

    std::vector<Photon> photonVec(photons, photons + count);

    switch (format) {
        case PhotonFileFormat::BINARY:
            return saveBinary(photonVec, filename);
        case PhotonFileFormat::TEXT:
            return saveText(photonVec, filename);
        default:
            std::cerr << "Error: Unknown photon file format." << std::endl;
            return false;
    }
}

bool PhotonFileManager::savePhotonsToFile(const std::vector<Photon>& photons, 
                                          const std::string& filename, 
                                          PhotonFileFormat format) {
    if (photons.empty()) {
        std::cerr << "Warning: No photons to save." << std::endl;
        return false;
    }
    
    std::cout << "Saving " << photons.size() << " photons to: " << filename << std::endl;
    
    switch (format) {
        case PhotonFileFormat::BINARY:
            return saveBinary(photons, filename);
        case PhotonFileFormat::TEXT:
            return saveText(photons, filename);
        default:
            std::cerr << "Error: Unknown photon file format." << std::endl;
            return false;
    }
}

std::vector<Photon> PhotonFileManager::loadPhotonsFromFile(const std::string& filename,
                                                           PhotonFileFormat format) {
    if (!std::filesystem::exists(filename)) {
        std::cerr << "Error: File does not exist: " << filename << std::endl;
        return {};
    }
    
    switch (format) {
        case PhotonFileFormat::BINARY:
            return loadBinary(filename);
        case PhotonFileFormat::TEXT:
            return loadText(filename);
        default:
            std::cerr << "Error: Unknown photon file format." << std::endl;
            return {};
    }
}

bool PhotonFileManager::saveBinary(const std::vector<Photon>& photons, const std::string& filename) {
    throw std::runtime_error("Binary format save not implemented yet");
}

bool PhotonFileManager::saveText(const std::vector<Photon>& photons, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return false;
    }
    
    file << "# PHOTON_MAP_V1" << std::endl;
    file << "# PhotonCount: " << photons.size() << std::endl;
    file << "# Format: x y z r g b power_r power_g power_b dir_x dir_y dir_z" << std::endl;
    
    // Write photon data
    for (const auto& photon : photons) {
        file << photon.coords[0] << " " << photon.coords[1] << " " << photon.coords[2] << " "
             << photon.colour[0] << " " << photon.colour[1] << " " << photon.colour[2] << " "
             << photon.power[0] << " " << photon.power[1] << " " << photon.power[2] << " "
             << photon.dir[0] << " " << photon.dir[1] << " " << photon.dir[2] << std::endl;
    }
    
    file.close();
    std::cout << "Successfully saved " << photons.size() << " photons to text file: " << filename << std::endl;
    return true;
}


std::vector<Photon> PhotonFileManager::loadBinary(const std::string& filename) {
    throw std::runtime_error("Binary format load not implemented yet");
}

std::vector<Photon> PhotonFileManager::loadText(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for reading: " << filename << std::endl;
        return {};
    }
    
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
    
    file.close();
    std::cout << "Successfully loaded " << photons.size() << " photons from text file: " << filename << std::endl;
    return photons;
}

bool PhotonFileManager::loadKdTreeFromFile(const std::string &filename,
                                           Photon *&photon_ptr,
                                           int &photon_count,
                                           PhotonFileFormat format) {
    try {
        const auto loadedPhotons = loadPhotonsFromFile(filename, format);
        photon_count = static_cast<int>(loadedPhotons.size());

        cudaMalloc(reinterpret_cast<void**>(&photon_ptr),sizeof(Photon) * photon_count);
        cudaMemcpy(photon_ptr, loadedPhotons.data(), sizeof(Photon) * photon_count, cudaMemcpyHostToDevice);

        build_kd_tree<Photon>(photon_ptr, photon_count);
    } catch (const std::exception& e) {
        std::cerr << "Error loading KD-Tree from file: " << e.what() << std::endl;
        return false;
    }
    return true;
}