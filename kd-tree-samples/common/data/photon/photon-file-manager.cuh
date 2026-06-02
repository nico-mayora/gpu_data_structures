#pragma once

#include <vector>
#include <string>
#include "../world.cuh"

enum class PhotonFileFormat {
    BINARY,
    TEXT
};

// Persists/loads the photon kd-tree shared between the two binaries.
//
// BINARY (default, production): the photon mapper builds the left-balanced kd-tree and
// writes the resulting Photon buffer verbatim. The path tracer memcpy's that same buffer
// onto the device and is ready to query — no rebuild.
//
// TEXT (debugging only): a human-readable dump of the *raw* (unbuilt) photons. On load the
// kd-tree is rebuilt, so it round-trips, but it's slower and lossy on float formatting.
class PhotonFileManager {
public:
    // Called by the photon mapper. For BINARY this builds the kd-tree from the freshly
    // emitted photons (on the device) and dumps the built buffer; for TEXT it dumps the
    // raw photons unbuilt.
    static bool saveKdTreeToFile(const EmittedPhoton* photons, int count,
                                 const std::string& filename,
                                 PhotonFileFormat format = PhotonFileFormat::BINARY);

    // Called by the path tracer. For BINARY it uploads the prebuilt tree as-is; for TEXT it
    // uploads the raw photons and builds the tree on load. Either way it also derives the
    // coords-only parallel array used by the traversal hot path.
    static bool loadKdTreeFromFile(const std::string& filename,
                                   Photon*& photon_ptr, PhotonCoord*& coord_ptr,
                                   int& photon_count,
                                   PhotonFileFormat format = PhotonFileFormat::BINARY);

private:
    // Converts emitted photons -> Photon records, then builds the kd-tree on the device and
    // returns the built buffer on the host.
    static std::vector<Photon> buildTreeHost(const std::vector<EmittedPhoton>& emitted);

    static bool writeBinary(const std::vector<Photon>& tree, const std::string& filename);
    static std::vector<Photon> readBinary(const std::string& filename);

    static bool writeText(const std::vector<EmittedPhoton>& photons, const std::string& filename);
    static std::vector<Photon> readText(const std::string& filename);
};
