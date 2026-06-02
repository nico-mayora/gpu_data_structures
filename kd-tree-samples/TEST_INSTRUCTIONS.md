# Test Plan Instructions

## Overview
This package contains everything needed to run the photon-mapping benchmark on a test PC.

## What is included
- **10 path-tracer executables** with different compile-time K values:
  - `pathTracer_kglobal_4.exe` through `pathTracer_kglobal_256.exe` — varying global photon gather count (K_GLOBAL_PHOTONS = 4, 16, 64, 128, 256), caustic K fixed at 1
  - `pathTracer_kcaustic_4.exe` through `pathTracer_kcaustic_256.exe` — varying caustic photon gather count (K_CAUSTIC_PHOTONS = 4, 16, 64, 128, 256), global K fixed at 24
- **photonEmitter.exe** — generates photon maps for a scene
- **run_benchmark.bat** — Windows batch script that runs all 10 executables and prints a results table
- **scenes/** — scene assets (cornell-box, sponza, water-caustic)
- **photon_maps/** — pre-generated photon map files (`.kdt`)
- **owl.dll** — required runtime library

## Folder structure expected on test PC
The executables and batch script expect to be in the same folder as `scenes/` and `photon_maps/`:
```
kd-tree-samples/
  pathTracer_kglobal_4.exe
  ...
  photonEmitter.exe
  run_benchmark.bat
  owl.dll
  scenes/
    cornell-box/
    sponza/
    water-caustic/
  photon_maps/
    <generated .kdt files>
```

## Running the benchmark script
Open a Command Prompt or PowerShell in the folder containing the executables and run:
```bat
run_benchmark.bat <scene> <normal_photons> <caustic_photons>
```

Example:
```bat
run_benchmark.bat cornell-box 1000000 100000
```

The script will:
1. Run each of the 10 path-tracer executables in benchmark mode
2. Each executable renders 10 frames, discards the first (warm-up), and averages the remaining 9
3. Captures the average frame time (ms)
4. Prints and saves a results table to `results_<scene>_n<normal>_c<caustic>.txt`

If an executable crashes (e.g., OOM due to a very large K) or fails to produce a benchmark result, the table shows **DNF** for that run and the script continues with the remaining variants.

## Generating additional photon maps
If you need photon maps with different counts, run:
```bat
photonEmitter.exe <scene> <normal_photons> <caustic_photons>
```

Example:
```bat
photonEmitter.exe sponza 5000000 100
```

Output files are automatically saved to `photon_maps/<scene>_normal_<count>.kdt` and `photon_maps/<scene>_caustic_<count>.kdt`.

## Running a single path tracer manually
```bat
pathTracer_kglobal_64.exe <scene> <normal_photons> <caustic_photons> [benchmark]
```

Omit `benchmark` for interactive mode (opens a window).

## Photon counts available
Pre-generated maps exist for:
- **Scenes:** cornell-box, sponza, water-caustic
- **Normal photons:** 1 000 000, 5 000 000, 10 000 000
- **Caustic photons:** 1 000 000, 5 000 000, 10 000 000

Mismatch is allowed (e.g., 1M normal + 5M caustic).

## Rebuilding from source (if needed)
If the pre-built executables don't work on your GPU, rebuild them from source.

### Prerequisites
- CMake ≥ 3.27
- Visual Studio 2022 with C++ and CUDA workloads
- CUDA Toolkit 12.x
- OptiX SDK 8.0 (set `OptiX_INSTALL_DIR` environment variable)

### Configure
Open **x64 Native Tools Command Prompt for VS 2022** and run:
```bat
cd kd-tree-samples
rmdir /s /q cmake-build-test 2>nul
mkdir cmake-build-test
cd cmake-build-test
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
```

### Build all executables
```bat
cd kd-tree-samples\cmake-build-test
msbuild kd-tree_samples.sln /p:Configuration=Release /p:Platform=x64 /m
```

Or build only the targets you need:
```bat
msbuild kd-tree_samples.sln /p:Configuration=Release /p:Platform=x64 /m /t:photonEmitter
msbuild kd-tree_samples.sln /p:Configuration=Release /p:Platform=x64 /m /t:pathTracer_kglobal_4;pathTracer_kglobal_16;pathTracer_kglobal_64;pathTracer_kglobal_128;pathTracer_kglobal_256;pathTracer_kcaustic_4;pathTracer_kcaustic_16;pathTracer_kcaustic_64;pathTracer_kcaustic_128;pathTracer_kcaustic_256
```

### Copy binaries to the package root
```bat
cd kd-tree-samples
copy cmake-build-test\Release\photonEmitter.exe .
copy cmake-build-test\Release\pathTracer_kglobal_*.exe .
copy cmake-build-test\Release\pathTracer_kcaustic_*.exe .
copy cmake-build-test\common\externals\owl\owl\Release\owl.dll .
```

### Regenerate all photon maps
Run these from the `kd-tree-samples` folder:
```bat
photonEmitter.exe cornell-box 1000000 1000000
photonEmitter.exe cornell-box 1000000 5000000
photonEmitter.exe cornell-box 1000000 10000000
photonEmitter.exe cornell-box 5000000 1000000
photonEmitter.exe cornell-box 5000000 5000000
photonEmitter.exe cornell-box 5000000 10000000
photonEmitter.exe cornell-box 10000000 1000000
photonEmitter.exe cornell-box 10000000 5000000
photonEmitter.exe cornell-box 10000000 10000000

photonEmitter.exe sponza 1000000 1000000
photonEmitter.exe sponza 1000000 5000000
photonEmitter.exe sponza 1000000 10000000
photonEmitter.exe sponza 5000000 1000000
photonEmitter.exe sponza 5000000 5000000
photonEmitter.exe sponza 5000000 10000000
photonEmitter.exe sponza 10000000 1000000
photonEmitter.exe sponza 10000000 5000000
photonEmitter.exe sponza 10000000 10000000

photonEmitter.exe water-caustic 1000000 1000000
photonEmitter.exe water-caustic 1000000 5000000
photonEmitter.exe water-caustic 1000000 10000000
photonEmitter.exe water-caustic 5000000 1000000
photonEmitter.exe water-caustic 5000000 5000000
photonEmitter.exe water-caustic 5000000 10000000
photonEmitter.exe water-caustic 10000000 1000000
photonEmitter.exe water-caustic 10000000 5000000
photonEmitter.exe water-caustic 10000000 10000000
```

*Note: sponza has no caustic-producing geometry, so all its caustic files will be empty (valid 16-byte headers). cornell-box and water-caustic 10M/10M may OOM on GPUs with limited memory; generate those on the test PC if they failed on the build machine.*

## Notes
- The `K` values are compile-time constants (template parameters), which is why we built 10 separate executables.
- Benchmark mode creates an **invisible** window and disables VSync so GPU measurements are not capped by the display refresh rate.
- Make sure `owl.dll` is in the same directory as the executables or in your system PATH.
- If a photon map file is missing, the path tracer will still load but caustic/global illumination will be disabled for that map.
