# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

CUDA + OptiX project built with CMake (CUDA/CXX std 20). The OptiX SDK is located via the `OptiX_INSTALL_DIR` environment variable — builds fail without it.

```powershell
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

Three executables are produced:

- `pathTracer` — interactive OWLViewer-based renderer. Loads a scene + photon-map files and renders with photon mapping.
- `photonEmitter` — offline photon-map generator. Writes `normal_photons.txt` and `caustic_photons.txt` to the working directory.
- `kdtree_benchmark` — standalone kd-tree KNN microbenchmark (no OWL/OptiX dependency at runtime).

Binaries expect to run from a `cmake-build-*` directory: the Mitsuba loader resolves scene paths as `..\scenes\<name>\scene_v3.xml`. `photonEmitter` takes the scene name as `argv[1]` (defaulting to `"cornell-box"`); `pathTracer`'s scene name is still hardcoded.

## Runtime pipeline

The renderer is a two-stage offline+online pipeline, not a single binary:

1. **`photonEmitter`** traces photons from the scene's point light through the OptiX BVH, separately for the diffuse global pass and a caustics-only pass (refractive/specular path). Photon counts are tuned via `castedDiffusePhotons` / `castedCausticsPhotons` in `photon-mapper/main.cu`. Output → two text files via `PhotonFileManager`.
2. **`pathTracer`** reads those files back, calls `build_kd_tree` on each photon array, and renders. At render time, the OptiX closest-hit shader does kNN lookups (`K_GLOBAL_PHOTONS=24`, `K_CAUSTIC_PHOTONS=128` — see `ray-tracer/cuda/pathTracer.cuh`) for radiance estimation.

You must run `photonEmitter` before `pathTracer` whenever scene geometry or lighting changes; the photon files are not regenerated automatically.

## Architecture

**`common/kdtree/`** — the data-structure heart of the project. It is consumed by both the path tracer (for photon lookups) and the benchmark (with synthetic point clouds).

- `builder.cuh` (`build_kd_tree<P>`) — builds a left-balanced kd-tree **in place** in a device buffer of `P`. Algorithm: per level `l`, segmented sort the (tag, point) zip iterator along axis `l % P::dimension`, then a `update_tags` kernel reassigns each point to its left/right child segment based on the computed pivot position. Implicit indexing — node `i`'s children are `2i+1`, `2i+2`. No allocations on the tree side.
- `queries.cuh` — iterative (no recursion, no stack) kd-tree traversal. Two result-collector types over the same `get_closest_k_points_in_range` traversal: `FixedQueryResult<K>` does insertion sort (good for tiny K, gives sorted results), `HeapQueryResult<K>` is a max-heap (better for large K, results unsorted). Both pack `(distance, index)` into a `uint64_t` with distance in the high 32 bits so unsigned compares preserve distance order.
- `data.cuh` — generic `Point<DIM>` used by the benchmark.

Any custom point type used with this kd-tree must expose `static constexpr int dimension`, a `float coords[]` array, and `__device__ float dist2(const float*)`. See `Photon`, `PhotonCoord` in `common/data/world.cuh`.

**Photon vs PhotonCoord split** (`common/data/world.cuh`): `Photon` carries position + color + power + direction (48 B); `PhotonCoord` is a parallel coords-only view (12 B). The kd-tree is built/traversed over `PhotonCoord` to avoid wasting ~75% of memory bandwidth on the traversal hot path; full `Photon` records are looked up by index only at the leaves the traversal converges to. Keep the two arrays in lockstep when modifying photon code.

**OWL + embedded PTX.** OptiX device code lives in `ray-tracer/cuda/pathTracer.cu` and `photon-mapper/cuda/photonEmitter.cu`. The `embed_ptx()` CMake function (`cmake/embed_ptx.cmake`) compiles each `.cu` to PTX then runs `bin2c` to embed it as a C symbol (e.g. `extern "C" char pathTracer_ptx[]`) linked into the host binary. To add a new OptiX kernel, add another `embed_ptx()` block in the top-level `CMakeLists.txt` and link the resulting `*_ptx` target to your executable.

**Scene loading** (`common/data/loader/mitsuba3.cu`). The loader consumes a Mitsuba 3 scene XML, resolves `$defaults`, loads referenced `.obj` meshes via tinyobjloader, and produces a `World*` (`common/data/world.cuh`) holding `Model*` (mesh + material), a single `PointLight*`, and a `Camera*`. Materials are simplified to one of `LAMBERTIAN`/`DIELECTRIC`/`CONDUCTOR` (mutually exclusive — combinations from the XML are collapsed).

**Benchmark layout** (`benchmark/`). Compares three strategies for storing per-query KNN result arrays on the GPU:
- `knn_local.cuh` — per-thread local arrays (registers/local memory)
- `knn_global.cuh` — pre-allocated `cudaMalloc` slices indexed by thread ID (this is what the ray tracer uses)
- `knn_shared.cuh` — `__shared__` memory partitioned per-thread within each block

Each kernel calls into the same `knn<K, Point<3>, HeapQueryResult<K>>` from `common/kdtree/queries.cuh`; the only thing that varies is where the result buffer lives. `validation.cuh` is shared across the three.

## Mitsuba ground-truth comparison

The repo includes a workflow for diffing the in-house renderer against Mitsuba 3. See `mitsuba/cheatsheet.md` for details. Quick path: `mitsuba -m cuda_ad_rgb scenes\cornell-box\scene_v3.xml -o cornell_gt`, then `python mitsuba/tonemap.py cornell_gt.exr cornell_gt.png`. The Python tonemap replicates the Hable curve + sqrt gamma from `ray-tracer/cuda/helpers.cu:filter_colour`, so the two PNGs land in matching response space. Plain ImageMagick conversion will *not* match.

## Scene files

Each scene directory under `scenes/` contains a `scene_v3.xml` (the one the loader reads). Some scenes also carry a `scene_v3_mitsuba.xml` variant kept as a Mitsuba-only reference — when editing scene geometry/lighting, edit `scene_v3.xml`. The CLI override syntax `-D spp=4096 -D resx=1024` works against the `<default>` entries near the top of those files.

# Current goals
We're currently focused on improving the path tracer to test the kd-tree in a more rigorous, production-like environment, instead of the current "tech-demo" approach we've taken so far. To this end, we need to do the following changes:

- Add texture support
- Improve Wavefront (.obj) model support, to render more complex scenes like Sponza
- Make the scene xml format strictly a subset of Mitsuba's, we'd like any scene we render on our program to be "renderable with Mitsuba" (see how we can support custom properties)
- Add support for configuring the camera focal length
- Make the photon mapper and ray tracer support multiple light sources.
- Add support for spot lights and directional lights
- Make the lighting system more robust, using physically based Power (Watts) and allowing them to have a hue.
- Add support for Volumetrics, leveraging the kd-tree and photon mapping capabilities.
- Decouple the drawn Window from frame timings. It currently hangs while the device code is drawing the frame. Ideally we would be able to see the frame drawn in real time.
- Add a hotkey for saving the current frame to disk as a PNG.
- Add stats on a ImGUI-based HUD.
