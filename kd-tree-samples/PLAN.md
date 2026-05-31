# PLAN.md

Breakdown of the work listed under "Current goals" in `CLAUDE.md`. The goals are grouped into four phases. Phase order reflects dependencies: each phase consumes capabilities introduced earlier and avoids rework. Within a phase, items can mostly proceed in parallel.

## Sequencing summary

```
Phase 1: Scene & asset pipeline   →  unblocks complex scenes (Sponza)
   1.1  Mitsuba-subset XML
   1.2  OBJ multi-material / groups
   1.3  Textures
   1.4  Camera focal length

Phase 2: Lighting overhaul        →  depends on 1.1 (light XML schema)
   2.1  Multi-light refactor
   2.2  Spot & directional light types
   2.3  Physically based units (Watts, hue)

Phase 3: Viewer & UX              →  independent, can run alongside any phase
   3.1  Decoupled render / progressive accumulation
   3.2  PNG save hotkey
   3.3  ImGui HUD

Phase 4: Volumetrics              →  depends on 2.1–2.3 and photon pipeline
   4.1  Volume photons in emitter
   4.2  In-scatter integration via kd-tree
   4.3  Media in scene XML
```

## Cross-cutting concerns to settle first

Before starting Phase 1, agree on three things — every later task touches them.

- **Scene XML conventions — decided.** Non-Mitsuba renderer parameters are carried as `<default name="X" value="Y" />` entries at the top of the scene. Mitsuba 3.8 accepts inert/unused defaults without error, so the file dual-loads. Our loader's existing `defaultValues` map is the read path; consumers look up the key after the main parse loop. *Originally tried a sibling `<extras>` block — Mitsuba 3.8 throws on unknown top-level elements, so that approach was retracted during Phase 1.1.* Convention is documented at the top of `common/data/loader/mitsuba3.cuh`.
- **Light + material data layout on the GPU — decided: tagged-union AoS.** Grow the existing structs in `common/data/world.cuh`: a tagged `Light` (point / spot / directional discriminated by an enum) and a fixed-size `Material` padded to ~64 B with texture-index fields and a `MaterialType` BSDF tag. Rationale: in a path tracer, neighboring warp threads hit different surfaces and sample different lights, so SoA's coalesced-read win does not materialize; AoS keeps `OWL_USER_TYPE` bindings simple and avoids N buffer-pointer plumbing per closest-hit. **Follow-up: profile.** Once textures and multi-light are in, measure material-buffer bandwidth (Nsight Compute, `dram__bytes_read` filtered to the material buffer) on a Sponza-class scene. If it pops as a bottleneck, revisit per-field SoA for the hottest material fields (likely albedo + texture index).
- **Photon pipeline — decided: dump the built kd-tree, not raw photons.** Run `build_kd_tree` on the photon-mapper side after emission, then `cudaMemcpy` the tree-ordered device buffers (both `Photon` and the parallel `PhotonCoord` view, kept in lockstep) back to host and dump as binary blobs through `PhotonFileManager` (the binary path already exists; extend, do not replace). The path tracer then `cudaMemcpy`s straight from a host buffer to device — no build at startup. Two files, surface and volume, kept separate since they feed independent kd-trees. Each blob gets a small versioned header (`magic + version + record_count + sizeof(Photon)`) so a stale dump fails loudly rather than silently corrupting layout. Keep the text dump as a debug-only mode. Light-id is **not** stored in the photon record — radiance estimation does not need it.

---

## Phase 1 — Scene & asset pipeline

### 1.1 Mitsuba-subset scene XML

**Goal.** Any scene file in `scenes/` is loadable by both our renderer *and* `mitsuba` unmodified. Our custom extensions live in a Mitsuba-ignored channel.

**Tasks.**
- Audit `common/data/loader/mitsuba3.cu` for every tag/attribute it consumes; compare against Mitsuba 3's actual schema and flag deviations.
- Pick the custom-property convention (see cross-cutting). Document it at the top of the loader header.
- Rewrite the loader to skip-with-warning on unknown tags instead of asserting, so future Mitsuba scenes drop in.
- Re-author existing scenes (`cornell-box`, `veach-bidir`, `water-caustic`) to satisfy both renderers. Delete the `scene_v3_mitsuba.xml` variant once they unify; until they do, keep the [[scene-file-targeting]] rule.
- Add a CI-friendly smoke test: load each scene through our loader and then through `mitsuba -m scalar_rgb --dry-run` (or equivalent) to confirm dual parsing.

**Acceptance.** `scene_v3.xml` for all three current scenes renders in both engines.

### 1.2 OBJ / Wavefront improvements

**Goal.** Render Crytek Sponza (or comparable) without preprocessing.

**Tasks.**
- Move OBJ loading out of `Mesh::loadObj` (in `common/data/world.cuh`) into its own translation unit; the current header is being included in too many places to absorb the future complexity.
- Honor `usemtl` boundaries: produce one `Model` (mesh + material) per material group, not one mesh per file. This is what currently blocks Sponza.
- Parse the `.mtl` sidecar for diffuse/specular/IOR/`map_Kd`. Bridge into our `Material` enum (or its successor after the layout decision).
- Preserve UVs alongside vertices (`tinyobj::attrib_t::texcoords`) — required by 1.3. Add a `std::vector<owl::vec2f> uvs` to `Mesh`.
- Replace the hard `assert(fv == 3)` with triangulation of n-gon faces, or document the restriction and fail with a clear error.
- Budget check: load Sponza (~280k tris, ~25 materials) and confirm BVH build + first-frame time on the target GPU.

**Acceptance.** Sponza loads, renders, and shades per-submesh correctly.

### 1.3 Texture support

**Depends on 1.2 (UVs) and 1.1 (texture refs in XML).**

**Tasks.**
- Add a `Texture` type to `common/data/world.cuh` holding an OWL/CUDA texture object plus metadata. Owner is `World`; materials reference textures by index.
- Image decode on the host: pick a loader (stb_image is already a transitive dep via OWL — confirm or vendor it). Support at least 8-bit RGB/RGBA PNG/JPG and EXR (project already has EXR ground-truth files).
- Plumb UVs from `Mesh` through OWL `TrianglesGeomData` so closest-hit can sample.
- Extend `Material` to carry texture indices for albedo first; specular/normal/roughness maps follow.
- Update both `pathTracer.cu` and `photonEmitter.cu` closest-hit programs to sample albedo from the texture when bound. Photons need this so caustic colors come from textures, not flat albedo.
- Add `<bsdf>` → texture wiring in the Mitsuba loader (Mitsuba uses `<texture type="bitmap">`; mirror that exactly).

**Acceptance.** Sponza renders with textured walls/columns; the same XML renders identically (modulo sampling noise) in Mitsuba.

### 1.4 Camera focal length

**Tasks.**
- Add `focal_length` to the `Camera` struct (`common/data/world.cuh`), populated from Mitsuba's `<float name="focal_length">` *or* derived from `fov` + sensor size when only one is given.
- Update the viewer's primary-ray generation (`ray-tracer/cuda/pathTracer.cu` raygen) and `Viewer::cameraChanged()` to use focal length.
- Decide whether to expose a runtime keybind for changing it (out of scope here unless trivial).

**Acceptance.** Same scene matches Mitsuba framing when the XML specifies focal length instead of FOV.

---

## Phase 2 — Lighting overhaul

### 2.1 Multi-light refactor

**Tasks.**
- Replace `World::scene_light` (`PointLight*`) with `std::vector<Light*>` plus a device-side flat buffer.
- Define a tagged `Light` struct or polymorphic encoding (settle in the cross-cutting decision). Today only point lights exist; the new layout must leave room for spot/directional and area later.
- Update `photon-mapper/main.cu` to loop over lights: split the global photon budget across them (proportional to power) and emit per light. `computePhotonsPerWatt` needs to operate on total scene wattage, not a single light's.
- Update `ray-tracer/cuda/pathTracer.cu` direct-lighting shading to iterate over the light buffer (or sample one stochastically per bounce — pick a strategy and document it).
- Photon records stay anonymous (no `light_id` — decided in cross-cutting).

**Acceptance.** A scene with two point lights renders correctly; total emitted photon count matches the single-light baseline when only one light is present.

### 2.2 Spot lights & directional lights

**Depends on 2.1.**

**Tasks.**
- Spot light: position + direction + inner/outer cone angles + power. Photon emission samples a cone; raygen direct lighting checks the cone falloff.
- Directional light: direction + irradiance. Photon emission picks a uniform point on a disk perpendicular to the direction sized to the scene AABB (compute scene bounds once at load).
- Add Mitsuba XML mapping for both (`<emitter type="spot">`, `<emitter type="directional">`). Keep our extensions Mitsuba-compatible.

**Acceptance.** A spot light produces a visible cone with proper falloff; a directional light produces parallel shadows.

### 2.3 Physically based units & spectral hue

**Tasks.**
- Standardize all light power on Watts (`owl::vec3f` per-channel radiant flux). Today `EmittedPhoton::power` is `int` and `PointLight::power` doubles as color — separate them.
- Update photon weighting: photon energy = `light.power / num_photons_for_this_light`, RGB-channeled.
- Update direct-lighting and photon-density estimation in `pathTracer.cu` to consume RGB watts consistently. Sanity-check that the Mitsuba ground truth matches within tonemap tolerance after the unit change.
- Document the unit convention in `CLAUDE.md` (one line) so future contributors don't reintroduce intensity scalars.

**Acceptance.** Cornell box ground-truth diff against Mitsuba stays within current error band after the refactor.

---

## Phase 3 — Viewer & UX

These are independent of the other phases. Do them whenever someone needs a context switch from the heavier work.

### 3.1 Decouple window from render

**Problem.** `Viewer::render()` blocks the GL thread, so the window freezes while OptiX runs.

**Approach.**
- Run OptiX launches on a worker thread; the GL thread only blits the latest accumulated buffer.
- Switch to progressive accumulation: each launch contributes `pixel_samples` per pixel into a float accum buffer; the display thread reads it whenever it wants.
- Frame-to-frame: on `cameraChanged()`, reset the accum buffer instead of throwing away the in-flight launch.

**Tasks.**
- Add an accumulation buffer to the OWL raygen output.
- Move the launch into a thread (or use OptiX async + CUDA streams + `cudaEventQuery` to poll completion without blocking).
- Confirm OWLViewer's GLFW loop tolerates the change (`OWLViewer::render` is called from the main thread by the GLFW callback — we may need to override more of the base class).

**Acceptance.** Window stays responsive at low spp; image converges smoothly as samples accumulate.

### 3.2 PNG save hotkey

**Trivial after 3.1's accum buffer exists.**

**Tasks.**
- Add a `key()` override in `Viewer` (OWLViewer exposes this) bound to e.g. `P`.
- Read back the current accumulated frame, tonemap it the same way the display path does, write via `stb_image_write` (already vendored transitively).
- Save path: `screenshots/<scene>_<timestamp>.png` next to the binary.

**Acceptance.** Pressing the bound key dumps a PNG matching what's on screen.

### 3.3 ImGui HUD

**Tasks.**
- Add ImGui as a submodule under `common/externals/imgui` and wire its GLFW + OpenGL3 backends into the OWLViewer GL context.
- Surface: current spp, ms/frame, photons-in-tree, camera pose, last screenshot path. Read-only first; controls (focal length slider, spp cap) are a follow-up.
- Make the HUD toggleable with a hotkey so screenshots can be clean.

**Acceptance.** HUD renders over the path-traced image without breaking the framebuffer blit; stats update at >10 Hz.

---

## Phase 4 — Volumetrics

This is the largest single goal and the reason to nail Phases 1–2 first: it touches the photon pipeline, the kd-tree query path, the material/light system, and the scene XML.

### 4.1 Volume photons in the emitter

**Tasks.**
- Extend `EmittedPhoton` with a `media_id` (or split into `surface_photons.txt` / `volume_photons.txt`).
- Russian-roulette the photon at each step against the medium's extinction coefficient; if it scatters in volume, record a photon at the scatter position with the scattered direction.
- Add a third photon-emitter pass for volumetric photons, with its own budget knob in `photon-mapper/main.cu`.

### 4.2 In-scatter integration

**Tasks.**
- Build a kd-tree over volume photons (the existing `build_kd_tree` works as-is — that's the payoff of the generic point interface).
- In the path tracer, ray-march participating media segments and accumulate in-scatter via radius-limited kNN over the volume photon kd-tree.
- Decide the kNN budget for volume queries (`K_VOLUME_PHOTONS`). Likely tunable in `pathTracer.cuh`.

### 4.3 Media in scene XML

**Tasks.**
- Mitsuba syntax: `<medium type="homogeneous"><rgb name="sigma_t" .../>...</medium>`. Mirror it exactly. Bind media to shapes via `<ref id="...">` inside `<shape>`.
- Add a `Medium` struct to `World` with absorption/scattering/g (Henyey-Greenstein).
- Extend the closest-hit / miss programs to traverse media boundaries (track current medium per ray segment).

**Acceptance.** A Cornell box with a homogeneous fog volume renders god-rays from the light through the fog, matching Mitsuba within the usual tolerance.

---

## Risks & open questions

- **OWLViewer threading model.** Phase 3.1 may run into OWLViewer assumptions about single-threaded GL. Budget time to either upstream a fix or fork the relevant header.
- **Mitsuba schema drift.** Phase 1.1 freezes our XML to Mitsuba 3's current grammar; when Mitsuba 4 lands we either pin or migrate. Note the chosen Mitsuba version in `CLAUDE.md`.
- **Photon file backwards-compat.** Phases 2.1 and 4.1 both change the photon record. The versioned blob header (cross-cutting decision) catches mismatches; bump the version on every layout change and regenerate dumps.
- **Volumetric kd-tree size.** Volume photon density scales with media optical thickness; for thick media the tree may not fit in GPU memory at the sample counts we want. Plan for chunked queries or a coarser secondary structure if this bites.
