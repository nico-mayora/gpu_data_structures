# Mitsuba Comparison Cheatsheet

Reference commands for rendering scenes in Mitsuba 3 and converting the EXR output to PNG for ground-truth comparison against the custom path tracer.

## Render with Mitsuba

Output goes to `<scene_name>.exr` regardless of the `-o` extension (the `hdrfilm` film in the scene XML forces openexr).

### CPU (scalar, no extra deps)
```powershell
mitsuba -m scalar_rgb scenes\cornell-box\scene_v3.xml -o cornell_gt
```
Slow but guaranteed to work. Single-threaded.

### CPU (LLVM, multi-core JIT)
Requires LLVM ≥11 installed and `DRJIT_LIBLLVM_PATH` pointing at `LLVM-C.dll`:
```powershell
$env:DRJIT_LIBLLVM_PATH = "C:\Program Files\LLVM\bin\LLVM-C.dll"
mitsuba -m llvm_ad_rgb scenes\cornell-box\scene_v3.xml -o cornell_gt
```

### GPU (CUDA)
```powershell
mitsuba -m cuda_ad_rgb scenes\cornell-box\scene_v3.xml -o cornell_gt
```
Fastest. Requires a current NVIDIA driver; no extra install beyond `pip install mitsuba`.

### Override scene defaults from the CLI
```powershell
mitsuba -m cuda_ad_rgb scenes\cornell-box\scene_v3.xml -D spp=8192 -D resx=1024 -D resy=1024 -o cornell_gt
```

## EXR → PNG: quick (ImageMagick, no tonemap)

Just linear-to-sRGB with an exposure multiplier. Fast iteration, but no tone curve — bright spots blow out.
```powershell
magick cornell_gt.exr -evaluate Multiply 0.05 -colorspace sRGB cornell_gt.png
```
Tune the `0.05` multiplier until the walls look right.

## EXR → PNG: matched tonemap (tonemap.py)

Applies the same Hable curve + sqrt gamma as `ray-tracer/cuda/helpers.cu:filter_colour`, so the Mitsuba output and your renderer's output land in the same response space.

### One-time install
```powershell
pip install opencv-python
```

### Usage
```powershell
python tonemap.py cornell_gt.exr cornell_gt.png
```

Optional third arg overrides exposure (default `0.5`, matches `filter_colour`):
```powershell
python tonemap.py cornell_gt.exr cornell_gt.png 0.3
```

## Workflow

1. Render Mitsuba ground truth:
   ```powershell
   mitsuba -m cuda_ad_rgb scenes\cornell-box\scene_v3.xml -D spp=4096 -o cornell_gt
   ```
2. Tonemap to PNG with the matched curve:
   ```powershell
   python tonemap.py cornell_gt.exr cornell_gt.png
   ```
3. Render your tracer's PNG separately.
4. Diff visually or with FLIP (`pip install flip-evaluator`).

## Notes

- Mitsuba 3's `hdrfilm` only accepts HDR formats (`openexr`, `pfm`, `rgbe`) — `file_format=png` will fail. Keep it on `openexr` and tonemap separately.
- For a fair comparison, both images must use the same tonemap. The `tonemap.py` script replicates your renderer's curve; using `magick` alone will not match.
- Mitsuba's `perspective` camera looks down local +Z by default; matrix transforms with a `-1` on column 2 flip that to world `-Z`. The custom scene loader uses the same convention.
