# Benchmark analysis

Turns the `pathTracer` benchmark dumps into tidy CSVs and figures. Stdlib Python
only — no numpy/matplotlib (the pair installed in this environment is version
mismatched, and neither is needed).

```sh
python parse_results.py \
  --run "../../../tmp_pathTracer_kcaustic_16=RTX 4070" \
  --run "../../../rtx3060ti_run=RTX 3060 Ti" \
  --out ./out
python make_figures.py --runs out/runs.csv --out ./out --baseline "RTX 4070"
```

Repeat `--run DIR=LABEL` per GPU. `--baseline` picks the reference hardware: it
gets the solid line style and is the denominator in fig 4. Without it the
reference would be whichever label sorts first, which is not a meaningful choice.

`parse_results.py` writes `runs.csv` (one row per measurement), `trees.csv`
(actual kd-tree sizes recovered from the logs) and `frames.csv` (per-frame
timings), then prints a data-quality report. **Read the report before trusting a
figure** — it is where the caveats live.

`make_figures.py` writes three standalone SVGs. They carry literal colours plus a
`prefers-color-scheme` overlay, so they render correctly both in a converter
(librsvg, Inkscape → PDF) and in a browser, light or dark.

| File | Shows |
|---|---|
| `fig1_k_cost_indexed.svg` | cost vs K, indexed per panel — the *shape* of K scaling |
| `fig2_k_cost_absolute.svg` | the same sweeps in absolute ms on a shared log axis |
| `fig3_k_sensitivity.svg` | K-sensitivity vs photon-map size — **baseline GPU only, see caveat below** |
| `fig4_hardware.svg` | slowdown of each GPU vs the baseline, per config and K |

In figs 1 and 2 the GPU is a *composite encoding*: hue is the sweep
(global/caustic), and line pattern + marker shape is the GPU (solid/filled =
baseline, dashed/hollow = the other). Identity is therefore never carried by
colour alone, and the vertical gap between a solid and a dashed line of the same
hue is the hardware difference.

Fig 3 is deliberately single-GPU. It is a ratio of two points, so its downward
slope is driven as much by the growing denominator as by K getting cheaper — on
two of three scenes the *absolute* cost of raising K rises with the photon
budget. Adding a second GPU would double its series and entrench a framing that
should be replaced (by frame time vs N at fixed K, with fitted exponents).

## The pinned K values

Each sweep varies one K and holds the other fixed:

| Series | Swept | Pinned |
|---|---|---|
| `pathTracer_kglobal_*` | `K_GLOBAL_PHOTONS` | `K_CAUSTIC_PHOTONS = 1` |
| `pathTracer_kcaustic_*` | `K_CAUSTIC_PHOTONS` | `K_GLOBAL_PHOTONS = 24` |

**These are not recorded in any run artifact.** The sweeps were built by
hand-editing the `constexpr`s in `ray-tracer/cuda/pathTracer.cuh` (there is no
`-D` or CMake option for them), and that working tree was never committed — the
runs fall between `3b27ccf` (2026-06-01) and `0a68e21` (2026-07-06). The values
above match what `3b27ccf` has committed and were confirmed by the team. They
live in `PIN_GLOBAL` / `PIN_CAUSTIC` in `make_figures.py` and are printed in the
figure legends.

Because the two series have different fixed baselines, **compare them on
absolute deltas, not ratios.** Over K=4→128 the caustic gather adds 30–200x less
wall time than the global gather on the same scene (cornell-box n10M: +11 ms vs
+2343 ms; water-caustic n500k: +22 ms vs +707 ms), so the "caustic K is nearly
free" result holds and is not an artifact of the global-24 baseline diluting it.

To stop this recurring: make K a build option and log both constants at startup.

## Known problems in the current data

These are limits of the runs, not of the tooling. The figures are annotated with
them, but they are the reason nothing here is publication-ready yet:

- **`DNF` conflates three causes.** K=256 fails with CUDA error 718 on every
  configuration — a real resource limit, and a genuine finding. But the caustic
  K=4 DNFs on `cornell-box n1000000` and `water-caustic n100000` sit between
  runs that succeeded at higher K, and `kcaustic_4.log` ends in `^C`: those are
  interrupted runs, not failures. The harness should record an exit reason.
- **The budget is not the tree size.** Emitter yield ranges 18%–186% (photons are
  stored at multiple bounces, so a "normal" map can exceed its budget). Only four
  configurations still have a log, so most axes can only be labelled with the
  *requested* budget.
- **Two photon maps are empty** (`cornell-box` caustic, `sponza` caustic → 0
  photons). Sweeps against them are dropped as `invalid_empty_map`.
- **Two result files are empty** and one contains a `%E DNF` row from a shell
  escaping bug: `results_sponza_n1000_c1000.txt`,
  `results_water-caustic_n10000000_c1000000.txt`,
  `results_water-caustic_n5000000_c5000000.txt`.
- **Frame 1 is a warm-up outlier** (up to +36%) and is currently inside the mean.
- **Runs were windowed** (`Visible: ON`), so GL interop and present are inside
  the frame time.
- **The 3060 Ti set shipped two 4070 files verbatim.**
  `results_cornell-box_n1000000_c100000.txt` and `results_sponza_n1000_c1000.txt`
  are byte-identical to the 4070's, timestamps and all — that hardware never
  measured those configurations. The parser hashes every file and drops
  cross-GPU duplicates; without that they would have entered fig 4 as a
  fabricated 1.00x.
- **The 3060 Ti set has no logs**, so it carries no tree sizes, no per-frame
  timings and no CUDA fault records. Empty-map invalidation is therefore
  *inferred* across GPUs: photon count depends on scene + budget + emitter, not
  on hardware, so a map known empty on one GPU is treated as empty on all.
- **Only 7 configurations were measured on both GPUs**, and the two sets diverge
  on cornell-box (the 4070 used caustic budgets of 100k/1M; the 3060 Ti used
  1M/10M). Fig 4 shows only the shared subset.

## What the harness should emit instead

A single CSV per run — `scene, requested_global, requested_caustic,
actual_global_n, actual_caustic_n, sweep, k, frame_idx, ms, status, exit_reason,
gpu, driver, cuda, clock_lock` — would make `parse_results.py` a ten-line reader
and remove every caveat above except the empty maps.
