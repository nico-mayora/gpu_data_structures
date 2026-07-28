#!/usr/bin/env python3
"""Parse pathTracer benchmark dumps into tidy CSVs.

Reads a directory of `results_<scene>_n<N>_c<C>.txt` summary tables plus any
`*.log` run logs left alongside them, and emits:

    runs.csv    one row per (scene, budget, sweep, K) measurement
    trees.csv   actual kd-tree sizes recovered from the logs
    frames.csv  per-frame timings recovered from the logs

Stdlib only -- the checked-in matplotlib/numpy pair in this environment is
mismatched, and a parser has no business needing a scientific stack anyway.

Usage:
    python parse_results.py --input ../../../tmp_pathTracer_kcaustic_16 --out ./out
"""

import argparse
import csv
import hashlib
import os
import re
import sys
from collections import defaultdict

# `Benchmark: scene=cornell-box normal=1000000 caustic=100000`
HEADER_RE = re.compile(
    r"Benchmark:\s*scene=(?P<scene>\S+)\s+normal=(?P<normal>\d+)\s+caustic=(?P<caustic>\d+)"
)
# `pathTracer_kglobal_4 606.121` / `pathTracer_kcaustic_256 DNF`
ROW_RE = re.compile(
    r"^pathTracer_k(?P<sweep>global|caustic)_(?P<k>\d+)\s+(?P<value>DNF|[0-9.eE+-]+)\s*$"
)
# `Loaded prebuilt kd-tree (1184684 photons) from: photon_maps/water-caustic_normal_1000000.kdt`
TREE_RE = re.compile(
    r"Loaded prebuilt kd-tree \((?P<count>\d+) photons\) from:\s*(?P<path>\S+)"
)
# `water-caustic_normal_1000000.kdt`
KDT_RE = re.compile(r"^(?P<scene>.+)_(?P<kind>normal|caustic)_(?P<budget>\d+)\.kdt$")
# `  [benchmark] frame 1/10: 11647.7 ms`
FRAME_RE = re.compile(
    r"\[benchmark\] frame (?P<idx>\d+)/(?P<total>\d+):\s*(?P<ms>[0-9.eE+-]+) ms"
)
SCENE_RE = re.compile(r"^Scene:\s*(?P<scene>\S+)")
# Sticky CUDA fault surfaced at the next CUDA call -- the real fault is upstream.
CUDA_ERR_RE = re.compile(r"failed with code (?P<code>\d+) \((?P<msg>[^)]+)\)")


def parse_results_file(path):
    """Yield measurement dicts from one results_*.txt summary table."""
    scene = normal = caustic = None
    rows = []
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            header = HEADER_RE.search(line)
            if header:
                scene = header.group("scene")
                normal = int(header.group("normal"))
                caustic = int(header.group("caustic"))
                continue
            row = ROW_RE.match(line)
            if row:
                if scene is None:
                    print(f"  ! {os.path.basename(path)}: data row before header, skipped",
                          file=sys.stderr)
                    continue
                value = row.group("value")
                rows.append({
                    "scene": scene,
                    "requested_global": normal,
                    "requested_caustic": caustic,
                    "sweep": row.group("sweep"),
                    "k": int(row.group("k")),
                    "ms": "" if value == "DNF" else float(value),
                    "status": "dnf" if value == "DNF" else "ok",
                    "source_file": os.path.basename(path),
                })
    return scene, normal, caustic, rows


def parse_log_file(path):
    """Recover tree sizes, per-frame times and any CUDA fault from one run log."""
    trees, frames = [], []
    scene = None
    fault = None
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            scene_m = SCENE_RE.match(line)
            if scene_m:
                scene = scene_m.group("scene")
            tree = TREE_RE.search(line)
            if tree:
                base = os.path.basename(tree.group("path"))
                kdt = KDT_RE.match(base)
                if kdt:
                    trees.append({
                        "scene": kdt.group("scene"),
                        "kind": kdt.group("kind"),
                        "requested": int(kdt.group("budget")),
                        "actual": int(tree.group("count")),
                    })
            frame = FRAME_RE.search(line)
            if frame:
                frames.append({
                    "log": os.path.basename(path),
                    "scene": scene or "",
                    "frame": int(frame.group("idx")),
                    "frames_total": int(frame.group("total")),
                    "ms": float(frame.group("ms")),
                })
            err = CUDA_ERR_RE.search(line)
            if err:
                fault = (err.group("code"), err.group("msg"))
    return trees, frames, fault


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="DIR=LABEL",
                    help="a run directory and its GPU label; repeat per GPU. "
                         "Earlier --run wins when two runs share an identical file.")
    ap.add_argument("--input", help="single-run shorthand; use with --gpu")
    ap.add_argument("--gpu", default="RTX 4070", help="GPU label for --input")
    ap.add_argument("--out", default="out", help="directory to write CSVs into")
    args = ap.parse_args()

    runs_spec = []
    for spec in args.run:
        if "=" not in spec:
            sys.exit(f"--run needs DIR=LABEL, got {spec!r}")
        d, _, label = spec.rpartition("=")
        runs_spec.append((d, label))
    if args.input:
        runs_spec.append((args.input, args.gpu))
    if not runs_spec:
        sys.exit("give at least one --run DIR=LABEL (or --input DIR)")

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    trees = {}          # (scene, kind, requested) -> actual
    frames = []
    faults = {}
    runs = []
    empty_files = []
    # content hash -> (gpu, filename) of the first run that supplied it. A run
    # that ships a byte-identical results file did not re-measure it.
    seen_hashes = {}
    duplicates = []

    for in_dir, gpu in runs_spec:
        if not os.path.isdir(in_dir):
            sys.exit(f"not a directory: {in_dir}")
        result_files = sorted(f for f in os.listdir(in_dir)
                              if f.startswith("results_") and f.endswith(".txt"))
        log_files = sorted(f for f in os.listdir(in_dir) if f.endswith(".log"))
        if not result_files:
            sys.exit(f"no results_*.txt found in {in_dir}")

        # --- logs first: tree sizes are needed to validate the measurements ---
        for name in log_files:
            t, f, fault = parse_log_file(os.path.join(in_dir, name))
            for row in t:
                trees[(gpu, row["scene"], row["kind"], row["requested"])] = row["actual"]
            for fr in f:
                fr["gpu"] = gpu
            frames.extend(f)
            if fault:
                faults[(gpu, name)] = fault

        # --- results tables ---
        for name in result_files:
            path = os.path.join(in_dir, name)
            with open(path, "rb") as fh:
                digest = hashlib.sha1(fh.read()).hexdigest()
            scene, normal, caustic, rows = parse_results_file(path)
            if not rows:
                empty_files.append((gpu, name))
            dup_of = seen_hashes.get(digest)
            if dup_of is None:
                seen_hashes[digest] = (gpu, name)
            elif dup_of[0] != gpu:
                duplicates.append((gpu, name, dup_of[0]))
            for r in rows:
                r["gpu"] = gpu
                if dup_of is not None and dup_of[0] != gpu:
                    r["status"] = f"duplicate_of:{dup_of[0]}"
            runs.extend(rows)

    # --- join actual tree sizes, and invalidate rows measured against an empty map ---
    for row in runs:
        gpu = row["gpu"]
        g = trees.get((gpu, row["scene"], "normal", row["requested_global"]))
        c = trees.get((gpu, row["scene"], "caustic", row["requested_caustic"]))
        row["actual_global_n"] = "" if g is None else g
        row["actual_caustic_n"] = "" if c is None else c
        # A sweep over K against a zero-photon map measures nothing about the
        # kd-tree; the query returns immediately on N == 0.
        if row["status"] == "ok":
            if row["sweep"] == "caustic" and c == 0:
                row["status"] = "invalid_empty_map"
            elif row["sweep"] == "global" and g == 0:
                row["status"] = "invalid_empty_map"

    # Photon count is a function of scene + budget + emitter, not of hardware. So a
    # map known empty on one GPU is empty on all of them -- propagate the
    # invalidation to runs whose own logs are missing, rather than plotting the
    # empty-map code path as if it were a gather measurement.
    empty_keys = {(s, k, r) for (_, s, k, r), a in trees.items() if a == 0}
    inferred = []
    for row in runs:
        if row["status"] != "ok":
            continue
        gk = (row["scene"], "normal", row["requested_global"])
        ck = (row["scene"], "caustic", row["requested_caustic"])
        hit = (row["sweep"] == "caustic" and ck in empty_keys) or \
              (row["sweep"] == "global" and gk in empty_keys)
        if hit and not row["actual_global_n"] and not row["actual_caustic_n"]:
            row["status"] = "invalid_empty_map_inferred"
            inferred.append((row["gpu"], row["scene"], row["requested_global"],
                             row["requested_caustic"], row["sweep"]))

    fields = ["gpu", "scene", "requested_global", "requested_caustic",
              "actual_global_n", "actual_caustic_n", "sweep", "k", "ms",
              "status", "source_file"]
    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(runs, key=lambda r: (r["gpu"], r["scene"], r["requested_global"],
                                                r["requested_caustic"], r["sweep"], r["k"])))

    with open(os.path.join(out_dir, "trees.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gpu", "scene", "kind", "requested", "actual", "yield_pct"])
        for (gpu, scene, kind, req), actual in sorted(trees.items()):
            w.writerow([gpu, scene, kind, req, actual,
                        round(100.0 * actual / req, 2) if req else ""])

    with open(os.path.join(out_dir, "frames.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gpu", "log", "scene", "frame", "frames_total", "ms"])
        w.writeheader()
        w.writerows(frames)

    # ------------------------------------------------------------------
    # Data-quality report. These are the caveats any figure must carry.
    # ------------------------------------------------------------------
    ok = [r for r in runs if r["status"] == "ok"]
    dnf = [r for r in runs if r["status"] == "dnf"]
    invalid = [r for r in runs if r["status"].startswith("invalid_empty_map")]
    dups = [r for r in runs if r["status"].startswith("duplicate_of:")]

    print(f"\nparsed {len(runs_spec)} run(s): " + ", ".join(g for _, g in runs_spec))
    print(f"  {len(ok)} usable measurements, {len(dnf)} DNF, "
          f"{len(invalid)} invalidated (empty photon map), {len(dups)} dropped as duplicates")

    if duplicates:
        print("\n[DUPLICATE] these files are byte-identical to another run's -- the GPU that")
        print("            shipped them did NOT re-measure that configuration. Dropped:")
        for gpu, name, origin in duplicates:
            print(f"  - {gpu}: {name}  (identical to {origin})")

    if inferred:
        print("\n[EMPTY MAP, INFERRED] no logs for these runs, but the same scene+budget")
        print("                      produced 0 photons on another GPU. Excluded:")
        for gpu, scene, g, c, sweep in sorted(set(inferred)):
            print(f"  - {gpu}: {scene} n{g} c{c} {sweep} sweep")

    # Caustic photons need specular/refractive geometry. A scene that yields 0 at
    # a large budget will yield 0 at every budget -- but that is a scene-level
    # inference, weaker than the exact scene+budget match above, so warn instead
    # of excluding.
    barren = {s for (_, s, kind, _), a in trees.items() if kind == "caustic" and a == 0}
    suspect_scene = sorted({(r["gpu"], r["scene"], r["requested_caustic"])
                            for r in runs
                            if r["status"] == "ok" and r["sweep"] == "caustic"
                            and r["scene"] in barren})
    if suspect_scene:
        print("\n[SUSPECT] this scene yielded 0 caustic photons at another budget, so it")
        print("          probably has no specular geometry at all. Still plotted -- verify")
        print("          with a log before trusting these as gather measurements:")
        for gpu, scene, c in suspect_scene:
            print(f"  - {gpu}: {scene} caustic sweep at budget {c:,}")

    if empty_files:
        print("\n[EMPTY] result files with no data rows -- re-run these:")
        for gpu, name in empty_files:
            print(f"  - {gpu}: {name}")

    # Configurations measured on one GPU but not the other -- these cannot enter
    # any hardware comparison.
    by_gpu = defaultdict(set)
    for r in runs:
        if r["status"] == "ok":
            by_gpu[r["gpu"]].add((r["scene"], r["requested_global"], r["requested_caustic"]))
    if len(by_gpu) > 1:
        gpus = sorted(by_gpu)
        shared = set.intersection(*by_gpu.values())
        print(f"\n[COVERAGE] {len(shared)} configuration(s) measured on all {len(gpus)} GPUs.")
        for gpu in gpus:
            only = sorted(by_gpu[gpu] - shared)
            if only:
                print(f"  {gpu} only ({len(only)}):")
                for scene, g, c in only:
                    print(f"    - {scene} n{g} c{c}")

    zero_maps = sorted({(g, s, k, r) for (g, s, k, r), a in trees.items() if a == 0})
    if zero_maps:
        print("\n[EMPTY MAP] emitter produced 0 photons -- sweeps against these measure nothing:")
        for gpu, scene, kind, req in zero_maps:
            print(f"  - {gpu}: {scene} {kind} (requested {req:,})")

    if trees:
        print("\n[YIELD] requested vs actual photon count:")
        for (gpu, scene, kind, req), actual in sorted(trees.items()):
            pct = 100.0 * actual / req if req else 0.0
            print(f"  - {gpu}: {scene:<14} {kind:<7} requested {req:>10,} -> "
                  f"actual {actual:>10,}  ({pct:6.1f}%)")

    if faults:
        print("\n[CUDA FAULT] recorded in logs (error is sticky; the real fault is the")
        print("             preceding kernel launch, not the call that reports it):")
        for (gpu, name), (code, msg) in sorted(faults.items()):
            print(f"  - {gpu}: {name}: code {code} ({msg})")

    if dnf:
        print("\n[DNF] these need a cause before they can be charted as failures.")
        print("      A DNF at a *smaller* K than a run that succeeded is not a")
        print("      resource limit -- suspect an interrupted run:")
        by_config = defaultdict(list)
        for r in dnf:
            by_config[(r["gpu"], r["scene"], r["requested_global"],
                       r["requested_caustic"], r["sweep"])].append(r["k"])
        for (gpu, scene, g, c, sweep), ks in sorted(by_config.items()):
            ks = sorted(ks)
            ok_ks = sorted(r["k"] for r in ok
                           if (r["gpu"], r["scene"], r["requested_global"],
                               r["requested_caustic"], r["sweep"]) == (gpu, scene, g, c, sweep))
            suspect = [k for k in ks if ok_ks and k < max(ok_ks)]
            flag = f"   <-- SUSPECT (succeeded at K={max(ok_ks)})" if suspect else ""
            print(f"  - {gpu}: {scene} n{g} c{c} {sweep}: K={ks}{flag}")

    # Warm-up: frame 1 vs the median of the rest, per log.
    by_log = defaultdict(list)
    for f in frames:
        by_log[f["log"]].append(f)
    warmups = []
    for log, fs in by_log.items():
        fs = sorted(fs, key=lambda x: x["frame"])
        if len(fs) < 3:
            continue
        rest = sorted(x["ms"] for x in fs[1:])
        median = rest[len(rest) // 2]
        delta = 100.0 * (fs[0]["ms"] - median) / median
        if abs(delta) > 5.0:
            warmups.append((log, fs[0]["ms"], median, delta))
    if warmups:
        print("\n[WARM-UP] frame 1 differs from the median of frames 2..n by >5%.")
        print("          Drop frame 1 before averaging:")
        for log, first, median, delta in sorted(warmups):
            print(f"  - {log:<32} frame1 {first:9.1f} ms vs median {median:9.1f} ms  ({delta:+.1f}%)")

    missing = sorted({(r["gpu"], r["scene"], r["requested_global"], r["requested_caustic"])
                      for r in runs if r["actual_global_n"] == ""})
    if missing:
        print("\n[NO TREE SIZE] no log survives for these configs, so only the *requested*")
        print("               budget is known -- charts must label the axis accordingly:")
        for gpu, scene, g, c in missing:
            print(f"  - {gpu}: {scene} n{g} c{c}")

    print(f"\nwrote {out_dir}/runs.csv, trees.csv, frames.csv")


if __name__ == "__main__":
    main()
