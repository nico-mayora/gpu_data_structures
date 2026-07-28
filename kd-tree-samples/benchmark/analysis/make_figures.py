#!/usr/bin/env python3
"""Render figures 1-3 from runs.csv as standalone SVG.

    fig1_k_cost_indexed.svg   cost vs K, indexed within each panel (shape)
    fig2_k_cost_absolute.svg  cost vs K, absolute ms, shared log axis (magnitude)
    fig3_k_sensitivity.svg    K-sensitivity vs photon-map size (first GPU only)
    fig4_hardware.svg         per-GPU slowdown heatmap

Stdlib only. Emits vector SVG with light/dark tokens, so the same file works in
a thesis PDF and in a browser.

Usage:
    python make_figures.py --runs out/runs.csv --out out
"""

import argparse
import csv
import math
import os
from collections import defaultdict

# Categorical slots 1-3 of the validated reference palette (light / dark steps).
# Validated all-pairs: worst CVD dE 9.2, worst normal-vision dE 24.0.
SERIES = {
    "global":  ("#2a78d6", "#3987e5"),
    "caustic": ("#eb6834", "#d95926"),
}
SCENE_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
SCENE_SLOTS_DARK = ["#3987e5", "#d95926", "#199e70"]

# The K held fixed while the other is swept. NOT recoverable from the run
# artifacts -- the sweeps were built by hand-editing the constexprs in
# ray-tracer/cuda/pathTracer.cuh, and that working tree was never committed.
# These are the values committed at 3b27ccf (2026-06-01), confirmed by the
# team as the ones used for the June RTX 4070 runs.
PIN_GLOBAL = 24    # K_GLOBAL_PHOTONS,  held fixed while sweeping caustic K
PIN_CAUSTIC = 1    # K_CAUSTIC_PHOTONS, held fixed while sweeping global K

# Ink/surface tokens are written as literal presentation attributes so that
# converters without CSS-variable support (librsvg, Inkscape) still render the
# figure correctly. Dark mode rides on top as a CSS-only overlay: class rules
# beat presentation attributes in a browser, and are ignored everywhere else.
INK = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
       "ink3": "#8a8880", "grid": "#e6e5e2"}
INK_DARK = {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
            "ink3": "#8a8880", "grid": "#2e2e2c"}

STYLE = """
  text  { font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  .title { font-size: 15px; font-weight: 600; }
  .sub   { font-size: 11px; }
  .panel { font-size: 11px; font-weight: 600; }
  .axis  { font-size: 9.5px; font-variant-numeric: tabular-nums; }
  .note  { font-size: 9.5px; }
  /* Heatmap cell values sit on ramp colours that do NOT change between modes,
     so they keep their own fill and are deliberately absent from the dark
     block below -- a .axis dark rule would repaint them into their cells. */
  .cellval { font-size: 9.5px; font-variant-numeric: tabular-nums; font-weight: 600; }
  @media (prefers-color-scheme: dark) {
    .bg    { fill: %(surface)s; }
    .title, .panel { fill: %(ink)s; }
    .sub   { fill: %(ink2)s; }
    .axis  { fill: %(ink2)s; }
    .note  { fill: %(ink3)s; }
    .grid  { stroke: %(grid)s; }
    .ring  { stroke: %(surface)s; }
    /* Element-qualified: a bare `.s-x { fill }` rule would beat the fill="none"
       presentation attribute on the polylines and fill each line into a closed
       polygon. Strokes and fills must be re-declared per element type. */
    polyline.s-global, path.s-global   { stroke: %(s1)s; fill: none; }
    polyline.s-caustic, path.s-caustic { stroke: %(s2)s; fill: none; }
    polyline.s-0, path.s-0 { stroke: %(d0)s; fill: none; }
    polyline.s-1, path.s-1 { stroke: %(d1)s; fill: none; }
    polyline.s-2, path.s-2 { stroke: %(d2)s; fill: none; }
    circle.s-global,  text.s-global  { fill: %(s1)s; }
    circle.s-caustic, text.s-caustic { fill: %(s2)s; }
    /* Hollow GPU markers keep the surface as their fill, not the series hue. */
    rect.s-global  { stroke: %(s1)s; fill: %(surface)s; }
    rect.s-caustic { stroke: %(s2)s; fill: %(surface)s; }
    circle.s-0, text.s-0 { fill: %(d0)s; }
    circle.s-1, text.s-1 { fill: %(d1)s; }
    circle.s-2, text.s-2 { fill: %(d2)s; }
    circle.ring { stroke: %(surface)s; }
  }
""" % dict(INK_DARK, s1="#3987e5", s2="#d95926",
           d0=SCENE_SLOTS_DARK[0], d1=SCENE_SLOTS_DARK[1], d2=SCENE_SLOTS_DARK[2])


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def read_runs(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["k"] = int(r["k"])
        r["ms"] = float(r["ms"]) if r["ms"] else None
        r["requested_global"] = int(r["requested_global"])
        r["requested_caustic"] = int(r["requested_caustic"])
    return rows


def nice_log_ticks(lo, hi):
    """1-2-5 ticks spanning [lo, hi]."""
    if lo <= 0:
        lo = min(x for x in (lo, hi) if x > 0) if hi > 0 else 1.0
    ticks, decade = [], math.floor(math.log10(lo))
    while decade <= math.ceil(math.log10(hi)):
        for m in (1, 2, 5):
            v = m * (10 ** decade)
            if lo / 1.001 <= v <= hi * 1.001:
                ticks.append(v)
        decade += 1
    return ticks or [lo, hi]


def fmt_ms(v):
    # Unit on every tick: a bare "500" next to "1s" on the same axis reads as a
    # scale break rather than 500 ms.
    if v >= 1000:
        return f"{v / 1000:g}s"
    return f"{v:g}ms"


def fmt_count(v):
    if v >= 1_000_000:
        return f"{v / 1_000_000:g}M"
    if v >= 1000:
        return f"{v / 1000:g}k"
    return str(v)


class Panel:
    """One plot box: log-x / log-y, drawn into an SVG fragment."""

    def __init__(self, x, y, w, h, xdom, ydom):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.x0, self.x1 = math.log2(xdom[0]), math.log2(xdom[1])
        self.y0, self.y1 = math.log10(ydom[0]), math.log10(ydom[1])

    def px(self, k):
        return self.x + (math.log2(k) - self.x0) / (self.x1 - self.x0) * self.w

    def py(self, v):
        return self.y + self.h - (math.log10(v) - self.y0) / (self.y1 - self.y0) * self.h


def txt(cls, x, y, s, out, *, anchor=None, extra=""):
    role = {"title": "ink", "panel": "ink", "sub": "ink2", "axis": "ink2", "note": "ink3"}[cls]
    a = f' text-anchor="{anchor}"' if anchor else ""
    out.append(f'<text class="{cls}" fill="{INK[role]}" x="{x:.1f}" y="{y:.1f}"{a}{extra}>'
               f'{esc(s)}</text>')


def draw_frame(p, xticks, yticks, yfmt, out, xlabels=None):
    """Hairline grid + axis labels. Solid, one shade off the surface."""
    for v in yticks:
        yy = p.py(v)
        if not (p.y - 0.5 <= yy <= p.y + p.h + 0.5):
            continue
        out.append(f'<line class="grid" stroke="{INK["grid"]}" stroke-width="1" '
                   f'x1="{p.x:.1f}" y1="{yy:.1f}" x2="{p.x + p.w:.1f}" y2="{yy:.1f}"/>')
        txt("axis", p.x - 5, yy + 3, yfmt(v), out, anchor="end")
    for i, v in enumerate(xticks):
        label = xlabels[i] if xlabels else str(v)
        txt("axis", p.px(v), p.y + p.h + 13, label, out, anchor="middle")
    out.append(f'<line class="grid" stroke="{INK["grid"]}" stroke-width="1" '
               f'x1="{p.x:.1f}" y1="{p.y + p.h:.1f}" x2="{p.x + p.w:.1f}" '
               f'y2="{p.y + p.h:.1f}"/>')


def marker(x, y, colour, cls, style, out):
    """8px mark. Hollow + square is the second channel for GPU identity, so the
    series are never told apart by hue alone."""
    if style["hollow"]:
        out.append(f'<rect class="{cls}" x="{x - 3.6:.1f}" y="{y - 3.6:.1f}" width="7.2" '
                   f'height="7.2" fill="{INK["surface"]}" stroke="{colour}" '
                   f'stroke-width="2"/>')
    else:
        out.append(f'<circle class="ring {cls}" cx="{x:.1f}" cy="{y:.1f}" r="4" '
                   f'fill="{colour}" stroke="{INK["surface"]}" stroke-width="2"/>')


def draw_series(p, pts, colour, cls, out, label=None, label_dy=-9, style=None):
    """2px polyline, 8px markers, optional direct label."""
    style = style or {"dash": "", "hollow": False}
    if not pts:
        return
    if len(pts) > 1:
        d = " ".join(f"{p.px(k):.1f},{p.py(v):.1f}" for k, v in pts)
        dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        out.append(f'<polyline class="{cls}" points="{d}" fill="none" stroke="{colour}" '
                   f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
    for k, v in pts:
        marker(p.px(k), p.py(v), colour, cls, style, out)
    if label:
        k, v = pts[-1]
        out.append(f'<text class="axis lbl {cls}" x="{p.px(k):.1f}" '
                   f'y="{p.py(v) + label_dy:.1f}" text-anchor="middle" fill="{colour}" '
                   f'font-weight="600">{esc(label)}</text>')


def draw_dnf(p, ks, colour, cls, out):
    """Failures marked on the axis -- an absent point would read as absent data."""
    for k in ks:
        xx, yy = p.px(k), p.y + p.h + 16
        out.append(f'<path class="{cls}" d="M{xx - 3.2:.1f},{yy:.1f} l6.4,6.4 '
                   f'M{xx + 3.2:.1f},{yy:.1f} l-6.4,6.4" stroke="{colour}" '
                   f'stroke-width="1.6" fill="none" opacity="0.85"/>')


def legend(x, y, entries, out):
    for label, colour, cls in entries:
        out.append(f'<circle class="{cls}" cx="{x + 4:.1f}" cy="{y - 3:.1f}" r="4" '
                   f'fill="{colour}"/>')
        txt("sub", x + 13, y, label, out)
        x += 15 + 7.4 * len(label)
    return x


def gpu_legend(x, y, gpus, styles, out):
    """Second legend group: the pattern channel, drawn in neutral ink so it
    cannot be mistaken for a series hue."""
    ink = INK["ink2"]
    for g in gpus:
        st = styles[g]
        dash = f' stroke-dasharray="{st["dash"]}"' if st["dash"] else ""
        out.append(f'<line x1="{x:.1f}" y1="{y - 3:.1f}" x2="{x + 26:.1f}" y2="{y - 3:.1f}" '
                   f'stroke="{ink}" stroke-width="2"{dash}/>')
        marker(x + 13, y - 3, ink, "", st, out)
        txt("sub", x + 33, y, g, out)
        x += 42 + 7.4 * len(g)
    return x


def svg_open(w, h, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" class="fig" role="img" aria-label="{esc(title)}">',
            f"<style>{STYLE}</style>",
            f'<rect class="bg" fill="{INK["surface"]}" x="0" y="0" width="{w}" height="{h}"/>']


# ----------------------------------------------------------------------------
# Panel assembly shared by figures 1 and 2
# ----------------------------------------------------------------------------

def gpu_order(rows, baseline=None):
    """GPUs with the baseline first. The baseline is the reference hardware: it
    gets the solid/filled style and is the denominator in fig 4. runs.csv is
    sorted alphabetically, so without this the reference would be whichever GPU
    happens to sort first."""
    seen = []
    for r in rows:
        if r["gpu"] not in seen:
            seen.append(r["gpu"])
    if baseline:
        match = [g for g in seen if g.lower() == baseline.lower()]
        if not match:
            raise SystemExit(f"--baseline {baseline!r} not in runs.csv; have: {seen}")
        seen = match + [g for g in seen if g != match[0]]
    return seen


def gpu_styles(gpus):
    base = [{"dash": "", "hollow": False}, {"dash": "6 3", "hollow": True},
            {"dash": "2 3", "hollow": True}]
    return {g: base[i % len(base)] for i, g in enumerate(gpus)}


def build_configs(rows):
    """(scene, budget) -> {(sweep, gpu): {k: ms}}, usable measurements only."""
    configs = defaultdict(lambda: defaultdict(dict))
    dnfs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["scene"], r["requested_global"], r["requested_caustic"])
        if r["status"] == "ok":
            configs[key][(r["sweep"], r["gpu"])][r["k"]] = r["ms"]
        elif r["status"] == "dnf":
            dnfs[key][(r["sweep"], r["gpu"])].append(r["k"])
    keep = {k: v for k, v in configs.items() if any(len(s) >= 2 for s in v.values())}
    return keep, dnfs


def panel_title(key):
    scene, g, c = key
    return scene, f"global {fmt_count(g)} / caustic {fmt_count(c)}"


def grid_figure(configs, dnfs, gpus, styles, out_path, *, indexed):
    keys = sorted(configs.keys(), key=lambda k: (k[0], k[1], k[2]))
    cols = 3
    rows_n = math.ceil(len(keys) / cols)
    pw, ph = 250, 165
    mx, my = 66, (132 if len(gpus) > 1 else 112)
    gx, gy = 52, 78
    W = mx + cols * pw + (cols - 1) * gx + 24
    H = my + rows_n * ph + (rows_n - 1) * gy + (89 if indexed else 76)

    # Shared y-domain across every panel -- small multiples are only comparable
    # on one scale.
    vals = []
    series_data = {}
    for key in keys:
        series_data[key] = {}
        anchors = {}
        live = [s for s, pts in configs[key].items() if len(pts) >= 2]
        common = (sorted(set.intersection(*[set(configs[key][s]) for s in live]))
                  if len(live) > 1 else [])
        for skey in live:
            pts = sorted(configs[key][skey].items())
            if indexed:
                anchor = common[0] if common else pts[0][0]
                base = configs[key][skey][anchor]
                pts = [(k, v / base) for k, v in pts]
                anchors[skey] = anchor
            series_data[key][skey] = pts
            vals.extend(v for _, v in pts)
        series_data[key]["_anchor"] = anchors
    lo, hi = min(vals), max(vals)
    ydom = (10 ** math.floor(math.log10(lo * 0.9)), 10 ** math.ceil(math.log10(hi * 1.1)))
    yticks = nice_log_ticks(*ydom)
    yfmt = (lambda v: f"{v:g}x") if indexed else fmt_ms

    title = ("Fig 1 - Cost of K, indexed within each configuration"
             if indexed else
             "Fig 2 - Cost of K, absolute frame time")
    sub = ("Each series indexed to its panel's smallest shared K. x = K (log2), y = log. "
           "Flat means K is free; rising means K dominates."
           if indexed else
           "Mean of 10 frames (frame 1 not excluded). x = K (log2), shared log y. "
           "Windowed runs -- GL present is included in the frame time.")

    out = svg_open(W, H, title)
    left = mx - 46
    txt("title", left, 26, title, out)
    txt("sub", left, 43, sub, out)
    legend(left, 60, [(f"global-map K sweep (caustic K pinned at {PIN_CAUSTIC})",
                       SERIES["global"][0], "s-global"),
                      (f"caustic-map K sweep (global K pinned at {PIN_GLOBAL})",
                       SERIES["caustic"][0], "s-caustic")], out)
    if len(gpus) > 1:
        gpu_legend(left, 80, gpus, styles, out)

    xticks = [4, 16, 64, 128, 256]
    for i, key in enumerate(keys):
        r, c = divmod(i, cols)
        px = mx + c * (pw + gx)
        py = my + r * (ph + gy)
        p = Panel(px, py, pw, ph, (4, 256), ydom)
        scene, budget = panel_title(key)
        txt("panel", px, py - 22, scene, out)
        txt("sub", px, py - 9, budget, out)
        draw_frame(p, xticks, yticks, yfmt, out)

        anchors = series_data[key].get("_anchor", {})
        live = [s for s in series_data[key] if s != "_anchor"]
        # With >2 series a direct label at every endpoint collides; identity is
        # carried by hue x pattern plus the legend, so it is never colour-alone.
        # Also skip when the panel holds one sweep on two GPUs -- two identical
        # "global" labels say nothing; the pattern channel already separates them.
        want_labels = len(live) <= 2 and len({sw for sw, _ in live}) == len(live)
        for sweep in ("global", "caustic"):
            for g in gpus:
                skey = (sweep, g)
                pts = series_data[key].get(skey)
                if not pts:
                    continue
                other = series_data[key].get(("caustic" if sweep == "global" else "global", g))
                dy = -9
                if other and other[-1][1] > pts[-1][1]:
                    dy = 15
                draw_series(p, pts, SERIES[sweep][0], f"s-{sweep}", out,
                            label=sweep if want_labels else None, label_dy=dy,
                            style=styles[g])
                draw_dnf(p, sorted(dnfs[key][skey]), SERIES[sweep][0], f"s-{sweep}", out)

        if indexed and anchors:
            a = sorted(set(anchors.values()))
            note = f"indexed to K={a[0]}" if len(a) == 1 else \
                   "indexed per series: " + ", ".join(f"K={k}" for k in a)
            txt("note", px, py + ph + 38, note, out)

    notes = [
        "x below axis = run did not finish. K=256 fails with CUDA error 718 on every "
        "configuration (sticky fault, real cause is the gather kernel).",
        "Sweeps against an empty photon map (cornell-box caustic, sponza caustic) are "
        "excluded -- they measure nothing.",
    ]
    if indexed:
        notes.append(
            "Each series is normalised to its own anchor, so the gap between the two "
            "lines compares growth, not cost. For cost, see fig 2.")
    for i, note in enumerate(notes):
        txt("note", left, H - 9 - 13 * (len(notes) - 1 - i), note, out)
    out.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    return len(keys)


# ----------------------------------------------------------------------------
# Figure 3
# ----------------------------------------------------------------------------

def sensitivity_figure(rows, out_path, k_lo=4, k_hi=128, only_gpu=None):
    by = defaultdict(dict)
    for r in rows:
        if r["status"] != "ok" or r["sweep"] != "global":
            continue
        if only_gpu and r["gpu"] != only_gpu:
            continue
        by[(r["scene"], r["requested_global"])][r["k"]] = r["ms"]

    scenes = defaultdict(list)
    for (scene, budget), ks in by.items():
        if k_lo in ks and k_hi in ks:
            scenes[scene].append((budget, ks[k_hi] / ks[k_lo]))
    for s in scenes:
        scenes[s].sort()

    W, H = 700, 430
    px, py, pw, ph = 78, 96, 540, 250
    xs = [b for pts in scenes.values() for b, _ in pts]
    ys = [v for pts in scenes.values() for _, v in pts]
    xdom = (10 ** math.floor(math.log10(min(xs))), 10 ** math.ceil(math.log10(max(xs))))
    # Headroom rather than a decade snap, so the topmost value label isn't clipped.
    ydom = (1.0, max(ys) * 1.4)

    p = Panel(px, py, pw, ph, xdom, ydom)
    p.x0, p.x1 = math.log10(xdom[0]), math.log10(xdom[1])
    p.px = lambda v: px + (math.log10(v) - p.x0) / (p.x1 - p.x0) * pw

    title = "Fig 3 - K-sensitivity collapses as the photon map grows"
    sub = ("Cost of the global-map gather at K=128 relative to K=4, per scene. "
           "A denser map finds K neighbours in a smaller radius.")
    out = svg_open(W, H, title)
    txt("title", 24, 30, title, out)
    txt("sub", 24, 48, sub, out)
    legend(24, 68, [(s, SCENE_SLOTS[i], f"s-{i}") for i, s in enumerate(sorted(scenes))], out)

    yticks = nice_log_ticks(*ydom)
    xticks = nice_log_ticks(*xdom)
    draw_frame(p, xticks, yticks, lambda v: f"{v:g}x", out,
               xlabels=[fmt_count(v) for v in xticks])

    # Reference: "K costs nothing".
    y1 = p.py(1.0)
    out.append(f'<line class="grid" stroke="{INK["grid"]}" x1="{px}" y1="{y1:.1f}" '
               f'x2="{px + pw}" y2="{y1:.1f}" stroke-width="1.5"/>')
    txt("note", px + 6, y1 - 6, "1x - K costs nothing", out)

    for i, scene in enumerate(sorted(scenes)):
        colour, cls = SCENE_SLOTS[i], f"s-{i}"
        pts = scenes[scene]
        if len(pts) > 1:
            d = " ".join(f"{p.px(b):.1f},{p.py(v):.1f}" for b, v in pts)
            out.append(f'<polyline class="{cls}" points="{d}" fill="none" stroke="{colour}" '
                       f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for b, v in pts:
            out.append(f'<circle class="ring {cls}" cx="{p.px(b):.1f}" cy="{p.py(v):.1f}" '
                       f'r="4" fill="{colour}" stroke="{INK["surface"]}" stroke-width="2"/>')
        # Direct labels are mandatory here: aqua sits below 3:1 on the light surface.
        b, v = pts[-1]
        out.append(f'<text class="axis lbl {cls}" x="{p.px(b) + 9:.1f}" '
                   f'y="{p.py(v) + 3.5:.1f}" fill="{colour}" font-weight="600">'
                   f'{esc(scene)}</text>')
        # Selective labels only: a value on every point collides where the series
        # cross (cornell 19.7x vs sponza 24.2x at 1M) and goes unread anyway.
        for b, v in {pts[0], pts[-1]}:
            txt("note", p.px(b), p.py(v) - 10, f"{v:.1f}x", out, anchor="middle")

    txt("sub", px + pw / 2, py + ph + 34, "requested global photon budget", out, anchor="middle")
    txt("sub", 20, py + ph / 2, "time(K=128) / time(K=4)", out, anchor="middle",
        extra=f' transform="rotate(-90 20 {py + ph / 2:.1f})"')
    txt("note", 24, H - 22,
        "x axis is the *requested* budget: the emitter's actual yield ranges 18%-186%, "
        "and no log survives for most of these configurations.", out)
    txt("note", 24, H - 9,
        "sponza has only two budgets and runs counter to the trend -- treat as "
        "provisional until re-run.", out)
    out.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    return {s: pts for s, pts in scenes.items()}


# Sequential blue ramp, light -> dark = bigger slowdown (one hue, never a rainbow).
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def ramp_colour(v, lo, hi):
    if hi <= lo:
        return RAMP[0]
    t = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return RAMP[max(0, min(len(RAMP) - 1, int(t * len(RAMP))))]


def hardware_figure(rows, gpus, out_path):
    """Fig 4 -- slowdown of each GPU relative to the first, per config and K.

    A grid of magnitude, so: heatmap. The value is printed in every cell, so the
    colour is a reading aid and never the only encoding.
    """
    if len(gpus) < 2:
        return None
    base_gpu, other = gpus[0], gpus[1]
    by = defaultdict(dict)
    for r in rows:
        if r["status"] != "ok":
            continue
        by[(r["scene"], r["requested_global"], r["requested_caustic"], r["sweep"], r["k"])][r["gpu"]] = r["ms"]

    ks = [4, 16, 64, 128]
    sweeps = ["global", "caustic"]
    configs = sorted({(s, g, c) for (s, g, c, _, _) in by})
    # Only configurations measured on BOTH GPUs can enter a hardware comparison.
    configs = [cfg for cfg in configs
               if any(base_gpu in by.get((*cfg, sw, k), {}) and other in by.get((*cfg, sw, k), {})
                      for sw in sweeps for k in ks)]
    if not configs:
        return None

    ratios = {}
    for cfg in configs:
        for sw in sweeps:
            for k in ks:
                cell = by.get((*cfg, sw, k), {})
                if base_gpu in cell and other in cell and cell[base_gpu] > 0:
                    ratios[(cfg, sw, k)] = cell[other] / cell[base_gpu]
    vals = list(ratios.values())
    lo, hi = min(vals), max(vals)

    cw, chh = 58, 30
    rowlab = 210
    gap = 26
    x0, y0 = 24 + rowlab, 132
    gw = len(ks) * cw
    W = x0 + 2 * gw + gap + 30
    H = y0 + len(configs) * chh + 96

    title = f"Fig 4 - Hardware: {other} runtime relative to {base_gpu}"
    sub = (f"Each cell is {other} ms / {base_gpu} ms for the same scene, photon budget and K. "
           f"1.0x = identical; 2.0x = twice as slow.")
    out = svg_open(W, H, title)
    txt("title", 24, 30, title, out)
    txt("sub", 24, 48, sub, out)

    # Ramp legend -- a continuous scale always ships its key.
    lx = 24
    txt("sub", lx, 74, f"slower ->", out)
    for i, c in enumerate(RAMP):
        out.append(f'<rect x="{lx + 56 + i * 22:.1f}" y="64" width="22" height="11" fill="{c}"/>')
    txt("axis", lx + 56, 88, f"{lo:.1f}x", out)
    txt("axis", lx + 56 + len(RAMP) * 22, 88, f"{hi:.1f}x", out, anchor="end")

    for si, sw in enumerate(sweeps):
        gx = x0 + si * (gw + gap)
        pin = PIN_CAUSTIC if sw == "global" else PIN_GLOBAL
        txt("panel", gx, y0 - 34, f"{sw}-map K sweep", out)
        txt("note", gx, y0 - 22, f"other K pinned at {pin}", out)
        for ki, k in enumerate(ks):
            txt("axis", gx + ki * cw + cw / 2, y0 - 6, f"K={k}", out, anchor="middle")

    for ri, cfg in enumerate(configs):
        scene, g, c = cfg
        yy = y0 + ri * chh
        txt("axis", x0 - 10, yy + chh / 2 + 3.5,
            f"{scene}  {fmt_count(g)}/{fmt_count(c)}", out, anchor="end")
        for si, sw in enumerate(sweeps):
            gx = x0 + si * (gw + gap)
            for ki, k in enumerate(ks):
                cx, cy = gx + ki * cw, yy
                v = ratios.get((cfg, sw, k))
                if v is None:
                    txt("note", cx + cw / 2, cy + chh / 2 + 3.5, "-", out, anchor="middle")
                    continue
                col = ramp_colour(v, lo, hi)
                # 2px surface gap between cells rather than a border.
                out.append(f'<rect x="{cx + 1:.1f}" y="{cy + 1:.1f}" width="{cw - 2}" '
                           f'height="{chh - 2}" fill="{col}"/>')
                dark_cell = RAMP.index(col) >= 4
                fill = "#ffffff" if dark_cell else INK["ink"]
                out.append(f'<text class="cellval" x="{cx + cw / 2:.1f}" '
                           f'y="{cy + chh / 2 + 3.5:.1f}" text-anchor="middle" '
                           f'fill="{fill}">{v:.2f}x</text>')

    txt("note", 24, H - 22,
        f"Only configurations measured on both GPUs appear. '-' = not measured on one of them. "
        f"K=256 is omitted: it fails on both.", out)
    txt("note", 24, H - 9,
        "Frame time includes OptiX traversal, shading and GL present -- this is whole-frame "
        "hardware scaling, not kd-tree gather scaling.", out)
    out.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    return ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="out/runs.csv")
    ap.add_argument("--out", default="out")
    ap.add_argument("--baseline", default="RTX 4070",
                    help="reference GPU: solid style, and the denominator in fig 4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = read_runs(args.runs)
    configs, dnfs = build_configs(rows)
    gpus = gpu_order(rows, args.baseline)
    styles = gpu_styles(gpus)

    n1 = grid_figure(configs, dnfs, gpus, styles,
                     os.path.join(args.out, "fig1_k_cost_indexed.svg"), indexed=True)
    grid_figure(configs, dnfs, gpus, styles,
                os.path.join(args.out, "fig2_k_cost_absolute.svg"), indexed=False)
    series = sensitivity_figure(rows, os.path.join(args.out, "fig3_k_sensitivity.svg"),
                                only_gpu=gpus[0])
    ratios = hardware_figure(rows, gpus, os.path.join(args.out, "fig4_hardware.svg"))

    print(f"fig1/fig2: {n1} panels")
    print("fig3 ratios time(K=128)/time(K=4):")
    for scene in sorted(series):
        pretty = ", ".join(f"{fmt_count(b)}->{v:.1f}x" for b, v in series[scene])
        print(f"  {scene:<14} {pretty}")
    if ratios:
        vals = sorted(ratios.values())
        med = vals[len(vals) // 2]
        print(f"\nfig4: {gpus[1]} vs {gpus[0]} over {len(vals)} shared measurements")
        print(f"  slowdown  min {vals[0]:.2f}x   median {med:.2f}x   max {vals[-1]:.2f}x")
        for sw in ("global", "caustic"):
            for k in (4, 16, 64, 128):
                sel = [v for (cfg, s_, k_), v in ratios.items() if s_ == sw and k_ == k]
                if sel:
                    sel.sort()
                    print(f"  {sw:<8} K={k:<4} median {sel[len(sel)//2]:.2f}x  "
                          f"(n={len(sel)}, {sel[0]:.2f}-{sel[-1]:.2f})")
    print(f"\nwrote {args.out}/fig1..fig4")


if __name__ == "__main__":
    main()
