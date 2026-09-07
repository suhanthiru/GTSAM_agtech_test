"""The static gate images.  These come before any camera work.

Gates 3 and 4 are the ones to protect.  If the washboard is not obvious in a
still cross-section of the blue map, and if the green cross-section does not
visibly hug the true profile, the problem is upstream and no amount of rendering
will save it.  So they are drawn first, from the finished map products, with
matplotlib and nothing else.

Four figures:

``crosssection.png``
    A one-metre slab cut perpendicular to the flight lines and viewed edge-on,
    which is the only view that shows this kind of error.  Top-down hides it
    entirely.  Blue above, green below, the true bare-earth profile drawn
    through both, and a residual panel where the corrugation is unmistakable.

``ghost.png``
    A close-up on a patch containing props and crop rows, coloured by which
    direction the aircraft was flying.  Drift shows here as doubled rows and
    smeared posts.

``boresight.png``
    Nominal, recovered and true mounting angles side by side.

``report.json``
    The numbers behind all of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import load_run_config
from ..io import RunReader, load_traj
from ..map import raster as rast
from ..sim.terrain import Heightfield, sample_bilinear

BLUE = "#2E6FD6"
GREEN = "#1D9E75"
TRUTH = "#111111"


def heightfield_of(reader: RunReader) -> Heightfield:
    d = reader.heightfield()
    return Heightfield(x0=float(d["x0"]), y0=float(d["y0"]), dx=float(d["dx"]),
                       z_terrain=d["z_terrain"], z_surface=d["z_surface"],
                       canopy_h=d["canopy_h"], class_id=d["class_id"],
                       ditch_p0=tuple(d["ditch_xy"][0]), ditch_p1=tuple(d["ditch_xy"][1]))


def pass_lines(cfg) -> list[float]:
    return [cfg.scene.survey_y - i * cfg.flight.spacing
            for i in range(cfg.flight.n_passes)]


def ripple(y: np.ndarray, resid_m: np.ndarray, spacing: float,
           row_spacing: float) -> dict:
    """Peak-to-peak and locked-in amplitude of the corrugation, in centimetres.

    The lock-in period is twice the flight-line spacing.  A mounting error tilts
    an outbound swath one way and a return swath the other, so the pattern
    repeats every *pair* of lines even though one peak or trough falls on each
    line.  Two other signals are removed first: the terrain, by a cubic fit, and
    an alias of the crop furrows, by smoothing over one and a half row spacings.
    """
    ok = np.isfinite(resid_m)
    empty = {"peak_to_peak_cm": 0.0, "amplitude_cm": 0.0, "residual_cm": 0.0,
             "significance": 0.0, "n": int(ok.sum())}
    if ok.sum() < 40:
        return empty
    yy, rr = y[ok], resid_m[ok]
    step = float(np.median(np.diff(yy)))
    w = max(int(round(1.5 * row_spacing / step)) | 1, 3)
    k = w // 2
    sm = np.convolve(rr, np.ones(w) / w, mode="same")[k:-k]
    yy = yy[k:-k]
    if yy.size < 24:
        return empty
    det = sm - np.polyval(np.polyfit(yy, sm, 3), yy)

    om = 2.0 * np.pi / (2.0 * spacing)
    A = np.stack([np.cos(om * yy), np.sin(om * yy),
                  np.cos(2 * om * yy), np.sin(2 * om * yy),
                  np.ones_like(yy)], axis=1)
    c, *_ = np.linalg.lstsq(A, det, rcond=None)
    amp = 2.0 * float(np.hypot(c[0], c[1])) * 100.0
    res = float(np.std(det - A @ c)) * 100.0
    return {"peak_to_peak_cm": float(np.percentile(det, 97)
                                     - np.percentile(det, 3)) * 100.0,
            "amplitude_cm": amp, "residual_cm": res,
            "significance": float(amp / max(2.0 * res, 1e-6)),
            "n": int(yy.size)}


# ---------------------------------------------------------------------------


def crosssection(run_dir: Path, cfg, hf, rasters, out: Path) -> dict:
    x0 = cfg.render.section_x
    half = cfg.render.section_halfwidth
    lines = pass_lines(cfg)

    res = {}
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 2]})

    ys = np.linspace(0.0, cfg.scene.survey_y, 1400)
    q = np.stack([np.full_like(ys, x0), ys], axis=1)
    z_true = sample_bilinear(hf, q, "terrain")

    for ax, (label, colour) in zip(axes[:2], (("blue", BLUE), ("green", GREEN))):
        xyz = np.load(run_dir / "map" / f"{label}_xyz.npy", mmap_mode="r")
        m = np.abs(xyz[:, 0] - x0) < half
        pts = np.asarray(xyz[m])
        ax.scatter(pts[:, 1], pts[:, 2], s=0.35, alpha=0.25, c=colour,
                   linewidths=0, rasterized=True)
        ax.plot(ys, z_true, "-", color="w", lw=2.6, alpha=0.9, zorder=4)
        ax.plot(ys, z_true, "-", color=TRUTH, lw=1.2, zorder=5,
                label="true bare earth")
        for y in lines:
            ax.axvline(y, color="0.75", lw=0.5, ls=":", zorder=0)
        ax.set_ylabel("z, m")
        ax.set_xlim(0, cfg.scene.survey_y)
        lo, hi = float(np.nanmin(z_true)), float(np.nanmax(z_true))
        ax.set_ylim(lo - 1.5, hi + 3.0)
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    for label, colour in (("blue", BLUE), ("green", GREEN)):
        r = rasters[label]
        y, line = rast.profile(r, x0, 3.0, "ground")
        keep = (y >= 0) & (y <= cfg.scene.survey_y)
        y, line = y[keep], line[keep]
        z_ref = sample_bilinear(hf, np.stack([np.full_like(y, x0), y], axis=1),
                                "terrain")
        resid = line - z_ref
        ax.plot(y, resid * 100.0, "-", color=colour, lw=1.4, label=label)
        res[label] = ripple(y, resid, cfg.flight.spacing, cfg.scene.row_spacing)
        res[label]["rms_cm"] = float(np.sqrt(np.nanmean(resid ** 2)) * 100.0)
        res[label]["mean_cm"] = float(np.nanmean(resid) * 100.0)
        res[label]["p95_abs_cm"] = float(
            np.nanpercentile(np.abs(resid), 95) * 100.0)
    ax.axhline(0.0, color=TRUTH, lw=1.2)
    for y in lines:
        ax.axvline(y, color="0.75", lw=0.5, ls=":")
    ax.set_ylabel("bare-earth error, cm")
    ax.set_xlabel("y, m   (across the flight lines)")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=9)

    b, g = res["blue"], res["green"]
    axes[0].set_title(
        f"blue: the aircraft's own trajectory, nominal mount     "
        f"bare-earth error {b['rms_cm']:.0f} cm rms, "
        f"ripple {b['peak_to_peak_cm']:.0f} cm at the line spacing",
        loc="left", fontsize=11)
    axes[1].set_title(
        f"green: trajectory and mount solved together     "
        f"bare-earth error {g['rms_cm']:.0f} cm rms, "
        f"ripple {g['peak_to_peak_cm']:.0f} cm",
        loc="left", fontsize=11)
    axes[2].set_title("bare-earth surface minus the truth; dotted lines are the "
                      "flight lines", loc="left", fontsize=11)
    fig.suptitle(f"cross-section at x = {x0:.0f} m, {2*half:.0f} m thick, "
                 f"viewed edge-on", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "crosssection.png", dpi=120)
    plt.close(fig)
    return res


def ghost(run_dir: Path, cfg, reader, out: Path, size: float = 7.0) -> dict:
    """Close-up on a patch with a vertical in it, coloured by flight direction.

    The top-down view shows the crop rows doubling.  The edge-on view is the one
    that settles it: where the aircraft's own trajectory is used, the ground
    comes back as two separate sheets, one per flight direction, tens of
    centimetres apart, with the post smeared between them.  The same returns
    through the solved trajectory and mount give a single sheet and a post you
    could measure.
    """
    props = reader.props()
    inside = [p for p in props
              if p["kind"] in ("post", "scarecrow")
              and 15 < p["xy"][0] < cfg.scene.survey_x - 15
              and 15 < p["xy"][1] < cfg.scene.survey_y - 15]
    centre = (np.array(inside[len(inside) // 2]["xy"]) if inside
              else np.array([cfg.scene.survey_x / 2, cfg.scene.survey_y / 2]))
    near = [p for p in props
            if np.linalg.norm(np.array(p["xy"]) - centre) < size * 1.5]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    stats = {}
    for col, (label, colour) in enumerate((("blue", BLUE), ("green", GREEN))):
        xyz = np.load(run_dir / "map" / f"{label}_xyz.npy", mmap_mode="r")
        sw = np.load(run_dir / "map" / f"{label}_sweep.npy", mmap_mode="r")
        m = ((np.abs(xyz[:, 0] - centre[0]) < size)
             & (np.abs(xyz[:, 1] - centre[1]) < size))
        pts = np.asarray(xyz[m])
        traj = load_traj(run_dir / "est" / f"{label}_traj.npz")
        t = np.asarray(sw[m]).astype(np.float64) * reader.sweep_period
        R, _ = traj.at(t)
        out_bound = R[:, 0, 0] > 0
        groups = ((out_bound, "#D64A2E", "outbound"), (~out_bound, colour, "return"))

        ax = axes[0, col]
        for sel, c, name in groups:
            ax.scatter(pts[sel, 0], pts[sel, 1], s=0.7, alpha=0.35, c=c,
                       linewidths=0, label=name, rasterized=True)
        for pr in near:
            ax.plot(*pr["xy"], "k+", ms=11, mew=1.6)
        ax.set_title(f"{label}: from above", fontsize=11)
        ax.set_xlabel("x, m")
        ax.set_ylabel("y, m")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8, markerscale=8)

        ax = axes[1, col]
        strip = np.abs(pts[:, 0] - centre[0]) < 1.2
        for sel, c, name in groups:
            k = sel & strip
            ax.scatter(pts[k, 1], pts[k, 2], s=1.4, alpha=0.45, c=c,
                       linewidths=0, label=name, rasterized=True)
        ax.set_title(f"{label}: the same patch edge-on", fontsize=11)
        ax.set_xlabel("y, m")
        ax.set_ylabel("z, m")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=8, markerscale=6)

        # How far apart do the two directions put the ground?  That separation
        # is the ghosting, in centimetres.
        st = {"n_points": int(pts.shape[0])}
        edges = np.arange(centre[1] - size, centre[1] + size + 0.5, 0.5)
        seps = []
        for a, b in zip(edges[:-1], edges[1:]):
            cell = (pts[:, 1] >= a) & (pts[:, 1] < b) & strip
            za = pts[cell & out_bound, 2]
            zb = pts[cell & ~out_bound, 2]
            if za.size > 20 and zb.size > 20:
                seps.append(abs(np.median(za) - np.median(zb)))
        if seps:
            st["direction_split_cm"] = float(np.median(seps) * 100)
        stats[label] = st

    sb = stats["blue"].get("direction_split_cm")
    sg = stats["green"].get("direction_split_cm")
    extra = (f"     the two directions disagree by {sb:.0f} cm in blue and "
             f"{sg:.0f} cm in green" if sb and sg else "")
    fig.suptitle(f"close-up at ({centre[0]:.0f}, {centre[1]:.0f}) m, "
                 f"{2*size:.0f} m across, {len(near)} vertical(s) in frame\n"
                 f"outbound in red, return in colour{extra}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "ghost.png", dpi=120)
    plt.close(fig)
    stats["centre"] = [float(centre[0]), float(centre[1])]
    stats["props_in_frame"] = len(near)
    return stats


def boresight_card(rep: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")
    rows = [("nominal", rep["nominal_rpy_deg"], ""),
            ("recovered", rep["recovered_rpy_deg"],
             "+/- " + "  ".join(f"{s:.3f}" for s in rep["recovered_sigma_deg"]))]
    if "true_rpy_deg" in rep:
        rows.append(("true", rep["true_rpy_deg"], "(gate only)"))

    ax.text(0.02, 0.90, "sensor mounting angle,  R_BS  as roll / pitch / yaw",
            fontsize=14, transform=ax.transAxes)
    ax.text(0.02, 0.76, f"{'':<12}{'roll':>10}{'pitch':>11}{'yaw':>10}",
            fontsize=12, family="monospace", transform=ax.transAxes)
    for i, (name, rpy, extra) in enumerate(rows):
        ax.text(0.02, 0.64 - 0.12 * i,
                f"{name:<12}{rpy[0]:10.3f}{rpy[1]:11.3f}{rpy[2]:10.3f}   {extra}",
                fontsize=12, family="monospace", transform=ax.transAxes)
    if "error_deg" in rep:
        verdict = "PASS" if rep["pass"] else "FAIL"
        ax.text(0.02, 0.16,
                f"nominal was {rep['nominal_error_deg']:.3f} deg from the truth;  "
                f"recovered is {rep['error_deg']:.3f} deg   "
                f"[gate 5 < {rep['gate_deg']:.1f}: {verdict}]",
                fontsize=12, transform=ax.transAxes,
                color=GREEN if rep["pass"] else "#B00020")
    fig.tight_layout()
    fig.savefig(out / "boresight.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------


def run(run_dir: Path) -> dict:
    reader = RunReader(run_dir)
    cfg = load_run_config(run_dir)
    hf = heightfield_of(reader)
    out = run_dir / "gates"
    out.mkdir(parents=True, exist_ok=True)

    cell = int(cfg.map.cell * 100)
    rasters = {k: rast.load(run_dir / "map" / f"{k}_raster_{cell}cm.npz")
               for k in ("blue", "green")}

    report = {"crosssection": crosssection(run_dir, cfg, hf, rasters, out),
              "ghost": ghost(run_dir, cfg, reader, out)}

    bore = json.loads((run_dir / "est" / "boresight.json").read_text())
    boresight_card(bore, out)
    report["boresight"] = {k: bore[k] for k in
                           ("nominal_rpy_deg", "recovered_rpy_deg",
                            "recovered_sigma_deg", "true_rpy_deg", "error_deg",
                            "nominal_error_deg", "pass") if k in bore}

    for name in ("blue", "green"):
        if f"{name}_vs_truth" in bore:
            report.setdefault("trajectory", {})[name] = bore[f"{name}_vs_truth"]

    b = report["crosssection"]["blue"]
    g = report["crosssection"]["green"]
    traj = report.get("trajectory", {})
    gates = {
        "gate2_blue_drift": {
            "xy_rms_m": traj.get("blue", {}).get("xy_rms_m"),
            "xy_p95_m": traj.get("blue", {}).get("xy_p95_m"),
            "pass": bool(1.0 <= traj.get("blue", {}).get("xy_rms_m", 0) <= 2.5),
        },
        "gate3_corrugation": {
            "peak_to_peak_cm": b["peak_to_peak_cm"],
            "at_line_spacing_cm": b["amplitude_cm"],
            "significance": b["significance"],
            "rms_cm": b["rms_cm"],
            "pass": bool(b["peak_to_peak_cm"] > 10.0 and b["amplitude_cm"] > 3.0),
        },
        # Gate 4 asks whether green visibly hugs the truth.  What that means
        # numerically is that its bare-earth surface sits within a few
        # centimetres of the real one and is several times closer than blue.  A
        # small positive bias survives in both, because a bare-earth percentile
        # taken through a standing crop can only ever sit at or above the soil.
        "gate4_green_hugs_truth": {
            "rms_cm": g["rms_cm"],
            "p95_abs_cm": g["p95_abs_cm"],
            "peak_to_peak_cm": g["peak_to_peak_cm"],
            "improvement": (b["rms_cm"] / g["rms_cm"]) if g["rms_cm"] else 0.0,
            "pass": bool(g["rms_cm"] < 15.0 and g["rms_cm"] < 0.5 * b["rms_cm"]),
        },
        "gate2b_ghosting": {
            "blue_direction_split_cm": report["ghost"]["blue"].get("direction_split_cm"),
            "green_direction_split_cm": report["ghost"]["green"].get("direction_split_cm"),
            "pass": bool(report["ghost"]["blue"].get("direction_split_cm", 0) > 10.0
                         and report["ghost"]["green"].get("direction_split_cm", 99) < 6.0),
        },
        "gate5_boresight": {
            "error_deg": report["boresight"].get("error_deg"),
            "pass": report["boresight"].get("pass"),
        },
    }
    report["gates"] = gates
    report["pass"] = all(bool(v.get("pass")) for v in gates.values())
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    rep = run(run_dir)

    g = rep["gates"]
    print(f"gate 2  blue drift          {g['gate2_blue_drift']['xy_rms_m']:.2f} m rms "
          f"(p95 {g['gate2_blue_drift']['xy_p95_m']:.2f})"
          f"   {'PASS' if g['gate2_blue_drift']['pass'] else 'FAIL'}")
    c = g["gate3_corrugation"]
    print(f"gate 3  corrugation         {c['peak_to_peak_cm']:.1f} cm peak-to-peak, "
          f"{c['at_line_spacing_cm']:.1f} cm at the line spacing "
          f"({c['significance']:.1f}x background)"
          f"   {'PASS' if c['pass'] else 'FAIL'}")
    h = g["gate4_green_hugs_truth"]
    print(f"gate 4  green vs truth      {h['rms_cm']:.1f} cm rms, "
          f"{h['improvement']:.1f}x better than blue"
          f"   {'PASS' if h['pass'] else 'FAIL'}")
    gh = g["gate2b_ghosting"]
    print(f"gate 2b flight directions   disagree by "
          f"{gh['blue_direction_split_cm']:.0f} cm in blue, "
          f"{gh['green_direction_split_cm']:.1f} cm in green"
          f"   {'PASS' if gh['pass'] else 'FAIL'}")
    b = g["gate5_boresight"]
    print(f"gate 5  mounting angle      {b['error_deg']:.3f} deg"
          f"   {'PASS' if b['pass'] else 'FAIL'}")
    print(f"\nwrote {run_dir/'gates'}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
