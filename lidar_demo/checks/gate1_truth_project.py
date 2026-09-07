"""Gate 1: project the recorded sweeps through the truth trajectory.

Two runs of the same check, and they have to disagree.

``--extrinsic truth`` uses the mount the simulator actually built.  The points
must land on the terrain to within the range noise.  If they do not, the
recording is broken and nothing downstream can be trusted.

``--extrinsic nominal`` uses the mount the reconstruction is told about.  Two
distinct errors appear, and it is worth being precise about which is which,
because only one of them alternates.

Write the mount error as a small rotation ``eps`` in the body frame.  A point at
body offset ``p`` is displaced by ``-eps x p``, whose vertical part is
``-(eps_x p_y - eps_y p_x)``.

* The ``eps_y`` term multiplies the *along-track* offset.  The sensor is tilted
  forward, so returns land ahead of the aircraft on every pass, and "ahead"
  reverses in world coordinates exactly when the heading does.  The two
  reversals cancel: every pass is lifted the same way.  This is a bulk offset of
  the whole map, and it is the larger of the two.
* The ``eps_x`` term multiplies the *across-track* offset, which does **not**
  reverse with heading.  So the swath is tilted one way flying out and the other
  way flying back.  Neighbouring swaths disagree about where the ground is, in
  opposite senses, and the surface comes out corrugated with one ridge per
  flight line.

Measuring the second one needs care.  The usable footprint is a forward-looking
patch, so a point's across-track and along-track offsets are correlated, and a
plain fit of height error against across-track distance picks up a large slice of
the along-track term.  Both offsets are therefore regressed out together, and
the reported swath tilt is the across-track coefficient alone.

The check that actually decides the verdict is simpler and is the same
measurement gate 3 will make on the finished map: take a bare-earth profile
across the flight lines, remove the terrain, and look for a ripple whose period
is the flight-line spacing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import DemoConfig
from ..io import RunReader
from ..sim.terrain import Heightfield, sample_bilinear

GROUND = 1


def heightfield_of(reader: RunReader) -> Heightfield:
    d = reader.heightfield()
    return Heightfield(x0=float(d["x0"]), y0=float(d["y0"]), dx=float(d["dx"]),
                       z_terrain=d["z_terrain"], z_surface=d["z_surface"],
                       canopy_h=d["canopy_h"], class_id=d["class_id"],
                       ditch_p0=tuple(d["ditch_xy"][0]), ditch_p1=tuple(d["ditch_xy"][1]))


def project(reader: RunReader, which: str, every: int = 5):
    """Project every ``every``-th sweep and label the points.

    Returns world points, surface class, pass id, a turn flag, and each point's
    offset in the body frame at the instant it was measured.  The turn flag
    matters more than it looks: during a reversal the aircraft yaws through 180
    degrees and tilts by twenty, so those returns belong to neither heading.
    """
    E = reader.truth_extrinsic() if which == "truth" else reader.extrinsic_nominal
    truth = reader.truth_arrays()
    t_truth = truth["t"]
    pass_of_t, turn_of_t = truth["pass_id"], truth["is_turn"]

    P, cls, pid, turn, off = [], [], [], [], []
    for k in range(0, reader.n_sweeps, every):
        sw = reader.sweep(k)
        if len(sw.xyz) == 0:
            continue
        tr = reader.truth_sweep(k)
        col = sw.col.astype(int)
        R_WB = tr["col_R_WB"][col].astype(np.float64)
        p_WB = tr["col_p_WB"][col]
        p_B = sw.xyz.astype(np.float64) @ E.R.T + E.t
        P.append(np.einsum("nij,nj->ni", R_WB, p_B) + p_WB)
        off.append(p_B)
        cls.append(tr["hit_class"])
        j = min(int(np.searchsorted(t_truth, sw.t_start)), len(pass_of_t) - 1)
        n = len(sw.xyz)
        pid.append(np.full(n, pass_of_t[j]))
        turn.append(np.full(n, bool(turn_of_t[j])))
    return (np.concatenate(P), np.concatenate(cls), np.concatenate(pid),
            np.concatenate(turn), np.concatenate(off))


def pass_lines(cfg: DemoConfig) -> dict[int, float]:
    """Across-track position of each boustrophedon line."""
    return {i: cfg.scene.survey_y - i * cfg.flight.spacing
            for i in range(cfg.flight.n_passes)}


def bare_earth_profile(P, dz, x0: float, half: float, sy: float, bin_m: float = 0.5,
                       pct: float = 20.0, min_pts: int = 25):
    """Height error of the lowest returns, binned across the flight lines.

    A bare-earth surface is built from low percentiles per cell, so that is what
    the corrugation has to show up in.  Returns ``(y, residual)`` with empty bins
    dropped.
    """
    band = np.abs(P[:, 0] - x0) < half
    y, e = P[band, 1], dz[band]
    edges = np.arange(0.0, sy + bin_m, bin_m)
    idx = np.digitize(y, edges) - 1
    ys, prof = [], []
    for i in range(len(edges) - 1):
        s = e[idx == i]
        if s.size >= min_pts:
            ys.append(0.5 * (edges[i] + edges[i + 1]))
            prof.append(np.percentile(s, pct))
    return np.asarray(ys), np.asarray(prof)


def corrugation(ys: np.ndarray, prof: np.ndarray, spacing: float,
                row_spacing: float) -> dict:
    """Strength of a ripple at the flight-line spacing, in centimetres.

    Two other periodic signals live in this profile and both have to go first.
    The long one is the terrain, removed with a cubic fit in ``y``.  The short
    one is an alias of the crop furrows: a boresight yaw error slides every
    point sideways by tens of centimetres, so each point is compared against the
    terrain at slightly the wrong place, and a 10 cm furrow at 2 m pitch becomes
    a 20 cm ripple at 2 m pitch.  That is a real error but it is not the
    washboard, and smoothing over one and a half row spacings removes it.

    What is left is measured by fitting a single sinusoid at a known period
    rather than by picking an FFT peak.  The profile spans only a handful of
    flight lines, so its spectrum has barely a few usable bins and the peak
    wanders; a lock-in at the expected period is sharper and is honest about
    what is being asked, which is not "what period is strongest" but "how much
    ripple is there where the flight lines put it".

    That period is **twice** the line spacing, and the factor of two is the
    signature rather than an accident.  A boresight roll error tilts an outbound
    swath one way and a return swath the other, so the low surface between an
    outbound line and the return line below it sits high, and between that
    return line and the next outbound line it sits low.  The pattern repeats
    every *pair* of lines.  One peak or trough still falls on each flight line,
    which is the washboard an aerial LiDAR practitioner recognises on sight; the
    fundamental simply lives an octave down.
    """
    empty = {"peak_to_peak_cm": 0.0, "amplitude_cm": 0.0, "harmonic_cm": 0.0,
             "residual_cm": 0.0, "expected_period_m": 2.0 * spacing,
             "significance": 0.0, "period_m": 0.0, "period_ratio": 0.0}
    if ys.size < 30:
        return dict(empty)

    bin_m = float(np.median(np.diff(ys))) if ys.size > 1 else 0.5
    short = max(int(round(1.5 * row_spacing / bin_m)) | 1, 3)
    k = short // 2
    smooth = np.convolve(prof, np.ones(short) / short, mode="same")[k:-k]
    y = ys[k:-k]
    if y.size < 24:
        return dict(empty)

    det = smooth - np.polyval(np.polyfit(y, smooth, 3), y)

    expected = 2.0 * spacing
    w = 2.0 * np.pi / expected
    A = np.stack([np.cos(w * y), np.sin(w * y),
                  np.cos(2 * w * y), np.sin(2 * w * y),
                  np.ones_like(y)], axis=1)
    coef, *_ = np.linalg.lstsq(A, det, rcond=None)
    amp = 2.0 * float(np.hypot(coef[0], coef[1])) * 100.0
    amp_h = 2.0 * float(np.hypot(coef[2], coef[3])) * 100.0
    resid = float(np.std(det - A @ coef)) * 100.0

    spec = np.abs(np.fft.rfft(det - det.mean()))
    freq = np.fft.rfftfreq(det.size, bin_m)
    band = np.zeros_like(spec, dtype=bool)
    band[1:] = (1.0 / freq[1:] > 0.5 * spacing) & (1.0 / freq[1:] < 4.0 * spacing)
    j = int(np.argmax(np.where(band, spec, 0.0)))
    period = float(1.0 / freq[j]) if j > 0 else 0.0

    # The ripple is a sawtooth rather than a sine -- each swath tilts steadily
    # and then the next one takes over -- so its peak-to-peak swing is the
    # number a viewer actually sees, while the lock-in says whether that swing
    # is organised at the flight lines or is just noise.
    p2p = float(np.percentile(det, 97) - np.percentile(det, 3)) * 100.0

    return {"peak_to_peak_cm": p2p,
            "amplitude_cm": amp,
            "harmonic_cm": amp_h,
            "residual_cm": resid,
            "expected_period_m": expected,
            "significance": float(amp / max(2.0 * resid, 1e-6)),
            "period_m": period,
            "period_ratio": period / expected if period else 0.0}


def analyse(cfg: DemoConfig, hf: Heightfield, P, cls, pid, turn, off,
            every: int) -> dict:
    sx, sy = cfg.scene.survey_x, cfg.scene.survey_y
    # Along-track the survey edges are trimmed, because the aircraft is turning
    # there.  Across-track nothing is trimmed: a swath overhangs the survey
    # rectangle, and clipping it would leave the first and last passes fitted on
    # half a swath, which is exactly where the tilt estimate goes wrong.
    margin = 1.6 * cfg.derived.footprint_radius
    inside = ((P[:, 0] > 5.0) & (P[:, 0] < sx - 5.0)
              & (P[:, 1] > -margin) & (P[:, 1] < sy + margin))
    ground = (cls == GROUND) & inside & ~turn
    dz = P[:, 2] - sample_bilinear(hf, P[:, :2], "terrain")
    dz_g, pid_g, P_g, off_g = dz[ground], pid[ground], P[ground], off[ground]

    in_survey = (P_g[:, 1] > 0.0) & (P_g[:, 1] < sy)

    lines = pass_lines(cfg)
    per_pass = {}
    for k, y_line in lines.items():
        m = pid_g == k
        if m.sum() < 500:
            continue
        d = P_g[m, 1] - y_line
        keep = np.abs(d) < 1.3 * cfg.derived.footprint_radius
        if keep.sum() < 500:
            continue
        # Regress the height error on across-track offset *and* along-track
        # offset together.  The footprint is a forward patch, so the two are
        # correlated, and fitting the across-track term alone would absorb a
        # large slice of the constant lift.
        a = np.linalg.norm(off_g[m][keep][:, :2], axis=1)
        A = np.stack([d[keep], a, np.ones(keep.sum())], axis=1)
        coef, *_ = np.linalg.lstsq(A, dz_g[m][keep], rcond=None)
        per_pass[int(k)] = {
            "n": int(m.sum()),
            "slope_deg": float(np.degrees(np.arctan(coef[0]))),
            "mean_dz_m": float(dz_g[m].mean()),
            "half_swath_m": float(np.percentile(np.abs(d), 95)),
        }

    slopes = [per_pass[k]["slope_deg"] for k in sorted(per_pass)]
    flips = sum(1 for a, b in zip(slopes, slopes[1:]) if a * b < 0)
    gap = float(np.mean([abs(a - b) for a, b in zip(slopes, slopes[1:])])) if len(slopes) > 1 else 0.0

    ys, prof = bare_earth_profile(P_g[in_survey], dz_g[in_survey],
                                  cfg.render.section_x, 20.0, sy)
    corr = corrugation(ys, prof, cfg.flight.spacing, cfg.scene.row_spacing)

    cell = 1.0
    nx, ny = int(sx / cell), int(sy / cell)

    def density(pts):
        ix = np.clip((pts[:, 0] / cell).astype(int), 0, nx - 1)
        iy = np.clip((pts[:, 1] / cell).astype(int), 0, ny - 1)
        c = np.bincount(iy * nx + ix, minlength=nx * ny)
        return c[c > 0] * every / (cell * cell)

    survey_all = ((P[:, 0] > 0) & (P[:, 0] < sx) & (P[:, 1] > 0) & (P[:, 1] < sy))
    dens_all = density(P[survey_all])
    dens = density(P_g[in_survey])

    return {
        "n_points": int(P.shape[0]),
        "n_ground": int(ground.sum()),
        "ground_rms_m": float(np.sqrt((dz_g ** 2).mean())),
        "ground_mean_m": float(dz_g.mean()),
        "ground_p95_abs_m": float(np.percentile(np.abs(dz_g), 95)),
        "density_all_median_per_m2": float(np.median(dens_all)),
        "density_median_per_m2": float(np.median(dens)),
        "density_p10_per_m2": float(np.percentile(dens, 10)),
        "half_swath_m": float(np.median([v["half_swath_m"] for v in per_pass.values()])),
        "per_pass": per_pass,
        "slope_sign_flips": int(flips),
        "slope_sign_flips_possible": max(len(slopes) - 1, 0),
        "mean_abs_slope_deg": float(np.mean(np.abs(slopes))) if slopes else 0.0,
        "mean_adjacent_slope_gap_deg": gap,
        "corrugation": corr,
        "_profile": (ys, prof),
    }


def verdict(which: str, res: dict) -> bool:
    """Gate 1 passes when both projections behave as the physics says they must."""
    if which == "truth":
        # The cloud reproduces the terrain, at the density the spec asks for,
        # with no swath structure at all.
        return bool(res["ground_rms_m"] < 0.05
                    and abs(res["ground_mean_m"]) < 0.02
                    and 300.0 <= res["density_all_median_per_m2"] <= 1000.0
                    and res["corrugation"]["peak_to_peak_cm"] < 3.0
                    and res["mean_abs_slope_deg"] < 0.05)
    # The nominal mount lifts the whole map and tilts alternate swaths in
    # opposite directions, leaving a ripple at the flight-line spacing.
    # The decisive measurement is the swath tilt: it is the mechanism itself,
    # measured per pass, and it has to alternate.  The profile ripple is the
    # visible consequence and is checked more loosely here, because gate 3 will
    # judge it properly on the finished map.
    n = res["slope_sign_flips_possible"]
    c = res["corrugation"]
    return bool(res["ground_rms_m"] > 0.10
                and res["slope_sign_flips"] >= max(n - 1, 1)
                and res["mean_abs_slope_deg"] > 0.3
                and c["peak_to_peak_cm"] > 5.0)


def figure(cfg, hf, P, cls, pid, turn, which, res, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    x0 = cfg.render.section_x
    ground = (cls == GROUND) & ~turn
    band = (np.abs(P[:, 0] - x0) < 1.5) & ground

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.2))

    ax = axes[0]
    ys = np.linspace(0.0, cfg.scene.survey_y, 900)
    q = np.stack([np.full_like(ys, x0), ys], axis=1)
    ax.scatter(P[band, 1], P[band, 2], s=0.7, alpha=0.35, c=pid[band],
               cmap="coolwarm", linewidths=0)
    ax.plot(ys, sample_bilinear(hf, q, "terrain"), "-", color="k", lw=1.5,
            label="true bare earth")
    for _, y in pass_lines(cfg).items():
        ax.axvline(y, color="0.7", lw=0.5, ls=":")
    ax.set_title(f"cross-section at x = {x0:.0f} m ({which} mount)\n"
                 "colour = pass, dotted = flight lines")
    ax.set_xlabel("y, m")
    ax.set_ylabel("z, m")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ys_p, prof = res["_profile"]
    ax.plot(ys_p, prof * 100.0, "-", lw=1.3, color="#2E6FD6")
    ax.axhline(0.0, color="k", lw=1.0)
    for _, y in pass_lines(cfg).items():
        ax.axvline(y, color="0.7", lw=0.5, ls=":")
    c = res["corrugation"]
    ax.set_title(f"bare-earth height error across the lines\n"
                 f"ripple {c['peak_to_peak_cm']:.1f} cm peak-to-peak, one feature "
                 f"per {cfg.flight.spacing:.0f} m flight line")
    ax.set_xlabel("y, m")
    ax.set_ylabel("error, cm")
    ax.grid(alpha=0.2)

    ax = axes[2]
    for k in sorted(res["per_pass"]):
        y_line = pass_lines(cfg)[k]
        m = (pid == k) & ground & (np.abs(P[:, 1] - y_line) < 16.0)
        if m.sum() < 200:
            continue
        d = P[m, 1] - y_line
        e = P[m, 2] - sample_bilinear(hf, P[m, :2], "terrain")
        bins = np.linspace(-14, 14, 29)
        idx = np.digitize(d, bins) - 1
        prof_k = [np.median(e[idx == i]) if (idx == i).sum() > 30 else np.nan
                  for i in range(len(bins) - 1)]
        ax.plot(0.5 * (bins[:-1] + bins[1:]), np.array(prof_k) * 100.0, lw=1.2,
                color="#2E6FD6" if k % 2 == 0 else "#B00020",
                label=("outbound" if k == 0 else "return" if k == 1 else None))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("error across the swath, by pass")
    ax.set_xlabel("across-track offset from the flight line, m")
    ax.set_ylabel("dz, cm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[3]
    ks = sorted(res["per_pass"])
    sl = [res["per_pass"][k]["slope_deg"] for k in ks]
    ax.bar([str(k) for k in ks], sl,
           color=["#2E6FD6" if v < 0 else "#B00020" for v in sl])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("across-track tilt of each swath\n"
                 "alternating sign = the washboard")
    ax.set_xlabel("pass")
    ax.set_ylabel("tilt, deg")
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle(
        f"gate 1 [{which} mount]   ground RMS {res['ground_rms_m']*100:.1f} cm, "
        f"mean {res['ground_mean_m']*100:+.1f} cm, "
        f"density {res['density_all_median_per_m2']:.0f} pts/m2, "
        f"half swath {res['half_swath_m']:.1f} m, "
        f"tilt flips {res['slope_sign_flips']}/{res['slope_sign_flips_possible']}, "
        f"ripple {res['corrugation']['peak_to_peak_cm']:.1f} cm, one per flight line"
        f"   ->  {'PASS' if res['pass'] else 'FAIL'}")
    fig.tight_layout()
    path = out_dir / f"gate1_{which}.png"
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return path


def run(run_dir: Path, which: str, every: int) -> dict:
    reader = RunReader(run_dir)
    cfg = DemoConfig.from_yaml(run_dir / "config_resolved.yaml")
    hf = heightfield_of(reader)
    P, cls, pid, turn, off = project(reader, which, every)
    res = analyse(cfg, hf, P, cls, pid, turn, off, every)
    res["extrinsic"] = which
    res["every"] = every
    res["pass"] = verdict(which, res)
    figure(cfg, hf, P, cls, pid, turn, which, res, run_dir / "gates")
    res.pop("_profile")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--extrinsic", default="both",
                    choices=["truth", "nominal", "both"])
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    which = ["truth", "nominal"] if args.extrinsic == "both" else [args.extrinsic]

    results, ok = {}, True
    for w in which:
        r = run(run_dir, w, args.every)
        results[w] = r
        ok &= r["pass"]
        c = r["corrugation"]
        print(f"\n[{w} mount]")
        print(f"  ground points        {r['n_ground']:,} of {r['n_points']:,}")
        print(f"  error vs bare earth  RMS {r['ground_rms_m']*100:7.2f} cm   "
              f"mean {r['ground_mean_m']*100:+7.2f} cm   "
              f"p95 {r['ground_p95_abs_m']*100:7.2f} cm")
        print(f"  density              all {r['density_all_median_per_m2']:.0f} pts/m2, "
              f"ground {r['density_median_per_m2']:.0f} pts/m2 "
              f"(p10 {r['density_p10_per_m2']:.0f})")
        print(f"  half swath           {r['half_swath_m']:.1f} m")
        print("  swath tilt per pass  " + "  ".join(
            f"{k}:{v['slope_deg']:+.3f}" for k, v in sorted(r["per_pass"].items())))
        print(f"  tilt sign flips      {r['slope_sign_flips']}"
              f"/{r['slope_sign_flips_possible']}   "
              f"mean |tilt| {r['mean_abs_slope_deg']:.3f} deg")
        spacing = DemoConfig.from_yaml(run_dir / "config_resolved.yaml").flight.spacing
        print(f"  corrugation          {c['peak_to_peak_cm']:.1f} cm peak-to-peak, "
              f"{c['amplitude_cm']:.1f} cm of it at {c['expected_period_m']:.0f} m "
              f"(one feature per {spacing:.0f} m flight line), "
              f"{c['significance']:.1f}x the {c['residual_cm']:.1f} cm background")
        print(f"  -> {'PASS' if r['pass'] else 'FAIL'}")

    out = run_dir / "gates"
    (out / "gate1.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out/'gate1.json'} and gate1_*.png")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
