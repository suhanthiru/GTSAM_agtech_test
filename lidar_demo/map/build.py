"""Build the map products for both reconstructions.

Writes, for blue and for green: the dense world cloud as raw ``.npy`` for the
renderer, a coloured PLY for anything else, and the bare-earth and canopy-height
rasters.

The two clouds are projected with identical decimation so they come out with the
same points in the same order.  That is what lets the render interpolate each
point from where blue put it to where green put it, one for one, instead of
cross-fading two unrelated images.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..config import load, load_run_config
from ..io import RunReader, load_traj, write_ply_points
from . import raster as rast
from .project import project_run

BLUE = (0x2E, 0x6F, 0xD6)
GREEN = (0x1D, 0x9E, 0x75)


def colourise(cloud, rgb) -> np.ndarray:
    """Flat hue, brightness from intensity, so the cloud reads as one surface."""
    v = np.clip(cloud.intensity.astype(np.float32), 0.0, 1.0)
    shade = (0.45 + 0.55 * (v / max(float(v.max()), 1e-6)))[:, None]
    return np.clip(np.asarray(rgb, np.float32)[None, :] * shade, 0, 255).astype(np.uint8)


def build_one(reader: RunReader, traj, label: str, rgb, out: Path, cfg,
              bounds, stride: int, sweep_step: int, verbose=True) -> dict:
    t0 = time.time()

    def tick(k, n):
        if verbose:
            print(f"  {label}: sweep {k}/{n}", flush=True)

    cloud = project_run(reader, traj, traj.extrinsic, stride, sweep_step, tick)
    if verbose:
        print(f"  {label}: {len(cloud)/1e6:.2f} M points [{time.time()-t0:.0f} s]")

    np.save(out / f"{label}_xyz.npy", cloud.xyz)
    np.save(out / f"{label}_intensity.npy", cloud.intensity)
    np.save(out / f"{label}_sweep.npy", cloud.sweep)
    write_ply_points(out / f"{label}.ply", cloud.xyz, colourise(cloud, rgb))

    r = rast.rasterise(cloud.xyz, cfg.map.cell, bounds, cfg.map.ground_pct,
                       cfg.map.canopy_pct, cfg.map.min_pts_per_cell, label)
    rast.save(r, out / f"{label}_raster_{int(cfg.map.cell*100)}cm.npz")
    if verbose:
        print(f"  {label}: raster {r.ground.shape}, {r.coverage():.0%} of cells "
              f"have ground")
    return {"n_points": len(cloud), "raster_coverage": r.coverage(),
            "extrinsic_rpy_deg": [float(v) for v in traj.extrinsic.rpy_deg]}


def build(run_dir: Path, cfg, which=("blue", "green"), stride: int | None = None,
          sweep_step: int = 1, verbose: bool = True) -> dict:
    reader = RunReader(run_dir)
    run_cfg = load_run_config(run_dir)
    out = run_dir / "map"
    out.mkdir(parents=True, exist_ok=True)
    stride = int(cfg.map.stride if stride is None else stride)

    s = run_cfg.scene
    pad = 5.0
    bounds = (-pad, s.survey_x + pad, s.treeline_y - pad, s.survey_y + pad)

    summary = {"stride": stride, "sweep_step": sweep_step,
               "cell": cfg.map.cell, "bounds": list(bounds)}
    for label, rgb in (("blue", BLUE), ("green", GREEN)):
        if label not in which:
            continue
        path = run_dir / "est" / f"{label}_traj.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run lidar_demo.est.solve first")
        traj = load_traj(path)
        if traj.extrinsic is None:
            raise ValueError(f"{path} carries no extrinsic")
        if verbose:
            print(f"{label}: mount {np.round(traj.extrinsic.rpy_deg, 3)} deg")
        summary[label] = build_one(reader, traj, label, rgb, out, cfg, bounds,
                                   stride, sweep_step, verbose)

    if {"blue", "green"} <= set(which):
        nb = summary["blue"]["n_points"]
        ng = summary["green"]["n_points"]
        if nb != ng:
            raise AssertionError(
                f"blue and green clouds differ in size ({nb} vs {ng}); the morph "
                "needs one-to-one correspondence between them")
        summary["point_correspondence"] = "one-to-one"

    (out / "map.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--which", default="both", choices=["blue", "green", "both"])
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--sweep-step", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = load(args.config)
    which = ("blue", "green") if args.which == "both" else (args.which,)
    s = build(Path(args.run), cfg, which, args.stride, args.sweep_step)
    print(json.dumps({k: v for k, v in s.items() if k != "bounds"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
