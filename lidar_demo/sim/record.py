"""Record one run: fly the pattern, cast every sweep, write the run directory.

The estimator does not run in here.  Record first, solve offline: debugging a
factor graph inside a simulator is miserable, and separating them is also what
makes the two reconstructions comparable, because both read exactly the same
recorded returns.

Random draws are split into independent streams -- scene, IMU, GNSS, LiDAR -- so
retuning the sensor noise does not reshuffle the terrain, and a seed means the
same thing across runs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..config import DemoConfig, load
from ..frames import rpy_zyx_to_R
from ..io import Extrinsic, RunWriter, Sweep, build_manifest
from .backend import make_backend
from .flight import plan_survey, truth_on_clock
from .gnss import dropout_summary, simulate_gnss
from .imu import drift_report, simulate_imu
from .lidar import LidarModel, simulate_sweep
from .scene import build_scene, export_scene


def true_extrinsic(cfg: DemoConfig) -> Extrinsic:
    """The mount the airframe actually has.

    The offset is added to the nominal Euler angles, so the true mount reads as
    roll 0.8, pitch 36.5, yaw 2.0 against a nominal 0, 35, 0.  That is what goes
    on the title card, side by side with the recovered value.  The gate itself
    compares rotations rather than Euler triples, so it stays convention-free
    either way.
    """
    rpy = (np.asarray(cfg.extrinsic.nominal_rpy_deg, dtype=float)
           + np.asarray(cfg.extrinsic.true_rpy_offset_deg, dtype=float))
    t = np.array(cfg.extrinsic.nominal_t) + np.array(cfg.extrinsic.true_t_offset)
    return Extrinsic(rpy_zyx_to_R(*rpy), t)


def record(cfg: DemoConfig, out_root: str | Path | None = None,
           limit: int | None = None, verbose: bool = True) -> Path:
    root = Path(out_root or cfg.run.out_root) / cfg.run.name
    writer = RunWriter(root)
    seeds = np.random.SeedSequence(cfg.run.seed).spawn(4)

    t0 = time.time()
    scene = build_scene(cfg.scene, np.random.default_rng(seeds[0]))
    export_scene(scene, root / "scene")
    if verbose:
        print(f"scene: {scene.n_triangles/1e6:.2f} M triangles, "
              f"{len(scene.props)} props  [{time.time()-t0:.1f} s]")

    path = plan_survey(cfg.flight, cfg.scene)
    truth = truth_on_clock(path, cfg.imu.rate_hz)

    accel, gyro, b_a, b_g = simulate_imu(cfg.imu, truth, np.random.default_rng(seeds[1]))
    writer.write_imu(truth.t, accel, gyro)

    t_g, idx_g, p_g, valid, sigma = simulate_gnss(
        cfg.gnss, truth, cfg.scene.treeline_y, np.random.default_rng(seeds[2]))
    writer.write_gnss(t_g, p_g, valid, sigma, idx_g)

    writer.write_truth_trajectory(
        t=truth.t, p_WB=truth.position, v_WB=truth.velocity, a_WB=truth.acceleration,
        R_WB=truth.rotation, omega_B=truth.omega, bias_accel=b_a, bias_gyro=b_g,
        pass_id=truth.pass_id, leg_id=truth.leg_id, is_turn=truth.is_turn)

    E_true = true_extrinsic(cfg)
    writer.write_truth_extrinsic(E_true)

    if verbose:
        rep = drift_report(cfg.imu, truth, accel, gyro, b_a, b_g)
        ds = dropout_summary(t_g, valid)
        print(rep)
        print(f"GNSS: {int(valid.sum())}/{valid.size} valid, longest outage "
              f"{ds['longest_s']:.0f} s")
        print(f"true mount rpy {np.round(E_true.rpy_deg, 3)} deg "
              f"(nominal {list(cfg.extrinsic.nominal_rpy_deg)}) -- written to "
              f"truth/, never to the manifest")

    # ---- sweeps ----
    model = LidarModel.from_cfg(cfg.lidar)
    backend = make_backend(cfg.run.backend)
    backend.build(scene)

    n_sweeps = int(np.floor(path.duration / model.period))
    if limit is not None:
        n_sweeps = min(n_sweeps, limit)

    rng = np.random.default_rng(seeds[3])
    t1 = time.time()
    total_pts = 0
    for k in range(n_sweeps):
        raw = simulate_sweep(model, backend, path, k, E_true.R, E_true.t,
                             cfg.lidar, rng)
        writer.write_sweep(Sweep(index=k, t_start=raw.t_start, t_offset=raw.t_offset,
                                 xyz=raw.xyz, range=raw.range, intensity=raw.intensity,
                                 ring=raw.ring, col=raw.col))
        if cfg.run.write_sweep_truth:
            writer.write_sweep_truth(k, raw.col_t, raw.col_R_WB, raw.col_p_WB,
                                     raw.hit_class)
        total_pts += len(raw.xyz)
        if verbose and (k + 1) % 200 == 0:
            el = time.time() - t1
            print(f"  sweep {k+1}/{n_sweeps}  {total_pts/1e6:.1f} M pts  "
                  f"{el:.0f} s elapsed, {el/(k+1)*(n_sweeps-k-1):.0f} s left")

    manifest = build_manifest(cfg, n_sweeps, np.degrees(model.ring_el),
                              backend.name,
                              extra={"flight": _flight_manifest(cfg, path),
                                     "recording": {
                                         "duration_s": float(path.duration),
                                         "total_points": int(total_pts),
                                         "mean_points_per_sweep": float(total_pts / max(n_sweeps, 1)),
                                         "wall_seconds": float(time.time() - t1)}})
    writer.write_manifest(manifest)
    cfg.to_yaml(root / "config_resolved.yaml")

    if verbose:
        print(f"wrote {n_sweeps} sweeps, {total_pts/1e6:.1f} M points to {root} "
              f"[{time.time()-t1:.0f} s]")
    return root


def _flight_manifest(cfg: DemoConfig, path) -> dict:
    leg_start, _, total = path._schedule()
    passes = []
    for i, leg in enumerate(path.legs):
        d = leg.direction
        passes.append({
            "pass_id": int(leg.pass_id),
            "name": leg.name,
            "y" if abs(d[1]) < 0.5 else "x": float(leg.p0[1] if abs(d[1]) < 0.5 else leg.p0[0]),
            "heading": [float(d[0]), float(d[1])],
            "t_start": float(leg_start[i]),
        })
    return {
        "passes": passes,
        "treeline_pass_id": int(cfg.flight.treeline_pass),
        "crossing_pass_id": int(cfg.flight.n_passes),
        "altitude_agl_m": float(cfg.flight.altitude),
        "speed_mps": float(cfg.flight.speed),
        "spacing_m": float(cfg.flight.spacing),
        "note": ("adjacent passes fly opposite directions; that is what makes the "
                 "sensor mounting angle observable"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--backend", default=None, choices=["open3d", "heightfield"])
    ap.add_argument("--limit", type=int, default=None,
                    help="record only the first N sweeps (smoke tests)")
    args = ap.parse_args(argv)

    cfg = load(args.config)
    if args.name:
        cfg = cfg.override(**{"run.name": args.name})
    if args.backend:
        cfg = cfg.override(**{"run.backend": args.backend})
    record(cfg, out_root=args.out, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
