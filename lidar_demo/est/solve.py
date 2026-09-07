"""The offline pipeline: one recorded run in, two reconstructions out.

    blue      the aircraft's own belief, filtered forward, nominal mount
    green     the graph's answer, trajectory and mount solved together

Both are written as dense trajectories on the IMU clock, each carrying the
extrinsic it was built with, so the map stage can project the same stored
returns through either one.

The solve runs in rounds.  Round one registers from the blue trajectory with the
nominal mount, which is the best guess available; the submaps it builds are
distorted by that wrong mount, so its measurements are biased and its answer is
approximate.  Round two rebuilds the submaps from the solved trajectory and the
recovered mount and registers again, and by then both are close enough that the
distortion has largely gone.  Rounds stop when the mounting angle stops moving.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..config import DemoConfig, load, load_run_config
from ..io import DenseTraj, Extrinsic, RunReader, save_traj
from ..frames import angle_between_deg
from .blue import blue_trajectory, trajectory_error
from .keyframes import (assign_pass_ids, build_submaps, keyframe_arrays,
                        select_keyframes)
from .overlap import (look_ahead_distance, overlap_pairs, sequential_pairs,
                      summarise)
from .register import available_backend, init_from_poses, register_pair
from .strapdown import imu_bridge


class Log:
    """Prints and keeps, so the run log and the console say the same thing."""

    def __init__(self, path: Path | None = None):
        self.lines: list[str] = []
        self.path = path

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.lines.append(msg)

    def save(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("\n".join(self.lines) + "\n")


def register_all(kfs, pairs, E, cfg, kind, threads=8):
    """Register a list of pairs, returning only the ones that survived."""
    out = {}
    for i, j in pairs:
        a, b = kfs[i], kfs[j]
        if a.n_points < 500 or b.n_points < 500:
            continue
        T_init = init_from_poses(a.R, a.p, b.R, b.p, E)
        res = register_pair(a.cloud, b.cloud, T_init, cfg, kind, threads)
        if res.ok:
            out[(i, j)] = res
    return out


def run(run_dir: Path, cfg: DemoConfig, rounds: int | None = None,
        threads: int = 8, verbose: bool = False) -> dict:
    import gtsam  # noqa: F401  (fail early and clearly if it is missing)

    from .graph import (build_graph, extract, extrinsic_marginal,
                        extrinsic_report, format_report, solve, to_pose3)

    reader = RunReader(run_dir)
    run_cfg = load_run_config(run_dir)
    est_dir = run_dir / "est"
    est_dir.mkdir(parents=True, exist_ok=True)
    log = Log(est_dir / "solve_log.txt")

    rounds = int(cfg.graph.rounds if rounds is None else rounds)
    imu_cfg = run_cfg.imu               # the sensor's own numbers, from the run
    E_nom = reader.extrinsic_nominal

    log(f"run {run_dir}  ({reader.n_sweeps} sweeps, registration backend "
        f"{available_backend()})")
    log(f"nominal mount rpy {np.round(E_nom.rpy_deg, 3)} deg  -- the only mount "
        f"the pipeline is told about")

    # ---------------- blue ----------------
    t0 = time.time()
    t, accel, gyro = reader.imu()
    gnss = reader.gnss()
    t_gnss, p_gnss, valid, sigma_gnss, gnss_idx = gnss
    blue = blue_trajectory(t, accel, gyro, t_gnss, p_gnss, valid, gnss_idx,
                           cfg.blue, imu_cfg, sigma_gnss, E_nom)
    log(f"\n{blue.summary()}  [{time.time()-t0:.1f} s]")
    save_traj(est_dir / "blue_traj.npz", blue.traj)

    # ---------------- keyframes ----------------
    sweep_t = reader.sweep_start_times()
    kfs = select_keyframes(blue.traj, sweep_t, cfg.graph.keyframe_trans,
                           cfg.graph.keyframe_rot_deg)
    assign_pass_ids(kfs)
    n_kf = len(kfs)
    stride = max(int(round(2.0 * cfg.graph.submap_radius / cfg.graph.keyframe_trans)), 1)
    look_ahead = look_ahead_distance(cfg.flight.altitude, 35.0,
                                     cfg.lidar.max_scan_angle_deg)
    log(f"{n_kf} keyframes over {len(sweep_t)} sweeps, "
        f"{len(set(k.pass_id for k in kfs))} passes; "
        f"submap radius {cfg.graph.submap_radius:.1f} m -> along-track stride "
        f"{stride} keyframes; footprint sits {look_ahead:.1f} m ahead")

    traj = blue.traj
    E_cur = E_nom
    history = []
    result = build = None

    for rnd in range(1, rounds + 1):
        log(f"\n--- round {rnd} ---")
        t0 = time.time()
        build_submaps(kfs, reader, traj, E_cur, cfg.graph.voxel,
                      cfg.graph.submap_radius, cfg.graph.submap_max_points)
        pts = np.array([kf.n_points for kf in kfs])
        log(f"submaps built in {time.time()-t0:.0f} s, median {int(np.median(pts))} "
            f"points")

        t0 = time.time()
        seq_pairs = sequential_pairs(kfs, stride)
        seq = register_all(kfs, seq_pairs, E_cur, cfg.graph, "seq", threads)
        log(f"along-track: {len(seq)}/{len(seq_pairs)} pairs registered "
            f"[{time.time()-t0:.0f} s]")

        # Stage A first, with the along-track factors only.  Its job is to fix
        # the trajectory before any inter-pass pair is attempted, and that
        # ordering matters more than it looks: an inter-pass pair spans tens of
        # metres of flying, so a seed that is metres out over the GNSS-denied
        # stretch starts the matcher far from the answer, it slides, and the
        # resulting bias is indistinguishable from a mounting-angle error.
        t0 = time.time()
        seeds = _seeds(result, kfs, n_kf) if result is not None else (None, None, None)
        stage_a = build_graph(kfs, t, accel, gyro, gnss, seq, {}, E_cur,
                              cfg.graph, imu_cfg, *seeds, E_prior=E_nom)
        res_a, stats_a = solve(stage_a, cfg.graph, cfg.graph.lm_stage_a_iters, verbose)
        log(f"stage A ({stage_a.n_imu} imu, {stage_a.n_gnss} gnss, "
            f"{stage_a.n_seq} along-track): error {stats_a['error_before']:.3g} -> "
            f"{stats_a['error_after']:.3g} in {stats_a['iterations']} its "
            f"[{time.time()-t0:.0f} s]")

        Ra, pa, va, _, _ = extract(res_a, n_kf)
        for kf, R, p_, v in zip(kfs, Ra, pa, va):
            kf.R, kf.p, kf.v = R, p_, v

        t0 = time.time()
        cross_pairs = overlap_pairs(kfs, cfg.graph, look_ahead)
        info = summarise(kfs, cross_pairs)
        cross = register_all(kfs, cross_pairs, E_cur, cfg.graph, "cross", threads)
        n_cross_ok = sum(1 for (i, j) in cross
                         if kfs[i].pass_id != kfs[j].pass_id)
        log(f"overlapping: {len(cross)}/{info['n']} pairs registered "
            f"({info['cross_pass']} candidates were on different passes, "
            f"{n_cross_ok} survived) [{time.time()-t0:.0f} s]")

        if not cross:
            log("no inter-pass constraints survived; the mounting angle cannot be "
                "recovered from this round")

        t0 = time.time()
        build = build_graph(kfs, t, accel, gyro, gnss, seq, cross, E_cur,
                            cfg.graph, imu_cfg, *_seeds(res_a, kfs, n_kf),
                            E_prior=E_nom)
        result, stats = solve(build, cfg.graph, cfg.graph.lm_max_iters, verbose)
        log(f"stage B ({build.n_imu} imu, {build.n_gnss} gnss, {build.n_seq} "
            f"along-track, {build.n_cross} overlapping): error "
            f"{stats['error_before']:.3g} -> {stats['error_after']:.3g} in "
            f"{stats['iterations']} its  [{time.time()-t0:.0f} s]")

        Rs, ps, vs, bs, E_new = extract(result, n_kf)
        moved = angle_between_deg(E_cur.R, E_new.R)
        log(f"mount moved {moved:.3f} deg this round; now "
            f"{np.round(E_new.rpy_deg, 3)}")
        history.append({"round": rnd, "moved_deg": float(moved),
                        "rpy_deg": [float(v) for v in E_new.rpy_deg],
                        **stats})

        kf_idx = np.array([kf.imu_idx for kf in kfs])
        traj = imu_bridge(t, accel, gyro, kf_idx, Rs, ps, vs, bs,
                          label="green", extrinsic=E_new)
        for kf, R, p, v in zip(kfs, Rs, ps, vs):
            kf.R, kf.p, kf.v = R, p, v
        E_cur = E_new

        if moved < cfg.graph.converge_E_deg and rnd < rounds:
            log(f"mount settled to under {cfg.graph.converge_E_deg} deg; stopping "
                f"after round {rnd}")
            break

    # ---------------- report ----------------
    sigma = extrinsic_marginal(build, result)
    E_true = None
    try:
        E_true = reader.truth_extrinsic()
    except FileNotFoundError:
        pass
    rep = extrinsic_report(E_cur, E_nom, sigma, E_true)
    rep["rounds"] = history
    log("")
    log(format_report(rep))

    save_traj(est_dir / "green_traj.npz", traj)
    np.savez(est_dir / "keyframes.npz", **keyframe_arrays(kfs),
             seq_pairs=np.array(sorted(seq), dtype=np.int64).reshape(-1, 2),
             cross_pairs=np.array(sorted(cross), dtype=np.int64).reshape(-1, 2))
    (est_dir / "boresight.json").write_text(json.dumps(rep, indent=2))
    (est_dir / "boresight.txt").write_text(format_report(rep) + "\n")

    # Trajectory accuracy, for the log only: the pipeline does not read truth.
    try:
        truth = reader.truth_trajectory()
        for name, tr in (("blue", blue.traj), ("green", traj)):
            e = trajectory_error(tr, truth.p, truth.R)
            log(f"\n{name:>5} vs truth: xy rms {e['xy_rms_m']:.2f} m "
                f"(p95 {e['xy_p95_m']:.2f}, max {e['xy_max_m']:.2f}), "
                f"z rms {e['z_rms_m']:.2f} m, attitude rms "
                f"{e['att_rms_deg']:.3f} deg")
            rep[f"{name}_vs_truth"] = e
    except FileNotFoundError:
        pass

    (est_dir / "boresight.json").write_text(json.dumps(rep, indent=2))
    log.save()
    return rep


def _seeds(result, kfs, n):
    """Warm start from a previous solve."""
    from gtsam.symbol_shorthand import B, V, X

    poses = [result.atPose3(X(i)) for i in range(n)]
    vels = [np.asarray(result.atVector(V(i)), dtype=float) for i in range(n)]
    biases = [result.atConstantBias(B(i)) for i in range(n)]
    return poses, vels, biases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    cfg = load(args.config)
    rep = run(Path(args.run), cfg, args.rounds, args.threads, args.verbose)
    return 0 if rep.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
