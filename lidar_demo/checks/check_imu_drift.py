"""Measure dead-reckoned drift and the GNSS outage, and tune against them.

Run this before recording.  It answers two questions that decide whether the
blue map will be worth looking at: how far the inertial solution wanders when
left alone, and how long it is left alone for.
"""

from __future__ import annotations

import argparse

import numpy as np

from ..config import load
from ..sim.flight import plan_survey, truth_on_clock
from ..sim.gnss import dropout_summary, simulate_gnss
from ..sim.imu import drift_report, predicted_drift, simulate_imu


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--window", type=float, default=None,
                    help="drift window in seconds (default: imu.drift_window_s)")
    args = ap.parse_args(argv)

    cfg = load(args.config)
    path = plan_survey(cfg.flight, cfg.scene)
    truth = truth_on_clock(path, cfg.imu.rate_hz)

    seeds = np.random.SeedSequence(cfg.run.seed).spawn(4)
    accel, gyro, b_a, b_g = simulate_imu(cfg.imu, truth,
                                         np.random.default_rng(seeds[1]))
    t_g, idx_g, p_g, valid, sigma = simulate_gnss(
        cfg.gnss, truth, cfg.scene.treeline_y, np.random.default_rng(seeds[2]))

    print(f"flight {path.duration:.1f} s, {len(truth)} IMU samples at "
          f"{cfg.imu.rate_hz:.0f} Hz")

    win = args.window if args.window is not None else cfg.imu.drift_window_s
    rep = drift_report(cfg.imu, truth, accel, gyro, b_a, b_g, window_s=win)
    print(rep)

    print("\nclosed-form error budget per axis:")
    for k, v in predicted_drift(cfg.imu, win).items():
        print(f"  {k:16s} {v:7.3f} m")
    full = predicted_drift(cfg.imu, path.duration)
    total = np.sqrt(sum(v ** 2 for v in full.values()))
    print(f"\nover the whole {path.duration:.0f} s flight the same budget gives "
          f"{total:.0f} m per axis, which is why the drift target is read over "
          f"the {win:.0f} s GNSS-denied window and the blue trajectory keeps a "
          f"loose GNSS pull.")

    ds = dropout_summary(t_g, valid)
    print(f"\nGNSS: {valid.sum()}/{valid.size} fixes valid, longest outage "
          f"{ds['longest_s']:.0f} s, total {ds['total_s']:.0f} s "
          f"({ds['fraction']:.0%} of the flight)")
    print(f"      sigma {sigma[0]:.2f} / {sigma[1]:.2f} / {sigma[2]:.2f} m "
          f"(H/H/V; vertical is deliberately the worse axis)")

    ok_xy = 0.8 <= rep.xy_median <= 2.5
    ok_z = 0.3 <= rep.z_median <= 1.3
    print(f"\ngate 2 precursor: XY {'ok' if ok_xy else 'OUT OF BAND'}, "
          f"Z {'ok' if ok_z else 'OUT OF BAND'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
