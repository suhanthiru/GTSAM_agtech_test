"""Record a run with Isaac Sim doing the ray casting.

    scripts/isaacpy -m lidar_demo.sim.isaac.record --name run_isaac

Kit has to be started before ``omni`` or ``pxr`` can be imported, so this module
does that first and only then reaches for anything else.  Everything after that
is the ordinary recorder: same scene, same flight, same beam table, same rolling
shutter, same noise, same run directory.  The only thing that changes is who
traces the rays.

``--compare`` additionally casts a sample of sweeps through both backends from
identical poses and reports how far apart they land.  That is the check worth
having: two independent tracers over one geometry should agree to well under the
range noise, and if they do not, the USD export is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _bootstrap(headless: bool = True, api: str | None = None):
    """Start Kit.  Nothing that touches omni or pxr may be imported before this."""
    from ...checks.check_isaac import select_graphics_api

    select_graphics_api(api)
    from isaacsim import SimulationApp

    return SimulationApp({"headless": bool(headless),
                          "renderer": "RayTracedLighting"})


def compare_backends(cfg, scene, isaac, n_sweeps: int = 12, seed: int = 0):
    """Cast the same sweeps through Isaac and through the offline caster."""
    import numpy as np

    from ...frames import rpy_zyx_to_R
    from ..backend import make_backend
    from ..flight import plan_survey
    from ..lidar import LidarModel, simulate_sweep
    from ..record import true_extrinsic

    other = make_backend("open3d")
    other.build(scene)

    model = LidarModel.from_cfg(cfg.lidar)
    path = plan_survey(cfg.flight, cfg.scene)
    E = true_extrinsic(cfg)
    total = int(path.duration / model.period)
    picks = np.linspace(int(0.1 * total), int(0.9 * total), n_sweeps).astype(int)

    rows = []
    for k in picks:
        a = simulate_sweep(model, isaac, path, int(k), E.R, E.t, cfg.lidar,
                           np.random.default_rng(int(k)))
        b = simulate_sweep(model, other, path, int(k), E.R, E.t, cfg.lidar,
                           np.random.default_rng(int(k)))
        # Match on the beam that produced each point, so the comparison is
        # per-ray rather than per-cloud.
        key_a = a.ring.astype(np.int64) * 100000 + a.col.astype(np.int64)
        key_b = b.ring.astype(np.int64) * 100000 + b.col.astype(np.int64)
        common, ia, ib = np.intersect1d(key_a, key_b, return_indices=True)
        if common.size == 0:
            continue
        d = np.abs(a.range[ia] - b.range[ib])
        rows.append({"sweep": int(k),
                     "isaac_points": len(a.xyz), "offline_points": len(b.xyz),
                     "shared_beams": int(common.size),
                     "range_median_cm": float(np.median(d) * 100),
                     "range_p99_cm": float(np.percentile(d, 99) * 100)})

    med = float(np.median([r["range_median_cm"] for r in rows])) if rows else float("nan")
    p99 = float(np.max([r["range_p99_cm"] for r in rows])) if rows else float("nan")
    shared = float(np.mean([r["shared_beams"] / max(r["isaac_points"], 1)
                            for r in rows])) if rows else 0.0
    return {"sweeps": rows, "range_median_cm": med, "range_worst_p99_cm": p99,
            "shared_beam_fraction": shared,
            # Both tracers see the same triangles and both add the same noise
            # from the same seed, so agreement should be at the millimetre level.
            "agree": bool(med < 1.0 and p99 < 20.0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="run_isaac")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="record only the first N sweeps")
    ap.add_argument("--compare", action="store_true",
                    help="also cast some sweeps through the offline backend")
    ap.add_argument("--compare-only", action="store_true",
                    help="run the comparison and skip the recording")
    ap.add_argument("--api", default=None, choices=["d3d12", "vulkan"])
    ap.add_argument("--gui", action="store_true")
    args, _ = ap.parse_known_args(argv)

    t0 = time.time()
    app = _bootstrap(headless=not args.gui, api=args.api)
    ok = False
    try:
        import numpy as np

        from ...config import load
        from ..record import record
        from ..scene import build_scene
        from .backend import IsaacLabBackend

        cfg = load(args.config).override(**{"run.name": args.name,
                                            "run.backend": "isaaclab"})
        root = Path(args.out or cfg.run.out_root) / cfg.run.name
        root.mkdir(parents=True, exist_ok=True)

        seeds = np.random.SeedSequence(cfg.run.seed).spawn(4)
        scene = build_scene(cfg.scene, np.random.default_rng(seeds[0]))
        print(f"scene: {scene.n_triangles/1e6:.2f} M triangles", flush=True)

        backend = IsaacLabBackend(usd_path=str(root / "scene" / "farm.usd"))
        backend.build(scene)
        print(f"stage: {root/'scene'/'farm.usd'}  "
              f"surface {backend.triangles('surface')} tris, "
              f"ground {backend.triangles('ground')} tris, "
              f"device {backend.device}  [{time.time()-t0:.0f} s]", flush=True)

        report = {"stage": str(root / "scene" / "farm.usd"),
                  "device": str(backend.device),
                  "surface_triangles": backend.triangles("surface"),
                  "ground_triangles": backend.triangles("ground")}

        if args.compare or args.compare_only:
            report["comparison"] = compare_backends(cfg, scene, backend)
            c = report["comparison"]
            print(f"\nagainst the offline caster over {len(c['sweeps'])} sweeps: "
                  f"median range difference {c['range_median_cm']:.3f} cm, "
                  f"worst 99th percentile {c['range_worst_p99_cm']:.2f} cm, "
                  f"{c['shared_beam_fraction']:.0%} of beams shared  -> "
                  f"{'AGREE' if c['agree'] else 'DISAGREE'}", flush=True)

        if not args.compare_only:
            record(cfg, out_root=args.out, limit=args.limit, verbose=True,
                   backend=backend)
            report["recorded"] = True

        (root / "isaac_report.json").write_text(json.dumps(report, indent=2))
        print(f"\nwrote {root/'isaac_report.json'}  "
              f"[{time.time()-t0:.0f} s total]", flush=True)
        ok = report.get("comparison", {}).get("agree", True)
    except Exception:
        # SimulationApp.close() ends the process, so a traceback printed after
        # it never reaches the terminal.  Print it here or lose it.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
