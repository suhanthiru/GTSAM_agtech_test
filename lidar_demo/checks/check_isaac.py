"""Is the Isaac Sim install actually usable, and for what?

Run this straight after installing.  It answers, in order: does the package
import, does the simulation app start headless, is the RTX renderer available on
this GPU, and does a LiDAR sensor exist to attach to a prim.  Each answer is
printed separately, because they fail for different reasons and a single
pass/fail would hide which one went wrong.

Nothing here touches the demo.  It exists so that the first time an Isaac
backend misbehaves, it is already known whether the problem is the install.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def _line(name: str, ok: bool, detail: str = "") -> str:
    return f"  {'ok  ' if ok else 'FAIL'}  {name:<28} {detail}"


def check(headless: bool = True, verbose: bool = True) -> dict:
    out: dict = {"python": sys.version.split()[0],
                 "executable": sys.executable,
                 "platform": platform.platform()}
    log = print if verbose else (lambda *a, **k: None)

    log(f"python {out['python']} at {sys.executable}")

    # ---- 1. the package imports ----
    try:
        import isaacsim
        out["isaacsim_version"] = getattr(isaacsim, "__version__", "unknown")
        out["import"] = True
        log(_line("import isaacsim", True, out["isaacsim_version"]))
    except Exception as exc:
        out["import"] = False
        out["import_error"] = str(exc)
        log(_line("import isaacsim", False, str(exc)[:120]))
        return out

    # ---- 2. the app starts ----
    t0 = time.time()
    app = None
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": bool(headless)})
        out["app_start_s"] = round(time.time() - t0, 1)
        out["app"] = True
        log(_line("SimulationApp starts", True,
                  f"headless={headless}, {out['app_start_s']} s"))
    except Exception as exc:
        out["app"] = False
        out["app_error"] = str(exc)
        log(_line("SimulationApp starts", False, str(exc)[:160]))
        return out

    try:
        # ---- 3. a stage can be made and a prim added ----
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            omni.usd.get_context().new_stage()
            stage = omni.usd.get_context().get_stage()
            UsdGeom.Xform.Define(stage, "/World")
            mesh = UsdGeom.Mesh.Define(stage, "/World/probe")
            mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            out["stage"] = bool(stage.GetPrimAtPath("/World/probe"))
            log(_line("USD stage and mesh", out["stage"]))
        except Exception as exc:
            out["stage"] = False
            out["stage_error"] = str(exc)
            log(_line("USD stage and mesh", False, str(exc)[:160]))

        # ---- 4. the renderer, which is what a LiDAR sensor needs ----
        try:
            import carb

            settings = carb.settings.get_settings()
            renderer = settings.get("/renderer/active")
            out["renderer"] = renderer
            log(_line("renderer", renderer is not None, str(renderer)))
        except Exception as exc:
            out["renderer"] = None
            log(_line("renderer", False, str(exc)[:120]))

        # ---- 5. a range sensor to attach ----
        found = {}
        for name, mod in (("isaacsim.sensors.rtx", "isaacsim.sensors.rtx"),
                          ("isaacsim.sensors.physx", "isaacsim.sensors.physx"),
                          ("omni.isaac.range_sensor", "omni.isaac.range_sensor")):
            try:
                __import__(mod)
                found[name] = True
            except Exception:
                found[name] = False
        out["sensor_modules"] = found
        any_sensor = any(found.values())
        log(_line("a LiDAR sensor module", any_sensor,
                  ", ".join(k for k, v in found.items() if v) or "none found"))

        # ---- 6. Isaac Lab, if it is installed ----
        try:
            import isaaclab

            out["isaaclab_version"] = getattr(isaaclab, "__version__", "unknown")
            out["isaaclab"] = True
            log(_line("import isaaclab", True, out["isaaclab_version"]))
        except Exception as exc:
            out["isaaclab"] = False
            out["isaaclab_error"] = str(exc)
            log(_line("import isaaclab", False, str(exc)[:120]))
    finally:
        if app is not None:
            try:
                app.close()
            except Exception:
                pass

    out["usable"] = bool(out.get("app") and out.get("stage")
                         and any(out.get("sensor_modules", {}).values()))
    log("")
    log(f"usable for the demo: {'yes' if out['usable'] else 'not yet'}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gui", action="store_true",
                    help="start with a window instead of headless")
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args(argv)

    rep = check(headless=not args.gui)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2))
        print(f"wrote {args.json}")
    return 0 if rep.get("usable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
