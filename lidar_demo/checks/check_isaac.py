"""Is the Isaac Sim install actually usable, and for what?

Run this straight after installing.  It answers, in order: does the package
import, does the simulation app start headless, is the RTX renderer available on
this GPU, and does a LiDAR sensor exist to attach to a prim.  Each answer is
printed separately, because they fail for different reasons and a single
pass/fail would hide which one went wrong.

Nothing here touches the demo.  It exists so that the first time an Isaac
backend misbehaves, it is already known whether the problem is the install.

**Kit is asked for D3D12 rather than Vulkan.**  On this machine -- an RTX 3080 Ti
on driver 610.74 -- the Vulkan backend segfaults inside ``rtx.scenedb`` while the
material library compiles its base MDL shaders, which happens on the first frame
and so takes every run with it, whatever experience file is used.  The same
build on D3D12 starts in about ten seconds and NVIDIA's own compatibility
checker, which runs on D3D12, reports the driver and GPU as supported.  Set
``LIDAR_DEMO_ISAAC_API=vulkan`` to override.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


# Filled in by main() so the report is written before the app tears the process
# down; see the note in check().
args_json_path = [None]


def _line(name: str, ok: bool, detail: str = "") -> str:
    return f"  {'ok  ' if ok else 'FAIL'}  {name:<28} {detail}"


GRAPHICS_API_ARG = {"d3d12": "--/app/vulkan=false",
                    "vulkan": "--/app/vulkan=true"}


def select_graphics_api(api: str | None = None) -> str:
    """Put the graphics-API choice on ``sys.argv``, where Kit reads it."""
    import os

    api = (api or os.environ.get("LIDAR_DEMO_ISAAC_API")
           or ("d3d12" if platform.system() == "Windows" else "vulkan")).lower()
    arg = GRAPHICS_API_ARG.get(api)
    if arg and arg not in sys.argv:
        sys.argv.append(arg)
    return api


def check(headless: bool = True, verbose: bool = True,
          api: str | None = None) -> dict:
    out: dict = {"python": sys.version.split()[0],
                 "executable": sys.executable,
                 "platform": platform.platform()}
    log = print if verbose else (lambda *a, **k: None)

    out["graphics_api"] = select_graphics_api(api)
    log(f"python {out['python']} at {sys.executable}")
    log(f"graphics api: {out['graphics_api']}")

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

        app = SimulationApp({"headless": bool(headless),
                             "renderer": "RayTracedLighting"})
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
        out["usable"] = bool(out.get("app") and out.get("stage")
                             and any(out.get("sensor_modules", {}).values()))
        log("")
        log(f"usable for the demo: {'yes' if out['usable'] else 'not yet'}")
    finally:
        # Write the report before closing: SimulationApp.close() ends the
        # process, so anything printed after it never appears.
        if args_json_path[0]:
            Path(args_json_path[0]).write_text(json.dumps(out, indent=2))
            log(f"wrote {args_json_path[0]}")
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gui", action="store_true",
                    help="start with a window instead of headless")
    ap.add_argument("--json", default=None, help="also write the result here")
    ap.add_argument("--api", default=None, choices=["d3d12", "vulkan"],
                    help="graphics backend; D3D12 by default on Windows")
    args, _ = ap.parse_known_args(argv)

    args_json_path[0] = args.json
    rep = check(headless=not args.gui, api=args.api)
    return 0 if rep.get("usable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
