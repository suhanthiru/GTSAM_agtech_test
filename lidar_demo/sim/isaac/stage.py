"""The farm as a USD stage.

The same :class:`~lidar_demo.sim.scene.SceneMesh` the numpy ray caster uses, laid
out as USD prims: one mesh per part, grouped under ``/World/Farm``, with the
surface layer and the bare-earth layer as separate subtrees so a sensor can be
pointed at either.

Two things make this worth doing rather than casting against the numpy arrays
directly.  A USD stage is what Isaac Sim, Omniverse and Blender all read, so the
beauty pass and the simulation stop being two descriptions of the same field
that could drift apart.  And once the geometry is in the stage, the Isaac
backend reads it *back out* to build its ray-casting meshes, which means a
mistake in the export shows up as a bad cast rather than hiding.

Coordinates are metres and z-up, matching the rest of the demo, and the stage
records that explicitly so a consumer does not have to assume it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Class ids travel with the geometry as a primvar so a ray hit can be labelled
# without a second lookup table.
CLASS_PRIMVAR = "surfaceClass"

GROUND_PARTS = ("bare_earth",)


def _mesh_prim(stage, path: str, V: np.ndarray, F: np.ndarray, class_id: int):
    from pxr import Sdf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(V, np.float32)))
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.asarray(F, np.int32).ravel()))
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(F.shape[0], 3, np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    lo = np.asarray(V, np.float32).min(axis=0)
    hi = np.asarray(V, np.float32).max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([lo, hi])))

    prim = mesh.GetPrim()
    attr = prim.CreateAttribute(f"primvars:{CLASS_PRIMVAR}", Sdf.ValueTypeNames.Int)
    attr.Set(int(class_id))
    return mesh


def build_stage(scene, path: str | Path | None = None, up_axis: str = "Z"):
    """Create a stage holding the scene.  Returns ``(stage, layout)``.

    ``layout`` maps each layer to the prim paths it contains, in the order they
    were written, which is what the backend uses to rebuild its meshes.
    """
    from pxr import Usd, UsdGeom

    stage = (Usd.Stage.CreateNew(str(path)) if path is not None
             else Usd.Stage.CreateInMemory())
    UsdGeom.SetStageUpAxis(stage, getattr(UsdGeom.Tokens, up_axis.lower()))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Farm")
    UsdGeom.Xform.Define(stage, "/World/Farm/Surface")
    UsdGeom.Xform.Define(stage, "/World/Farm/Ground")

    layout: dict[str, list[str]] = {"surface": [], "ground": []}

    for part in scene.parts:
        if part.F.shape[0] == 0:
            continue
        p = f"/World/Farm/Surface/{_safe(part.name)}"
        _mesh_prim(stage, p, part.V, part.F, part.class_id)
        layout["surface"].append(p)

    # The bare-earth layer: the terrain without the crop, plus the props, which
    # is what a pulse that filters through the canopy actually meets.
    from ..terrain import heightfield_to_mesh

    Vg, Fg, _ = heightfield_to_mesh(scene.hf, "terrain")
    p = "/World/Farm/Ground/bare_earth"
    _mesh_prim(stage, p, Vg, Fg, 1)
    layout["ground"].append(p)
    for part in scene.parts:
        if part.F.shape[0] == 0 or part.name.startswith("ground"):
            continue
        p = f"/World/Farm/Ground/{_safe(part.name)}"
        _mesh_prim(stage, p, part.V, part.F, part.class_id)
        layout["ground"].append(p)

    return stage, layout


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in name)


def read_layer(stage, prim_paths: list[str]):
    """Read a layer back out of the stage as one triangle soup.

    Returns ``(V, F, face_class)``.  Reading back rather than keeping the numpy
    arrays around is deliberate: it means the rays are cast against what was
    actually written, so an export bug cannot hide behind correct source data.
    """
    from pxr import UsdGeom

    Vs, Fs, Cs, offset = [], [], [], 0
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        mesh = UsdGeom.Mesh(prim)
        V = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        if V.size == 0 or idx.size == 0:
            continue
        if not np.all(counts == 3):
            raise ValueError(f"{path} is not triangulated")
        F = idx.reshape(-1, 3)
        cls_attr = prim.GetAttribute(f"primvars:{CLASS_PRIMVAR}")
        cid = int(cls_attr.Get()) if cls_attr and cls_attr.HasValue() else 0
        Vs.append(V)
        Fs.append(F + offset)
        Cs.append(np.full(F.shape[0], cid, dtype=np.uint8))
        offset += V.shape[0]

    if not Vs:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64),
                np.zeros(0, np.uint8))
    return (np.concatenate(Vs).astype(np.float32),
            np.concatenate(Fs).astype(np.int64),
            np.concatenate(Cs))


def add_sun(stage, azimuth_deg: float = 170.0, elevation_deg: float = 22.0,
            intensity: float = 2000.0, path: str = "/World/Sun"):
    """A low warm key light, for the beauty pass rather than for the LiDAR."""
    from pxr import Gf, UsdLux

    light = UsdLux.DistantLight.Define(stage, path)
    light.CreateIntensityAttr(float(intensity))
    light.CreateAngleAttr(0.53)
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.78))

    from pxr import UsdGeom

    xform = UsdGeom.Xformable(light.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddRotateXYZOp().Set(Gf.Vec3f(float(-90.0 + elevation_deg), 0.0,
                                        float(azimuth_deg)))
    return light


def add_drone(stage, path: str = "/World/Drone", size: float = 1.2):
    """A marker prim at the aircraft, driven per frame by the flight path."""
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, path)
    body = UsdGeom.Cube.Define(stage, path + "/body")
    body.CreateSizeAttr(float(size))
    UsdGeom.Xformable(body.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(1.0, 1.0, 0.25))
    xf = UsdGeom.Xformable(xform.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp()
    xf.AddOrientOp()
    return xform


def set_pose(stage, path: str, R: np.ndarray, p: np.ndarray) -> None:
    """Place a prim at a rotation and position, world frame."""
    from pxr import Gf, UsdGeom

    from ...frames import orthonormalize

    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    for name, op in ops.items():
        if "translate" in name:
            op.Set(Gf.Vec3d(*[float(v) for v in p]))
        elif "orient" in name:
            m = Gf.Matrix3d(*[float(v) for v in orthonormalize(R).T.ravel()])
            op.Set(Gf.Quatf(Gf.Rotation(m).GetQuat()))


def export(scene, path: str | Path, sun: bool = True, drone: bool = True):
    """Write the scene to a ``.usd`` file and return its layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage, layout = build_stage(scene, path)
    if sun:
        add_sun(stage)
    if drone:
        add_drone(stage)
    stage.GetRootLayer().Save()
    return stage, layout
