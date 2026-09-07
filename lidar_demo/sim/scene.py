"""The farm: terrain, canopy and props, assembled into one triangle soup.

The art direction and the mathematics want the same objects.  A windmill, a
scarecrow, a farmhouse, fence posts and a treeline are all tall isolated
verticals: they break the horizon in the beauty pass, and they are the only
features a scan matcher can lock onto over a wheat canopy that is otherwise
nearly featureless.  :func:`vertical_coverage` refuses to build a scene where a
close-up could land on bare crop with nothing to register against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import SceneCfg
from ..io import write_ply_mesh
from . import primitives as prim
from .terrain import (CLASS_GROUND, Heightfield, build_heightfield,
                      heightfield_to_mesh, sample_bilinear)

CLASS_TREE_TRUNK = 4
CLASS_TREE_CROWN = 5
CLASS_BUILDING = 6
CLASS_ROOF = 7
CLASS_WINDMILL = 8
CLASS_POST = 9
CLASS_SCARECROW = 10

# Anything at least this tall counts as a registration feature.
VERTICAL_MIN_HEIGHT = 1.5

ALBEDO = {
    CLASS_GROUND: 0.25,
    2: 0.45,                 # healthy canopy
    3: 0.35,                 # stressed canopy
    CLASS_TREE_TRUNK: 0.30,
    CLASS_TREE_CROWN: 0.40,
    CLASS_BUILDING: 0.60,
    CLASS_ROOF: 0.50,
    CLASS_WINDMILL: 0.70,
    CLASS_POST: 0.35,
    CLASS_SCARECROW: 0.40,
}


@dataclass
class MeshPart:
    name: str
    class_id: int
    V: np.ndarray
    F: np.ndarray


@dataclass
class PropSpec:
    """A prop as the render and the gate checks see it."""

    name: str
    kind: str
    xy: tuple[float, float]
    height: float


@dataclass
class SceneMesh:
    hf: Heightfield
    parts: list[MeshPart] = field(default_factory=list)
    props: list[PropSpec] = field(default_factory=list)

    def merged(self):
        """All parts as one ``(V, F, face_class)``."""
        Vs, Fs, Cs, off = [], [], [], 0
        for p in self.parts:
            Vs.append(p.V)
            Fs.append(p.F + off)
            Cs.append(np.full(p.F.shape[0], p.class_id, dtype=np.uint8))
            off += p.V.shape[0]
        return (np.concatenate(Vs, axis=0),
                np.concatenate(Fs, axis=0),
                np.concatenate(Cs, axis=0))

    @property
    def n_triangles(self) -> int:
        return int(sum(p.F.shape[0] for p in self.parts))


# ---------------------------------------------------------------------------
# props
# ---------------------------------------------------------------------------


def _place(V, hf: Heightfield, xy) -> np.ndarray:
    """Drop a prim so its base sits on the terrain at ``xy``."""
    z = float(sample_bilinear(hf, np.array([xy]), "terrain")[0])
    return prim.transform(V, t=np.array([xy[0], xy[1], z]))


def _treeline(cfg: SceneCfg, hf: Heightfield, rng):
    """Windbreak along one field edge.

    This is also the GNSS multipath source, so the picture and the physical
    model agree about where it is: one place, one number in the config.
    """
    trunks, crowns, specs = [], [], []
    x = cfg.treeline_x[0]
    while x < cfg.treeline_x[1]:
        h = rng.uniform(*cfg.tree_height)
        y = cfg.treeline_y + rng.uniform(-1.0, 1.0)
        trunk_h = 3.0
        Vt, Ft = prim.cylinder(0.25, trunk_h, n=10)
        trunks.append((_place(Vt, hf, (x, y)), Ft))
        cz = float(sample_bilinear(hf, np.array([[x, y]]), "terrain")[0])
        semi_c = 0.5 * max(h - trunk_h, 1.0)
        Vc, Fc = prim.ellipsoid(2.5, 2.5, semi_c, n_u=12, n_v=6)
        crowns.append((prim.transform(Vc, t=np.array([x, y, cz + trunk_h + semi_c])), Fc))
        specs.append(PropSpec(f"tree_{len(specs)}", "tree", (x, y), h))
        x += rng.uniform(*cfg.tree_pitch)
    return trunks, crowns, specs


def _windmill(cfg: SceneCfg, hf: Heightfield):
    x, y = cfg.windmill_xy
    h = cfg.windmill_height
    Vt, Ft = prim.frustum(1.2, 0.5, h, n=12)
    parts = [(_place(Vt, hf, (x, y)), Ft)]
    z = float(sample_bilinear(hf, np.array([[x, y]]), "terrain")[0])
    for k in range(4):
        a = k * np.pi / 2.0 + 0.3
        # Blades stand in the vertical plane facing +x, radiating from the hub.
        Vb, Fb = prim.box(5.0, 0.3, 0.05)
        Vb = Vb - np.array([0.0, 0.0, 0.025])
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.cos(a), -np.sin(a)],
                      [0.0, np.sin(a), np.cos(a)]])
        Vb = prim.transform(Vb + np.array([2.5, 0.0, 0.0]), R=R)
        parts.append((prim.transform(Vb, t=np.array([x + 0.7, y, z + h])), Fb))
    return parts, PropSpec("windmill", "windmill", (x, y), h)


def _building(cfg: SceneCfg, hf: Heightfield, xy, sx, sy, wall, ridge, name):
    Vw, Fw = prim.box(sx, sy, wall)
    walls = (_place(Vw, hf, xy), Fw)
    Vr, Fr = prim.gable(sx * 1.08, sy * 1.08, wall, ridge)
    roof = (_place(Vr, hf, xy), Fr)
    return walls, roof, PropSpec(name, "building", tuple(xy), ridge)


def _scarecrow(cfg: SceneCfg, hf: Heightfield):
    x, y = cfg.scarecrow_xy
    Vp, Fp = prim.cylinder(0.08, 2.2, n=8)
    parts = [(_place(Vp, hf, (x, y)), Fp)]
    z = float(sample_bilinear(hf, np.array([[x, y]]), "terrain")[0])
    Vb, Fb = prim.box(0.12, 1.4, 0.1)
    parts.append((prim.transform(Vb, t=np.array([x, y, z + 1.6])), Fb))
    Vh, Fh = prim.sphere(0.16)
    parts.append((prim.transform(Vh, t=np.array([x, y, z + 2.3])), Fh))
    return parts, PropSpec("scarecrow", "scarecrow", (x, y), 2.4)


def _posts(cfg: SceneCfg, hf: Heightfield, rng):
    parts, specs = [], []
    for i in range(cfg.n_post_x):
        for y0 in cfg.post_y:
            x = cfg.post_x0 + cfg.post_dx * i + rng.uniform(-cfg.post_jitter, cfg.post_jitter)
            y = y0 + rng.uniform(-cfg.post_jitter, cfg.post_jitter)
            Vp, Fp = prim.cylinder(0.08, cfg.post_height, n=8)
            parts.append((_place(Vp, hf, (x, y)), Fp))
            specs.append(PropSpec(f"post_{len(specs)}", "post", (x, y), cfg.post_height))
    return parts, specs


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def build_scene(cfg: SceneCfg, rng: np.random.Generator) -> SceneMesh:
    hf = build_heightfield(cfg, rng)
    V, F, face_class = heightfield_to_mesh(hf, "surface")

    parts: list[MeshPart] = []
    # The ground mesh carries three classes; split it so every triangle has one.
    for cid in np.unique(face_class):
        sel = face_class == cid
        parts.append(MeshPart(f"ground_{int(cid)}", int(cid), V, F[sel]))

    props: list[PropSpec] = []

    trunks, crowns, tree_specs = _treeline(cfg, hf, rng)
    Vt, Ft = prim.concat(trunks)
    parts.append(MeshPart("tree_trunks", CLASS_TREE_TRUNK, Vt, Ft))
    Vc, Fc = prim.concat(crowns)
    parts.append(MeshPart("tree_crowns", CLASS_TREE_CROWN, Vc, Fc))
    props += tree_specs

    wm_parts, wm_spec = _windmill(cfg, hf)
    Vw, Fw = prim.concat(wm_parts)
    parts.append(MeshPart("windmill", CLASS_WINDMILL, Vw, Fw))
    props.append(wm_spec)

    walls, roof, spec = _building(cfg, hf, cfg.farmhouse_xy, 10.0, 7.0, 4.0, 6.5, "farmhouse")
    parts.append(MeshPart("farmhouse_walls", CLASS_BUILDING, *walls))
    parts.append(MeshPart("farmhouse_roof", CLASS_ROOF, *roof))
    props.append(spec)

    walls, roof, spec = _building(cfg, hf, cfg.outbuilding_xy, 6.0, 4.0, 3.0, 4.2, "outbuilding")
    parts.append(MeshPart("outbuilding_walls", CLASS_BUILDING, *walls))
    parts.append(MeshPart("outbuilding_roof", CLASS_ROOF, *roof))
    props.append(spec)

    sc_parts, sc_spec = _scarecrow(cfg, hf)
    Vs, Fs = prim.concat(sc_parts)
    parts.append(MeshPart("scarecrow", CLASS_SCARECROW, Vs, Fs))
    props.append(sc_spec)

    post_parts, post_specs = _posts(cfg, hf, rng)
    Vp, Fp = prim.concat(post_parts)
    parts.append(MeshPart("posts", CLASS_POST, Vp, Fp))
    props += post_specs

    scene = SceneMesh(hf=hf, parts=parts, props=props)

    n = vertical_coverage(scene, cfg)
    if n < cfg.min_verticals:
        raise ValueError(
            f"scene has a {cfg.coverage_window:.0f} m window containing only {n} "
            f"vertical feature(s); the spec requires at least {cfg.min_verticals} in "
            "any close-up framing, otherwise registration has nothing to lock onto")
    return scene


def vertical_coverage(scene: SceneMesh, cfg: SceneCfg, step: float = 5.0) -> int:
    """Fewest tall verticals visible from any window inside the survey.

    A close-up frames roughly ``coverage_window`` metres.  Trees sit outside the
    survey but are visible from inside it, so the count is taken over a window
    grown by half its own width rather than clipped to the survey rectangle.
    """
    w = cfg.coverage_window
    xy = np.array([p.xy for p in scene.props if p.height >= VERTICAL_MIN_HEIGHT])
    if xy.size == 0:
        return 0
    worst = 10 ** 9
    for cx in np.arange(0.0, cfg.survey_x + 1e-6, step):
        for cy in np.arange(0.0, cfg.survey_y + 1e-6, step):
            inside = ((np.abs(xy[:, 0] - cx) <= w) & (np.abs(xy[:, 1] - cy) <= w))
            worst = min(worst, int(inside.sum()))
    return worst


def export_scene(scene: SceneMesh, out_dir: str | Path) -> dict[str, Path]:
    """Write the meshes, the heightfield and the prop list."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    Vt, Ft, _ = heightfield_to_mesh(scene.hf, "terrain")
    paths = {"terrain": write_ply_mesh(out / "terrain.ply", Vt, Ft)}

    V, F, _ = scene.merged()
    paths["surface"] = write_ply_mesh(out / "surface.ply", V, F)

    prop_parts = [(p.V, p.F) for p in scene.parts if not p.name.startswith("ground")]
    Vp, Fp = prim.concat(prop_parts)
    paths["props"] = write_ply_mesh(out / "props.ply", Vp, Fp)

    hf = scene.hf
    np.savez(out / "terrain_heightfield.npz",
             x0=np.float64(hf.x0), y0=np.float64(hf.y0), dx=np.float64(hf.dx),
             z_terrain=hf.z_terrain, z_surface=hf.z_surface,
             canopy_h=hf.canopy_h, class_id=hf.class_id,
             ditch_xy=np.array([hf.ditch_p0, hf.ditch_p1], dtype=np.float64))
    paths["heightfield"] = out / "terrain_heightfield.npz"

    (out / "props.json").write_text(json.dumps(
        [{"name": p.name, "kind": p.kind, "xy": [float(p.xy[0]), float(p.xy[1])],
          "height": float(p.height)} for p in scene.props], indent=2))
    paths["props_json"] = out / "props.json"
    return paths
