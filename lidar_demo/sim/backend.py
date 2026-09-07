"""Ray casting, behind one interface.

Everything above :meth:`RayCastBackend.cast` -- the beam table, the rolling
shutter, the noise model, the export -- is backend agnostic.  That is the seam
along which an Isaac Lab RTX LiDAR is swapped in for the offline caster without
touching the data contract or the reconstruction.

Three implementations:

``Open3DBackend``
    Embree through Open3D's ``RaycastingScene``.  Casts the same triangle mesh
    that gets exported, so the gate-1 check compares like with like: there is no
    second geometry representation that could quietly disagree with the first.

``HeightfieldBackend``
    Pure numpy ray marching against the terrain surface only, no props.  Slow
    and incomplete on purpose: it exists so the geometry can be tested where
    Open3D is not installed, and as an independent cross-check that the Embree
    path is casting what we think it is.

``IsaacLabBackend``
    In :mod:`lidar_demo.sim.isaac.backend`.  Isaac Lab's warp kernels, tracing
    against a USD stage.  Imported only on request, because it needs Kit
    running.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .scene import SceneMesh
from .terrain import sample_bilinear


@dataclass
class RayHits:
    """Result of casting a batch of rays.  ``t_hit`` is ``inf`` where nothing was hit."""

    t_hit: np.ndarray        # (N,) float32, distance along the (unit) direction
    normal: np.ndarray       # (N, 3) float32, world frame, unit
    class_id: np.ndarray     # (N,) uint8
    prim_id: np.ndarray      # (N,) int64

    @property
    def hit(self) -> np.ndarray:
        return np.isfinite(self.t_hit)


class RayCastBackend(Protocol):
    """Two layers, not one.

    ``"surface"`` is everything a beam can strike: bare earth, canopy top and
    props.  ``"ground"`` is the same scene with the crop removed.  Both are
    needed because a LiDAR pulse does not stop at the canopy -- some fraction of
    every shot filters through the crop and returns from the soil, which is
    exactly how a bare-earth model gets made in the first place.  Casting both
    lets the sensor model decide, shot by shot, which return came back.
    """

    name: str

    def build(self, scene: SceneMesh) -> None:
        """Prepare the acceleration structures for one scene."""

    def cast(self, origins_W: np.ndarray, dirs_W: np.ndarray,
             layer: str = "surface") -> RayHits:
        """Cast ``(N, 3)`` origins along ``(N, 3)`` *unit* directions, world frame."""


# ---------------------------------------------------------------------------


class Open3DBackend:
    """Embree via ``open3d.t.geometry.RaycastingScene``."""

    name = "open3d"

    def __init__(self) -> None:
        self._layers: dict[str, object] = {}
        self._geom_class: dict[str, dict[int, int]] = {}

    def build(self, scene: SceneMesh) -> None:
        import open3d as o3d
        import open3d.core as o3c

        from .terrain import heightfield_to_mesh

        def add(parts):
            sc = o3d.t.geometry.RaycastingScene()
            cmap = {}
            for cid, V, F in parts:
                if F.shape[0] == 0:
                    continue
                gid = sc.add_triangles(
                    o3c.Tensor(np.ascontiguousarray(V, np.float32)),
                    o3c.Tensor(np.ascontiguousarray(F, np.uint32)))
                cmap[int(gid)] = int(cid)
            return sc, cmap

        surface = [(p.class_id, p.V, p.F) for p in scene.parts]
        Vg, Fg, _ = heightfield_to_mesh(scene.hf, "terrain")
        ground = [(1, Vg, Fg)]
        ground += [(p.class_id, p.V, p.F) for p in scene.parts
                   if not p.name.startswith("ground")]

        for layer, parts in (("surface", surface), ("ground", ground)):
            sc, cmap = add(parts)
            self._layers[layer] = sc
            self._geom_class[layer] = cmap

    def cast(self, origins_W: np.ndarray, dirs_W: np.ndarray,
             layer: str = "surface") -> RayHits:
        import open3d.core as o3c

        if layer not in self._layers:
            raise RuntimeError("build() the backend before casting")
        rays = np.concatenate([np.asarray(origins_W, np.float32),
                               np.asarray(dirs_W, np.float32)], axis=1)
        ans = self._layers[layer].cast_rays(o3c.Tensor(np.ascontiguousarray(rays)))
        t_hit = ans["t_hit"].numpy()
        gids = ans["geometry_ids"].numpy()
        normals = ans["primitive_normals"].numpy()
        prim = ans["primitive_ids"].numpy().astype(np.int64)

        class_id = np.zeros(t_hit.shape[0], dtype=np.uint8)
        for gid, cid in self._geom_class[layer].items():
            class_id[gids == gid] = cid
        return RayHits(t_hit=t_hit.astype(np.float32),
                       normal=normals.astype(np.float32),
                       class_id=class_id, prim_id=prim)

    def distance_to_mesh(self, points_W: np.ndarray) -> np.ndarray:
        """Unsigned distance from points to the scene; used by the gate-1 check."""
        import open3d.core as o3c

        return self._layers["surface"].compute_distance(
            o3c.Tensor(np.ascontiguousarray(points_W, np.float32))).numpy()


# ---------------------------------------------------------------------------


class HeightfieldBackend:
    """Ray march against the canopy-top surface.  Terrain only, no props.

    Marching is coarse-then-bisect: step along the ray until the sign of
    ``z_ray - z_surface`` flips, then bisect that bracket.  Exact for a
    single-valued surface, which a heightfield is by construction.
    """

    name = "heightfield"

    def __init__(self, step: float = 0.5, max_range: float = 120.0,
                 refine: int = 24) -> None:
        self.step = step
        self.max_range = max_range
        self.refine = refine
        self._hf = None

    def build(self, scene: SceneMesh) -> None:
        self._hf = scene.hf

    def _below(self, o: np.ndarray, d: np.ndarray, t: np.ndarray,
               which: str) -> np.ndarray:
        p = o + d * t[:, None]
        return p[:, 2] - sample_bilinear(self._hf, p[:, :2], which)

    def cast(self, origins_W: np.ndarray, dirs_W: np.ndarray,
             layer: str = "surface") -> RayHits:
        if self._hf is None:
            raise RuntimeError("build() the backend before casting")
        o = np.asarray(origins_W, dtype=float)
        d = np.asarray(dirs_W, dtype=float)
        n = o.shape[0]

        which = "terrain" if layer == "ground" else "surface"
        t_lo = np.full(n, 1e-3)
        f_lo = self._below(o, d, t_lo, which)
        t_hit = np.full(n, np.inf)
        open_ray = np.ones(n, dtype=bool)

        t = t_lo.copy()
        n_steps = int(self.max_range / self.step)
        for _ in range(n_steps):
            t_next = t + self.step
            f_next = self._below(o, d, t_next, which)
            crossed = open_ray & (f_lo > 0.0) & (f_next <= 0.0)
            if crossed.any():
                a = t[crossed].copy()
                b = t_next[crossed].copy()
                oc, dc = o[crossed], d[crossed]
                for _ in range(self.refine):
                    m = 0.5 * (a + b)
                    fm = self._below(oc, dc, m, which)
                    hi = fm > 0.0
                    a = np.where(hi, m, a)
                    b = np.where(hi, b, m)
                t_hit[crossed] = 0.5 * (a + b)
                open_ray[crossed] = False
            t = t_next
            f_lo = f_next
            if not open_ray.any():
                break

        # Surface normal from the heightfield gradient at the hit.
        normal = np.zeros((n, 3), dtype=np.float32)
        class_id = np.zeros(n, dtype=np.uint8)
        hit = np.isfinite(t_hit)
        if hit.any():
            p = o[hit] + d[hit] * t_hit[hit, None]
            h = self._hf.dx
            zx = ((sample_bilinear(self._hf, p[:, :2] + [h, 0], which)
                   - sample_bilinear(self._hf, p[:, :2] - [h, 0], which)) / (2 * h))
            zy = ((sample_bilinear(self._hf, p[:, :2] + [0, h], which)
                   - sample_bilinear(self._hf, p[:, :2] - [0, h], which)) / (2 * h))
            nn = np.stack([-zx, -zy, np.ones_like(zx)], axis=1)
            nn /= np.linalg.norm(nn, axis=1, keepdims=True)
            normal[hit] = nn.astype(np.float32)

            ny, nx = self._hf.class_id.shape
            i = np.clip(((p[:, 0] - self._hf.x0) / self._hf.dx).astype(int), 0, nx - 1)
            j = np.clip(((p[:, 1] - self._hf.y0) / self._hf.dx).astype(int), 0, ny - 1)
            class_id[hit] = 1 if which == "terrain" else self._hf.class_id[j, i]

        return RayHits(t_hit=t_hit.astype(np.float32), normal=normal,
                       class_id=class_id, prim_id=np.full(n, -1, dtype=np.int64))


# ---------------------------------------------------------------------------


def make_backend(name: str) -> RayCastBackend:
    """Build a backend by name.

    The Isaac one is imported lazily and only on request, because importing it
    reaches for ``omni`` and ``pxr``, which fail unless a ``SimulationApp`` is
    already running.  Everything else has to keep working on a machine with no
    Isaac installed at all.
    """
    if name == "open3d":
        return Open3DBackend()
    if name == "heightfield":
        return HeightfieldBackend()
    if name == "isaaclab":
        from .isaac.backend import IsaacLabBackend

        return IsaacLabBackend()
    raise ValueError(f"unknown ray-cast backend {name!r}")
