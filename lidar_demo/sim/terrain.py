"""Terrain and canopy as a heightfield.

A flat plane reconstructs to a flat plane and shows nothing, so relief is built
at three separate scales and each one earns its place:

* a constant slope, so the whole field has a direction and a cross-section has
  a trend to sit against;
* metre-scale rolling relief, which is what a boresight error visibly fails to
  reproduce;
* centimetre furrows at row spacing, which give registration something to bite
  on between the props and give the close-up its texture.

The drainage ditch is the only hard edge in the scene.  It reads instantly in a
cross-section and it gives the eye a fixed landmark while the map moves.

Two surfaces come out of here.  ``z_terrain`` is bare earth: it is the truth the
rendered cross-section overlays and the raster is scored against.  ``z_surface``
is bare earth plus canopy: it is what the LiDAR actually sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SceneCfg
from ..io import CLASS_NAMES  # noqa: F401  (documents the class ids used below)

CLASS_GROUND = 1
CLASS_CANOPY_HEALTHY = 2
CLASS_CANOPY_STRESSED = 3


@dataclass
class Heightfield:
    """Regular grid, ``z[j, i]`` at ``x = x0 + i dx``, ``y = y0 + j dx``."""

    x0: float
    y0: float
    dx: float
    z_terrain: np.ndarray     # (ny, nx) bare earth
    z_surface: np.ndarray     # (ny, nx) bare earth + canopy
    canopy_h: np.ndarray      # (ny, nx)
    class_id: np.ndarray      # (ny, nx) uint8
    ditch_p0: tuple[float, float]
    ditch_p1: tuple[float, float]

    @property
    def shape(self) -> tuple[int, int]:
        return self.z_terrain.shape

    @property
    def x(self) -> np.ndarray:
        return self.x0 + self.dx * np.arange(self.z_terrain.shape[1])

    @property
    def y(self) -> np.ndarray:
        return self.y0 + self.dx * np.arange(self.z_terrain.shape[0])

    def sample(self, xy: np.ndarray, which: str = "terrain") -> np.ndarray:
        return sample_bilinear(self, xy, which)


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _segment_distance(X: np.ndarray, Y: np.ndarray,
                      p0: tuple[float, float], p1: tuple[float, float]) -> np.ndarray:
    """Perpendicular distance from each grid point to a line segment."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    L2 = float(d @ d)
    wx = X - p0[0]
    wy = Y - p0[1]
    u = np.clip((wx * d[0] + wy * d[1]) / max(L2, 1e-9), 0.0, 1.0)
    return np.hypot(wx - u * d[0], wy - u * d[1])


def _cosine_modes(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
                  n: int, lam_lo: float, lam_hi: float) -> np.ndarray:
    """Sum of ``n`` random plane cosines; smooth, seeded, and cheap."""
    out = np.zeros_like(X)
    for _ in range(n):
        lam = rng.uniform(lam_lo, lam_hi)
        th = rng.uniform(0.0, 2.0 * np.pi)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.6, 1.0)
        out += amp * np.cos(2.0 * np.pi * (X * np.cos(th) + Y * np.sin(th)) / lam + ph)
    return out


def build_heightfield(cfg: SceneCfg, rng: np.random.Generator) -> Heightfield:
    """Terrain, canopy and per-cell class on a regular grid."""
    x0 = -cfg.context
    y0 = -cfg.context
    x1 = cfg.survey_x + cfg.context
    y1 = cfg.survey_y + cfg.context
    nx = int(round((x1 - x0) / cfg.grid_dx)) + 1
    ny = int(round((y1 - y0) / cfg.grid_dx)) + 1
    x = x0 + cfg.grid_dx * np.arange(nx)
    y = y0 + cfg.grid_dx * np.arange(ny)
    X, Y = np.meshgrid(x, y)

    # 1. constant slope
    phi = np.radians(cfg.slope_azimuth_deg)
    z = np.tan(np.radians(cfg.slope_deg)) * (X * np.cos(phi) + Y * np.sin(phi))

    # 2. macro relief, normalised over the survey area so the config value means
    #    what it says regardless of which modes were drawn
    relief = _cosine_modes(X, Y, rng, cfg.n_relief_modes, *cfg.relief_lambda)
    inside = ((X >= 0.0) & (X <= cfg.survey_x) & (Y >= 0.0) & (Y <= cfg.survey_y))
    span = float(relief[inside].max() - relief[inside].min())
    z = z + relief * (2.0 * cfg.relief_amp / max(span, 1e-6))

    # 3. drainage ditch: flat-bottomed, smooth walls
    dist = _segment_distance(X, Y, cfg.ditch_p0, cfg.ditch_p1)
    ditch = 1.0 - _smoothstep(dist / cfg.ditch_halfwidth)
    z = z - cfg.ditch_depth * ditch

    # 4. furrows at row spacing, amplitude drifting slowly across the field
    amp_lo, amp_hi = cfg.furrow_amp
    amp = amp_lo + (amp_hi - amp_lo) * 0.5 * (
        1.0 + np.cos(2.0 * np.pi * (X / 71.0 + Y / 53.0)))
    z = z + amp * np.cos(2.0 * np.pi * Y / cfg.row_spacing)

    # 5. broadband roughness so no surface anywhere is exactly planar
    z = z + cfg.roughness * _cosine_modes(X, Y, rng, 12, 0.8, 4.0) / 3.0

    z_terrain = z

    # ---- canopy ----
    canopy_h, class_id = _canopy(cfg, X, Y, dist, rng)
    z_surface = z_terrain + canopy_h

    return Heightfield(x0=x0, y0=y0, dx=cfg.grid_dx,
                       z_terrain=z_terrain.astype(np.float32),
                       z_surface=z_surface.astype(np.float32),
                       canopy_h=canopy_h.astype(np.float32),
                       class_id=class_id.astype(np.uint8),
                       ditch_p0=tuple(cfg.ditch_p0), ditch_p1=tuple(cfg.ditch_p1))


def _canopy(cfg: SceneCfg, X: np.ndarray, Y: np.ndarray, ditch_dist: np.ndarray,
            rng: np.random.Generator):
    """Blocked canopy: healthy, stressed and bare patches.

    Uniform crop reads as a featureless mat.  Blocks give the canopy render
    internal structure and give the camera something to look at.
    """
    n_blocks = cfg.block_nx * cfg.block_ny
    heights = np.full(n_blocks, cfg.canopy_healthy)
    order = rng.permutation(n_blocks)
    heights[order[:cfg.n_bare_blocks]] = 0.0
    heights[order[cfg.n_bare_blocks:cfg.n_bare_blocks + cfg.n_stressed_blocks]] = cfg.canopy_stressed

    bx = np.clip((X / cfg.survey_x * cfg.block_nx).astype(int), 0, cfg.block_nx - 1)
    by = np.clip((Y / cfg.survey_y * cfg.block_ny).astype(int), 0, cfg.block_ny - 1)
    block = by * cfg.block_nx + bx
    H = heights[block]

    # rows: the canopy is tallest on the ridge and thins in the furrow
    row = 0.75 + 0.25 * np.cos(2.0 * np.pi * Y / cfg.row_spacing)
    canopy = H * row + rng.normal(scale=0.03, size=X.shape)

    outside = (X < 0.0) | (X > cfg.survey_x) | (Y < 0.0) | (Y > cfg.survey_y)
    canopy[outside] = 0.0
    canopy[ditch_dist < cfg.canopy_ditch_clearance] = 0.0
    canopy = np.maximum(canopy, 0.0)

    # Below the stubble threshold the cell is bare, and bare means bare: the
    # surface the LiDAR sees is exactly the bare-earth surface there.  Leaving a
    # few centimetres of stray canopy on a ground-classified cell would put a
    # systematic positive bias into every check that scores ground returns
    # against the terrain mesh, and gate 1 would be measuring the scene rather
    # than the sensor.
    stubble = 0.15
    canopy = np.where(canopy < stubble, 0.0, canopy)

    class_id = np.full(X.shape, CLASS_GROUND, dtype=np.uint8)
    mid = 0.5 * (cfg.canopy_stressed + cfg.canopy_healthy)
    class_id[canopy > 0.0] = CLASS_CANOPY_STRESSED
    class_id[canopy > mid] = CLASS_CANOPY_HEALTHY
    return canopy, class_id


def sample_bilinear(hf: Heightfield, xy: np.ndarray, which: str = "terrain") -> np.ndarray:
    """Bilinear lookup, clamped at the grid edge."""
    z = {"terrain": hf.z_terrain, "surface": hf.z_surface, "canopy": hf.canopy_h}[which]
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    ny, nx = z.shape
    fx = np.clip((xy[:, 0] - hf.x0) / hf.dx, 0.0, nx - 1.0000001)
    fy = np.clip((xy[:, 1] - hf.y0) / hf.dx, 0.0, ny - 1.0000001)
    i0 = fx.astype(int)
    j0 = fy.astype(int)
    tx = fx - i0
    ty = fy - j0
    z00 = z[j0, i0]
    z10 = z[j0, i0 + 1]
    z01 = z[j0 + 1, i0]
    z11 = z[j0 + 1, i0 + 1]
    return ((1 - ty) * ((1 - tx) * z00 + tx * z10)
            + ty * ((1 - tx) * z01 + tx * z11))


def heightfield_to_mesh(hf: Heightfield, which: str = "surface",
                        stride: int = 1):
    """Triangulate the grid.

    Vertices are ordered row-major so a face index maps back to a grid cell,
    which is how per-face classes are carried into the ray caster.
    """
    z = {"terrain": hf.z_terrain, "surface": hf.z_surface}[which][::stride, ::stride]
    cls = hf.class_id[::stride, ::stride]
    ny, nx = z.shape
    dx = hf.dx * stride
    x = hf.x0 + dx * np.arange(nx)
    y = hf.y0 + dx * np.arange(ny)
    X, Y = np.meshgrid(x, y)
    V = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1).astype(np.float64)

    j, i = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    v00 = (j * nx + i).ravel()
    v10 = v00 + 1
    v01 = v00 + nx
    v11 = v01 + 1
    F = np.concatenate([np.stack([v00, v10, v11], axis=1),
                        np.stack([v00, v11, v01], axis=1)], axis=0).astype(np.int64)

    # A face is bare ground only if every corner it spans is bare.  At a block
    # boundary a quad with one bare corner and three in the crop is a ramp
    # climbing to canopy height; calling it ground would put a systematic
    # positive bias into any check that scores ground returns against the
    # terrain, and gate 1 would end up measuring the scene instead of the sensor.
    quad = np.maximum.reduce([cls[:-1, :-1], cls[:-1, 1:], cls[1:, :-1], cls[1:, 1:]])
    face_class = np.concatenate([quad.ravel(), quad.ravel()])
    return V, F, face_class.astype(np.uint8)
