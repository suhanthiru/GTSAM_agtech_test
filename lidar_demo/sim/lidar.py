"""The spinning LiDAR, including the detail everything else rests on.

**Points are stored in the sensor frame at their own timestamp.**  A sweep takes
100 ms, during which the aircraft moves half a metre, so a real spinning sensor
does not deliver a coherent snapshot: it delivers ranges along fixed beam
directions, each measured at a different instant from a different place.  That
is what this model produces, and that is what goes in the file.  The consumer
deskews it.

The simulation honours that by casting each azimuth column from the *true* pose
at the column's own sub-sweep time.  Because the flight path is analytic, that
pose is exact rather than interpolated, so the recorded data has a real motion
smear in it and no numerical smear on top.

Geometry stays clean.  Range noise is 2 cm, dropout and intensity are physical,
and nothing else is corrupted here.  All the damage in the demo comes from
navigation and from the mounting angle, which is where it can be repaired.

The other thing this model has to get right is that a pulse does not stop at the
top of the crop.  Some fraction of every shot filters through the canopy and
returns from the soil, and the surviving fraction falls off exponentially with
the path length through vegetation.  Without that there is no bare-earth surface
under the wheat at all -- no terrain raster, no canopy-height raster, and a
cross-section with the ground visible only in the bare patches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import LidarCfg
from ..frames import beam_dirs
from .backend import RayCastBackend, RayHits
from .flight import LegPath
from .scene import (ALBEDO, CLASS_TREE_CROWN, CLASS_GROUND)

CANOPY_CLASSES = (2, 3)


@dataclass
class LidarModel:
    """Beam table and column timing.  Fixed for the whole flight."""

    ring_el: np.ndarray      # (n_rings,) radians, ring 0 at the top
    col_az: np.ndarray       # (n_cols,) radians
    col_dt: np.ndarray       # (n_cols,) seconds after the sweep start
    dirs_S: np.ndarray       # (n_rings, n_cols, 3) unit, sensor frame
    period: float

    @classmethod
    def from_cfg(cls, cfg: LidarCfg) -> "LidarModel":
        hi, lo = max(cfg.vfov_deg), min(cfg.vfov_deg)
        ring_el = np.radians(np.linspace(hi, lo, cfg.n_rings))
        col_az = 2.0 * np.pi * np.arange(cfg.n_cols) / cfg.n_cols
        period = 1.0 / cfg.rate_hz
        col_dt = period * np.arange(cfg.n_cols) / cfg.n_cols
        return cls(ring_el=ring_el, col_az=col_az, col_dt=col_dt,
                   dirs_S=beam_dirs(col_az, ring_el), period=period)

    @property
    def n_rings(self) -> int:
        return int(self.ring_el.size)

    @property
    def n_cols(self) -> int:
        return int(self.col_az.size)


@dataclass
class RawSweep:
    """One revolution as measured, before it is written out."""

    index: int
    t_start: float
    t_offset: np.ndarray
    xyz: np.ndarray
    range: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    col: np.ndarray
    hit_class: np.ndarray
    col_t: np.ndarray
    col_R_WB: np.ndarray
    col_p_WB: np.ndarray


def _albedo_of(class_id: np.ndarray) -> np.ndarray:
    a = np.full(class_id.shape, 0.3, dtype=float)
    for cid, val in ALBEDO.items():
        a[class_id == cid] = val
    return a


def apply_returns(cfg: LidarCfg, top: RayHits, under: RayHits, dirs_W: np.ndarray,
                  rng: np.random.Generator):
    """Turn geometric intersections into measured ranges, intensity and dropout.

    ``top`` is the first surface the beam meets; ``under`` is the same beam
    against the scene with the crop removed.  The gap between the two range
    values is the path length through vegetation, and the pulse reaches the soil
    with probability ``exp(-extinction * path)``.  That is Beer-Lambert, and it
    is why a survey over a standing crop still produces a bare-earth model: the
    minority of shots that get all the way down are the ones that describe it.

    Shots that do not penetrate return from inside the canopy rather than from
    its very top, which is the small positive range bias.  Dropout rises steeply
    at grazing incidence, thinning the far edge of every swath.
    """
    t_top = top.t_hit.astype(np.float64)
    t_und = under.t_hit.astype(np.float64)
    valid = np.isfinite(t_top) & (t_top >= cfg.min_range) & (t_top <= cfg.max_range)

    normal = top.normal.astype(np.float64)
    class_id = top.class_id.copy()

    veg = np.isin(top.class_id, CANOPY_CLASSES) | (top.class_id == CLASS_TREE_CROWN)
    veg &= valid & np.isfinite(t_und)

    # Beer-Lambert through the vegetation the beam actually crossed.
    path = np.where(veg, np.maximum(t_und - t_top, 0.0), 0.0)
    mu = np.where(top.class_id == CLASS_TREE_CROWN, cfg.crown_extinction,
                  cfg.canopy_extinction)
    through = veg & (rng.random(t_top.shape) < np.exp(-mu * path))

    t = np.where(through, t_und, t_top)
    normal = np.where(through[:, None], under.normal.astype(np.float64), normal)
    class_id = np.where(through, under.class_id, class_id)

    cos_inc = np.abs(np.einsum("ij,ij->i", dirs_W, normal))
    cos_inc = np.clip(np.where(np.isfinite(cos_inc), cos_inc, 1.0), 0.0, 1.0)

    # Returns stopped by vegetation come from inside it, not off its skin.
    stopped_crop = veg & ~through & (top.class_id != CLASS_TREE_CROWN)
    stopped_crown = veg & ~through & (top.class_id == CLASS_TREE_CROWN)
    rng_out = t.copy()
    if stopped_crop.any():
        depth = rng.exponential(cfg.canopy_penetration, int(stopped_crop.sum()))
        rng_out[stopped_crop] += np.minimum(depth, path[stopped_crop])
    if stopped_crown.any():
        depth = rng.exponential(cfg.crown_penetration, int(stopped_crown.sum()))
        rng_out[stopped_crown] += np.minimum(depth, path[stopped_crown])
    rng_out = rng_out + rng.normal(scale=cfg.range_sigma, size=t.shape)

    valid = valid & (rng_out >= cfg.min_range) & (rng_out <= cfg.max_range)

    p_drop = cfg.base_dropout + cfg.grazing_dropout * (1.0 - cos_inc) ** 4
    p_drop = np.where(stopped_crop, p_drop + cfg.canopy_dropout, p_drop)
    p_drop = np.where(stopped_crown, p_drop + cfg.crown_dropout, p_drop)
    keep = valid & (rng.random(t.shape) >= np.clip(p_drop, 0.0, 0.995))

    intensity = (_albedo_of(class_id) * cos_inc ** 0.6
                 * np.exp(-rng_out / 150.0) + rng.normal(scale=0.02, size=t.shape))
    return rng_out, np.clip(intensity, 0.0, 1.0), keep, class_id


def simulate_sweep(model: LidarModel, backend: RayCastBackend, path: LegPath,
                   index: int, R_BS: np.ndarray, t_BS: np.ndarray,
                   cfg: LidarCfg, rng: np.random.Generator) -> RawSweep:
    """One revolution, cast column by column from the true pose at each instant.

    ``R_BS`` and ``t_BS`` are the *true* mount, including the boresight error.
    The manifest never sees them.
    """
    t_start = index * model.period
    col_t = t_start + model.col_dt
    R_WB, p_WB = path.pose(col_t)                      # (C, 3, 3), (C, 3)

    # Sensor pose per column, then one ray per (ring, column).
    R_WS = np.einsum("cij,jk->cik", R_WB, R_BS)        # (C, 3, 3)
    p_WS = p_WB + np.einsum("cij,j->ci", R_WB, t_BS)   # (C, 3)

    n_r, n_c = model.n_rings, model.n_cols

    # Aperture: keep only beams within the usable scan angle of body nadir.
    # This is applied to the *nominal* beam directions in the body frame, so it
    # is a property of the instrument and its mount, not of what was hit, and it
    # costs nothing to cast because the rejected rays are never cast at all.
    dirs_B = model.dirs_S @ R_BS.T                     # (R, C, 3)
    cos_nadir = -dirs_B[..., 2]
    aperture = cos_nadir >= np.cos(np.radians(cfg.max_scan_angle_deg))
    flat_ap = aperture.reshape(-1)
    idx_ap = np.flatnonzero(flat_ap)
    if idx_ap.size == 0:
        raise ValueError("the scan-angle aperture rejects every beam; check "
                         "lidar.max_scan_angle_deg against the mount pitch")

    dirs_W = np.einsum("cij,rcj->rci", R_WS, model.dirs_S).reshape(-1, 3)[idx_ap]
    origins = np.broadcast_to(p_WS[None, :, :], (n_r, n_c, 3)).reshape(-1, 3)[idx_ap]

    origins = np.ascontiguousarray(origins, np.float32)
    dirs_W = np.ascontiguousarray(dirs_W, np.float32)
    top = backend.cast(origins, dirs_W, "surface")
    under = backend.cast(origins, dirs_W, "ground")
    rng_m, intensity, keep, class_id = apply_returns(cfg, top, under, dirs_W, rng)

    ring_idx, col_idx = np.divmod(idx_ap, n_c)
    sel = np.flatnonzero(keep)

    # The stored coordinate: measured range along the *nominal* beam direction,
    # in the sensor frame of that column's own instant.  No compensation.
    xyz = model.dirs_S.reshape(-1, 3)[idx_ap][sel] * rng_m[sel, None]

    return RawSweep(
        index=index,
        t_start=float(t_start),
        t_offset=model.col_dt[col_idx[sel]].astype(np.float32),
        xyz=xyz.astype(np.float32),
        range=rng_m[sel].astype(np.float32),
        intensity=intensity[sel].astype(np.float32),
        ring=ring_idx[sel].astype(np.uint8),
        col=col_idx[sel].astype(np.uint16),
        hit_class=class_id[sel],
        col_t=col_t,
        col_R_WB=R_WB.astype(np.float32),
        col_p_WB=p_WB,
    )


def project_sweep_truth(sweep: RawSweep, R_BS: np.ndarray, t_BS: np.ndarray) -> np.ndarray:
    """World points from a sweep using its own per-column true poses.

    This is the gate-1 projection: it uses the truth trajectory and a given
    extrinsic, and nothing else.  Handed the *true* extrinsic it must reproduce
    the terrain; handed the nominal one it must not.
    """
    R_WB = sweep.col_R_WB[sweep.col].astype(np.float64)
    p_WB = sweep.col_p_WB[sweep.col]
    p_B = sweep.xyz.astype(np.float64) @ np.asarray(R_BS).T + np.asarray(t_BS)
    return np.einsum("nij,nj->ni", R_WB, p_B) + p_WB


def ground_density(points_W: np.ndarray, x_range: tuple[float, float],
                   y_range: tuple[float, float], cell: float = 1.0) -> np.ndarray:
    """Points per square metre in 1 m bins, for the density check."""
    nx = int((x_range[1] - x_range[0]) / cell)
    ny = int((y_range[1] - y_range[0]) / cell)
    ix = ((points_W[:, 0] - x_range[0]) / cell).astype(int)
    iy = ((points_W[:, 1] - y_range[0]) / cell).astype(int)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    counts = np.bincount(iy[m] * nx + ix[m], minlength=nx * ny)
    return counts.reshape(ny, nx) / (cell * cell)
