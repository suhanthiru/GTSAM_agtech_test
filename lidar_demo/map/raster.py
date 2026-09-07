"""Bare-earth and canopy-height rasters on a 10 cm grid.

Bare earth is a low percentile of the points in each cell and the canopy top is
a high one.  That works here for the reason it works in practice: a fraction of
every pulse filters through the crop and returns from the soil, so both surfaces
are present in the same cloud and separating them is a question of asking each
cell for its floor and its ceiling.

The grouping is done by sorting rather than by looping over cells.  A 120 by 84
metre field at 10 cm is a million cells and the cloud has millions of points, so
a per-cell loop is minutes and a lexsort is under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Raster:
    x0: float
    y0: float
    cell: float
    ground: np.ndarray       # (ny, nx) bare earth, NaN where nothing was seen
    canopy: np.ndarray       # (ny, nx) height above ground, NaN likewise
    count: np.ndarray        # (ny, nx) points per cell
    label: str = ""

    @property
    def extent(self):
        ny, nx = self.ground.shape
        return [self.x0, self.x0 + nx * self.cell,
                self.y0, self.y0 + ny * self.cell]

    @property
    def x(self) -> np.ndarray:
        return self.x0 + self.cell * (np.arange(self.ground.shape[1]) + 0.5)

    @property
    def y(self) -> np.ndarray:
        return self.y0 + self.cell * (np.arange(self.ground.shape[0]) + 0.5)

    def coverage(self) -> float:
        return float(np.isfinite(self.ground).mean())


def _group_percentiles(cell_id, z, n_cells, pct_lo, pct_hi, min_pts):
    """Low and high percentile of ``z`` within each cell, in one sorted pass."""
    order = np.lexsort((z, cell_id))
    cid, zs = cell_id[order], z[order]
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    ends = np.r_[starts[1:], cid.size]
    counts = ends - starts

    lo = np.full(n_cells, np.nan)
    hi = np.full(n_cells, np.nan)
    cnt = np.zeros(n_cells, dtype=np.int32)

    keep = counts >= min_pts
    s, e, c = starts[keep], ends[keep], counts[keep]
    ids = cid[s]
    # z is already sorted within each group, so a percentile is an index.
    lo[ids] = zs[s + np.floor(pct_lo / 100.0 * (c - 1)).astype(np.int64)]
    hi[ids] = zs[s + np.floor(pct_hi / 100.0 * (c - 1)).astype(np.int64)]
    cnt[cid[starts]] = counts
    return lo, hi, cnt


def rasterise(xyz: np.ndarray, cell: float = 0.10,
              bounds=None, ground_pct: float = 5.0, canopy_pct: float = 95.0,
              min_pts: int = 3, label: str = "") -> Raster:
    """Bare-earth and canopy-height grids from a world-frame cloud."""
    xyz = np.asarray(xyz)
    if bounds is None:
        bounds = (float(xyz[:, 0].min()), float(xyz[:, 0].max()),
                  float(xyz[:, 1].min()), float(xyz[:, 1].max()))
    x0, x1, y0, y1 = bounds
    nx = max(int(np.ceil((x1 - x0) / cell)), 1)
    ny = max(int(np.ceil((y1 - y0) / cell)), 1)

    ix = ((xyz[:, 0] - x0) / cell).astype(np.int64)
    iy = ((xyz[:, 1] - y0) / cell).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cid = (iy[inside] * nx + ix[inside]).astype(np.int64)
    z = xyz[inside, 2].astype(np.float64)

    lo, hi, cnt = _group_percentiles(cid, z, nx * ny, ground_pct, canopy_pct,
                                     min_pts)
    ground = lo.reshape(ny, nx)
    canopy = (hi - lo).reshape(ny, nx)
    return Raster(x0=x0, y0=y0, cell=cell, ground=ground, canopy=canopy,
                  count=cnt.reshape(ny, nx), label=label)


def profile(raster: Raster, x0: float, half_width: float,
            which: str = "ground") -> tuple[np.ndarray, np.ndarray]:
    """A cross-section line: median of the raster over a slab of columns."""
    grid = raster.ground if which == "ground" else raster.canopy
    x = raster.x
    m = np.abs(x - x0) <= half_width
    if not m.any():
        return raster.y, np.full(raster.y.size, np.nan)
    band = grid[:, m]
    # Columns with nothing in them are normal near the field edge and under the
    # densest crop; they come back as NaN rather than as a warning.
    line = np.full(band.shape[0], np.nan)
    any_pt = np.isfinite(band).any(axis=1)
    if any_pt.any():
        line[any_pt] = np.nanmedian(band[any_pt], axis=1)
    return raster.y, line


def compare(raster: Raster, truth_fn) -> dict:
    """Residual statistics of a bare-earth raster against the true surface."""
    X, Y = np.meshgrid(raster.x, raster.y)
    ok = np.isfinite(raster.ground)
    if not ok.any():
        return {"n": 0}
    z_true = truth_fn(np.stack([X[ok], Y[ok]], axis=1))
    d = raster.ground[ok] - z_true
    return {
        "n": int(ok.sum()),
        "coverage": float(ok.mean()),
        "rms_m": float(np.sqrt((d ** 2).mean())),
        "mean_m": float(d.mean()),
        "p95_abs_m": float(np.percentile(np.abs(d), 95)),
    }


def save(raster: Raster, path) -> None:
    np.savez(path, x0=np.float64(raster.x0), y0=np.float64(raster.y0),
             cell=np.float64(raster.cell), ground=raster.ground.astype(np.float32),
             canopy=raster.canopy.astype(np.float32), count=raster.count,
             label=np.array(raster.label))


def load(path) -> Raster:
    d = np.load(path, allow_pickle=False)
    return Raster(x0=float(d["x0"]), y0=float(d["y0"]), cell=float(d["cell"]),
                  ground=d["ground"], canopy=d["canopy"], count=d["count"],
                  label=str(d["label"]) if "label" in d else "")
