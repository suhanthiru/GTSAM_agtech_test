"""The cross-section shot.

A one-metre slab cut perpendicular to the flight lines, viewed edge-on under
parallel projection so the ground becomes a line and the truth profile can be
laid over it.  This is the only view in which either failure is legible: from
above, a map that floats half a metre high and ripples between flight lines
looks exactly like a map that does not.

The section is drawn *flattened*: every point is plotted at its height above the
true ground rather than at its absolute height, which turns the reference
profile into a straight line at zero and puts the whole vertical range of the
picture at the disposal of the error.  This is not a trick and it is not
optional here.  The terrain swings four metres across the field while the
corrugation is twenty centimetres, so at true scale the thing the shot exists to
show occupies two percent of the frame.  Removing the surface you are comparing
against is what a survey does to inspect exactly this, the white line is still
the truth, and the vertical scale is stated on screen.
"""

from __future__ import annotations

import numpy as np

from .camera import edge_on
from .scene import add_cloud, add_profile_line


def slab(xyz: np.ndarray, x0: float, half: float,
         y_range: tuple[float, float] | None = None) -> np.ndarray:
    """Indices of the points inside the slab."""
    m = np.abs(xyz[:, 0] - x0) < half
    if y_range is not None:
        m &= (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] <= y_range[1])
    return np.flatnonzero(m)


def bare_earth(xyz: np.ndarray, cell: float = 0.75, pct: float = 4.0,
               band: float = 0.30, min_pts: int = 6) -> np.ndarray:
    """Indices of the returns that came from the soil rather than the crop.

    The section has to show the ground, and a slab through a standing crop is
    mostly crop: at any useful vertical scale a 1.2 m canopy is off the top of
    the frame and the twenty centimetres of error underneath it are invisible.

    The filter is the same one the raster uses and it reads nothing but the
    cloud: bin along the section, take a low percentile in each bin, keep what
    sits close to it.  So the picture is made from the map, not from the answer.
    """
    y = xyz[:, 1]
    idx = np.floor((y - y.min()) / cell).astype(np.int64)
    order = np.argsort(idx, kind="stable")
    idx_s = idx[order]
    z_s = xyz[order, 2]
    starts = np.flatnonzero(np.r_[True, idx_s[1:] != idx_s[:-1]])
    ends = np.r_[starts[1:], idx_s.size]

    keep = np.zeros(idx_s.size, dtype=bool)
    for a, b in zip(starts, ends):
        if b - a < min_pts:
            continue
        floor = np.percentile(z_s[a:b], pct)
        keep[a:b] = np.abs(z_s[a:b] - floor) < band
    return order[keep]


def flatten(xyz: np.ndarray, hf, exaggeration: float = 1.0,
            x0: float | None = None) -> np.ndarray:
    """Replace absolute height with height above the true ground, exaggerated.

    Points keep their ``y``.  Their ``x`` is collapsed onto the section plane so
    the slab does not read as a wedge under parallel projection.
    """
    from ..sim.terrain import sample_bilinear

    z_ref = sample_bilinear(hf, xyz[:, :2], "terrain")
    out = np.empty_like(xyz, dtype=np.float32)
    out[:, 0] = xyz[:, 0] if x0 is None else x0
    out[:, 1] = xyz[:, 1]
    out[:, 2] = (xyz[:, 2] - z_ref) * exaggeration
    return out


def flat_reference(x0: float, y0: float, y1: float, n: int = 400) -> np.ndarray:
    """The truth, which in a flattened section is a straight line at zero."""
    y = np.linspace(y0, y1, n)
    return np.stack([np.full_like(y, x0), y, np.zeros_like(y)], axis=1)


def truth_profile(hf, x0: float, y0: float, y1: float, n: int = 900,
                  which: str = "terrain") -> np.ndarray:
    from ..sim.terrain import sample_bilinear

    y = np.linspace(y0, y1, n)
    q = np.stack([np.full_like(y, x0), y], axis=1)
    z = sample_bilinear(hf, q, which)
    return np.stack([np.full_like(y, x0), y, z], axis=1)


def frame_camera(x0: float, y_range, half_height: float):
    """Frame a flattened section: ``y`` across the picture, error up it."""
    centre = (x0, 0.5 * (y_range[0] + y_range[1]), 0.0)
    return edge_on(centre, distance=300.0, half_height=half_height,
                   along=(1.0, 0.0, 0.0))


def raster_section(raster, hf, x0: float, half: float, y_range,
                   exaggeration: float, smooth_m: float = 1.5):
    """The bare-earth surface across the flight lines, as a band of points.

    Built from the finished 10 cm raster rather than from raw returns, for two
    reasons.  The raster *is* the bare-earth product, so this shows the thing
    the survey would deliver.  And a raw slab through a crop is dominated by two
    signals that are not the washboard: the canopy, an order of magnitude taller
    than the error, and an alias of the furrows, because a boresight yaw error
    slides points sideways and a 2 m furrow then reads as a 20 cm height
    residual.  Smoothing along the section over rather more than a row spacing
    removes the second; the raster's own low percentile removed the first.

    Returns points at ``(x0, y, residual * exaggeration)``, one per raster cell
    in the band, so blue and green share a grid and can be morphed cell for cell.
    """
    from ..sim.terrain import sample_bilinear

    xs_ = raster.x
    ys_ = raster.y
    cols = np.flatnonzero(np.abs(xs_ - x0) <= half)
    rows = np.flatnonzero((ys_ >= y_range[0]) & (ys_ <= y_range[1]))
    if cols.size == 0 or rows.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32)

    grid = raster.ground[np.ix_(rows, cols)]
    Y = np.repeat(ys_[rows][:, None], cols.size, axis=1)
    X = np.repeat(xs_[cols][None, :], rows.size, axis=0)
    z_true = sample_bilinear(hf, np.stack([X.ravel(), Y.ravel()], axis=1),
                             "terrain").reshape(grid.shape)
    resid = grid - z_true

    w = max(int(round(smooth_m / raster.cell)) | 1, 3)
    kern = np.ones(w) / w
    out = np.full_like(resid, np.nan)
    for c in range(resid.shape[1]):
        col = resid[:, c]
        ok = np.isfinite(col)
        if ok.sum() < w:
            continue
        filled = np.interp(np.arange(col.size), np.flatnonzero(ok), col[ok])
        sm = np.convolve(filled, kern, mode="same")
        sm[: w // 2] = np.nan
        sm[-(w // 2):] = np.nan
        out[:, c] = np.where(ok | np.isfinite(sm), sm, np.nan)

    good = np.isfinite(out)
    pts = np.stack([np.full(int(good.sum()), x0, np.float32),
                    Y[good].astype(np.float32),
                    (out[good] * exaggeration).astype(np.float32)], axis=1)
    key = (np.flatnonzero(good.ravel())).astype(np.int64)
    return pts, key


def align_sections(pts_a, key_a, pts_b, key_b):
    """Keep only the cells both rasters filled, so the morph is cell for cell."""
    common, ia, ib = np.intersect1d(key_a, key_b, return_indices=True)
    return pts_a[ia], pts_b[ib]


def build(pl, pts, colour: str, x0: float, y_range, point_size: float = 2.2,
          alpha: float = 0.8):
    """Add a section band and the reference line at zero."""
    mesh, actor = add_cloud(pl, pts, colour, point_size, alpha)
    add_profile_line(pl, flat_reference(x0, *y_range), colour="#FFFFFF",
                     width=3.0)
    return mesh, actor
