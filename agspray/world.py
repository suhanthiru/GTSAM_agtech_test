"""The static field: crop landmarks, flight lines, treeline, sun.

Visual landmarks are laid out on a grid over the whole field and given
descriptors drawn from a smooth random field in ``(x, y mod row_spacing)``.
That modulus is the whole point.  Controlled-traffic farming puts tramlines at
the boom width, so the ground genuinely repeats with a period equal to the row
spacing, and two plants one row apart really do look alike.  False loop
closures then fall out of the geometry instead of being injected by hand, and a
map fold lands exactly one row spacing off - which is the failure the coverage
metrics are built to catch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config

DESCRIPTOR_DIM = 6
_N_FOURIER = 24


@dataclass
class World:
    """Everything about the field that does not move (before the wind gets to it)."""

    landmarks: np.ndarray      # (N, 3) nominal positions in the world frame
    descriptors: np.ndarray    # (N, DESCRIPTOR_DIM) unit rows
    row_y: np.ndarray          # (n_rows,) y of each flight line
    treeline_y: float          # y of the windbreak, outside the field
    sun_dir: np.ndarray        # unit vector from the field towards the sun
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]

    @property
    def n_landmarks(self) -> int:
        return int(self.landmarks.shape[0])

    def multipath_zone(self, xy: np.ndarray, radius: float) -> np.ndarray:
        """Boolean mask: which of the given ``(..., 2)`` positions sit in the
        degraded-GNSS band along the treeline."""
        return np.abs(np.asarray(xy)[..., 1] - self.treeline_y) < radius


def _smooth_random_field(q: np.ndarray, rng: np.random.Generator,
                         wavelengths: np.ndarray) -> np.ndarray:
    """Random Fourier features over 2-D query points ``q`` of shape (N, 2).

    Returns (N, DESCRIPTOR_DIM).  Smooth in ``q``, so nearby ground looks alike
    and distant ground does not, which is what a real descriptor does.
    """
    n_terms = _N_FOURIER
    directions = rng.normal(size=(n_terms, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    scale = 2.0 * math.pi / rng.choice(wavelengths, size=n_terms)
    omega = directions * scale[:, None]
    phase = rng.uniform(0.0, 2.0 * math.pi, size=n_terms)
    weights = rng.normal(size=(n_terms, DESCRIPTOR_DIM)) / math.sqrt(n_terms)
    basis = np.cos(q @ omega.T + phase)          # (N, n_terms)
    return basis @ weights


def build_world(cfg: Config, rng: np.random.Generator) -> World:
    """Lay out the field for one trial.

    The same ``rng`` stream is used by every method in a paired-seed trial, so
    all six estimators fly over an identical field.
    """
    fld = cfg.field_
    der = cfg.derived
    spacing = fld.row_spacing
    row_y = np.arange(fld.n_rows, dtype=float) * spacing

    # Landmark grid.  Density is chosen so a camera footprint holds roughly the
    # configured feature budget rather than being an arbitrary number.
    pitch_x = max(fld.plant_pitch, 0.1)
    pitch_y = spacing / 3.0
    margin = spacing
    xs = np.arange(-margin, fld.length_x + margin + 1e-9, pitch_x)
    ys = np.arange(-margin, der.field_width + margin + 1e-9, pitch_y)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts += rng.normal(scale=fld.plant_jitter, size=pts.shape)

    # Canopy top is the surface the camera sees; give it a little relief so the
    # projection is not perfectly planar (a planar scene is degenerate for
    # structure-from-motion and would flatter every method equally).
    relief = 0.25 * np.sin(pts[:, 0] / 17.0) * np.cos(pts[:, 1] / 11.0)
    z = relief + rng.normal(scale=0.05, size=pts.shape[0])
    landmarks = np.column_stack([pts, z])

    # Descriptors: smooth in along-row position, periodic across rows.  The
    # row-identifying component is scaled down by the aliasing severity, so
    # severity 1 makes adjacent rows indistinguishable.
    alias = float(np.clip(cfg.degradation.row_aliasing, 0.0, 1.0))
    y_phase = np.mod(pts[:, 1], spacing)
    q_shared = np.column_stack([pts[:, 0], y_phase])
    wavelengths = np.array([6.0, 11.0, 19.0, 31.0])
    shared = _smooth_random_field(q_shared, rng, wavelengths)

    # A per-row signature that a non-aliased field would carry.
    row_index = np.rint(pts[:, 1] / spacing).astype(int)
    row_index = np.clip(row_index, 0, fld.n_rows - 1)
    row_sig = rng.normal(size=(fld.n_rows, DESCRIPTOR_DIM))
    distinct = row_sig[row_index]

    desc = alias * shared + (1.0 - alias) * distinct
    desc += rng.normal(scale=0.02, size=desc.shape)
    desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-9)

    az = math.radians(fld.sun_azimuth_deg)
    el = math.radians(fld.sun_elevation_deg)
    sun_dir = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])

    return World(
        landmarks=landmarks,
        descriptors=desc,
        row_y=row_y,
        treeline_y=-fld.treeline_offset,
        sun_dir=sun_dir,
        x_bounds=(0.0, fld.length_x),
        y_bounds=(0.0, der.field_width),
    )
