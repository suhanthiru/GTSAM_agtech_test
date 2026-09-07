"""Triangle-mesh primitives for the props.

Deliberately plain numpy: a handful of shapes, no dependency, and every one of
them closed enough that a ray caster cannot fall through it.  The props exist
for two reasons at once.  They are the art direction (windmill, scarecrow,
farmhouse breaking the horizon) and they are the geometry that makes scan
matching converge, because a wheat canopy is nearly featureless and a tall
isolated vertical is not.
"""

from __future__ import annotations

import numpy as np


def _fan(center_index: int, ring: np.ndarray, flip: bool = False) -> np.ndarray:
    n = ring.size
    a = ring
    b = np.roll(ring, -1)
    c = np.full(n, center_index)
    return np.stack([c, b, a] if flip else [c, a, b], axis=1)


def _tube(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Quad strip between two equal-length index rings."""
    n = lower.size
    a, b = lower, np.roll(lower, -1)
    c, d = upper, np.roll(upper, -1)
    return np.concatenate([np.stack([a, b, d], axis=1),
                           np.stack([a, d, c], axis=1)], axis=0)


def frustum(r0: float, r1: float, h: float, n: int = 16, cap: bool = True):
    """Cone frustum with its base at the origin, axis along +z."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    lo = np.stack([r0 * np.cos(th), r0 * np.sin(th), np.zeros(n)], axis=1)
    hi = np.stack([r1 * np.cos(th), r1 * np.sin(th), np.full(n, h)], axis=1)
    V = np.concatenate([lo, hi], axis=0)
    F = _tube(np.arange(n), np.arange(n) + n)
    if cap:
        V = np.concatenate([V, [[0.0, 0.0, 0.0], [0.0, 0.0, h]]], axis=0)
        F = np.concatenate([F,
                            _fan(2 * n, np.arange(n), flip=True),
                            _fan(2 * n + 1, np.arange(n) + n)], axis=0)
    return V, F.astype(np.int64)


def cylinder(r: float, h: float, n: int = 16, cap: bool = True):
    return frustum(r, r, h, n=n, cap=cap)


def box(sx: float, sy: float, sz: float):
    """Axis-aligned box, base centred on the origin, extending up in +z."""
    hx, hy = sx * 0.5, sy * 0.5
    V = np.array([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0],
                  [-hx, -hy, sz], [hx, -hy, sz], [hx, hy, sz], [-hx, hy, sz]])
    F = np.array([[0, 2, 1], [0, 3, 2],        # bottom
                  [4, 5, 6], [4, 6, 7],        # top
                  [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6],
                  [3, 0, 4], [3, 4, 7]], dtype=np.int64)
    return V, F


def gable(sx: float, sy: float, wall_h: float, ridge_h: float):
    """Pitched roof sitting on top of a box of the same footprint.

    The ridge runs along ``x``, which is what gives the farmhouse a recognisable
    silhouette against the horizon.
    """
    hx, hy = sx * 0.5, sy * 0.5
    V = np.array([[-hx, -hy, wall_h], [hx, -hy, wall_h],
                  [hx, hy, wall_h], [-hx, hy, wall_h],
                  [-hx, 0.0, ridge_h], [hx, 0.0, ridge_h]])
    F = np.array([[0, 1, 5], [0, 5, 4],        # front slope
                  [2, 3, 4], [2, 4, 5],        # back slope
                  [0, 4, 3],                   # gable ends
                  [1, 2, 5],
                  [0, 2, 1], [0, 3, 2]], dtype=np.int64)
    return V, F


def ellipsoid(a: float, b: float, c: float, n_u: int = 14, n_v: int = 8):
    """Ellipsoid centred on the origin.  Used for tree crowns."""
    u = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
    v = np.linspace(0.0, np.pi, n_v + 2)[1:-1]
    U, Vv = np.meshgrid(u, v)
    X = a * np.sin(Vv) * np.cos(U)
    Y = b * np.sin(Vv) * np.sin(U)
    Z = c * np.cos(Vv)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    rows = v.size
    F = []
    for j in range(rows - 1):
        lower = np.arange(n_u) + j * n_u
        upper = lower + n_u
        F.append(_tube(lower, upper))
    V = np.concatenate([V, [[0.0, 0.0, c], [0.0, 0.0, -c]]], axis=0)
    top, bot = V.shape[0] - 2, V.shape[0] - 1
    F.append(_fan(top, np.arange(n_u), flip=True))
    F.append(_fan(bot, np.arange(n_u) + (rows - 1) * n_u))
    return V, np.concatenate(F, axis=0).astype(np.int64)


def sphere(r: float, n_u: int = 12, n_v: int = 6):
    return ellipsoid(r, r, r, n_u=n_u, n_v=n_v)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def transform(V: np.ndarray, R: np.ndarray | None = None,
              t: np.ndarray | None = None) -> np.ndarray:
    out = np.asarray(V, dtype=float)
    if R is not None:
        out = out @ np.asarray(R, dtype=float).T
    if t is not None:
        out = out + np.asarray(t, dtype=float)
    return out


def concat(parts: list[tuple[np.ndarray, np.ndarray]]):
    """Merge ``(V, F)`` pairs, offsetting the face indices."""
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(np.asarray(V, dtype=float))
        Fs.append(np.asarray(F, dtype=np.int64) + off)
        off += V.shape[0]
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(Vs, axis=0), np.concatenate(Fs, axis=0)
