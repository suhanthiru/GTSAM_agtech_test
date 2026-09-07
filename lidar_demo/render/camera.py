"""Camera paths.

Keyframed positions and focal points with smoothstep easing between them, so a
move starts and ends at rest.  Nothing here is clever; it exists so the shots in
:mod:`lidar_demo.render.sequence` read as camera moves rather than as jumps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import smoothstep


@dataclass
class CameraKey:
    pos: tuple[float, float, float]
    focal: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    parallel_scale: float | None = None


def interp(keys: list[CameraKey], u: float) -> CameraKey:
    """Position along a list of keys, ``u`` in ``[0, 1]``, eased."""
    if len(keys) == 1:
        return keys[0]
    u = float(np.clip(u, 0.0, 1.0)) * (len(keys) - 1)
    i = min(int(u), len(keys) - 2)
    f = smoothstep(u - i)
    a, b = keys[i], keys[i + 1]

    def mix(p, q):
        return tuple((1 - f) * np.asarray(p, float) + f * np.asarray(q, float))

    scale = None
    if a.parallel_scale is not None and b.parallel_scale is not None:
        scale = (1 - f) * a.parallel_scale + f * b.parallel_scale
    return CameraKey(mix(a.pos, b.pos), mix(a.focal, b.focal), mix(a.up, b.up),
                     scale)


def apply(pl, key: CameraKey) -> None:
    pl.camera_position = [tuple(key.pos), tuple(key.focal), tuple(key.up)]
    if key.parallel_scale is not None:
        pl.enable_parallel_projection()
        pl.camera.parallel_scale = float(key.parallel_scale)


def orbit(centre, radius: float, height: float, az0: float, az1: float,
          n: int) -> list[CameraKey]:
    """A circular move around a point, useful for the pull-back."""
    az = np.radians(np.linspace(az0, az1, n))
    c = np.asarray(centre, float)
    return [CameraKey(pos=(c[0] + radius * np.cos(a), c[1] + radius * np.sin(a),
                           c[2] + height), focal=tuple(c)) for a in az]


def three_quarter(centre, distance: float, elevation_deg: float,
                  azimuth_deg: float) -> CameraKey:
    """The standard framing: high, off to one side, looking down at a point."""
    c = np.asarray(centre, float)
    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)
    off = distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                               np.sin(el)])
    return CameraKey(pos=tuple(c + off), focal=tuple(c))


def edge_on(centre, distance: float, half_height: float,
            along=(1.0, 0.0, 0.0)) -> CameraKey:
    """Orthographic, looking along the flight lines: the cross-section view.

    A perspective camera makes a cross-section lie, because points further from
    the lens shrink and the surface appears to curve.  Parallel projection is
    what turns the ground into a line.
    """
    c = np.asarray(centre, float)
    d = np.asarray(along, float)
    d = d / max(np.linalg.norm(d), 1e-9)
    return CameraKey(pos=tuple(c + distance * d), focal=tuple(c),
                     up=(0.0, 0.0, 1.0), parallel_scale=half_height)
