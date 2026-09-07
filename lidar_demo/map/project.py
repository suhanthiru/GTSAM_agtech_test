"""Turning stored returns into a world point cloud.

One function, and the whole demo rests on it being able to run twice.  The
sweeps on disk are raw sensor-frame measurements, so the map is not a thing that
was recorded; it is a thing that is *computed*, from a trajectory and a mounting
angle.  Hand it the aircraft's own belief and the nominal mount and it produces
the blue map.  Hand it the graph's answer and the recovered mount and it
produces the green one.  Same returns, both times.

Each point is placed using the pose at its own timestamp, which deskews it and
maps it in a single step.  There is no intermediate reference frame and so
nothing is approximated twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io import DenseTraj, Extrinsic, RunReader


@dataclass
class Cloud:
    xyz: np.ndarray          # (N, 3) float32, world ENU
    intensity: np.ndarray    # (N,) float32
    sweep: np.ndarray        # (N,) int32, which sweep each point came from
    label: str = ""

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


def project_run(reader: RunReader, traj: DenseTraj, E: Extrinsic,
                stride: int = 1, sweep_step: int = 1,
                progress=None) -> Cloud:
    """Project every stored sweep into the world.

    ``stride`` decimates points within a sweep and ``sweep_step`` skips whole
    sweeps.  Both keep the point ordering deterministic, which matters: the blue
    and green clouds have to come out with the same points in the same order so
    the morph can interpolate them one to one.
    """
    xyz, inten, sid = [], [], []
    n = reader.n_sweeps
    for k in range(0, n, sweep_step):
        sw = reader.sweep(k)
        if len(sw) == 0:
            continue
        pts = sw.xyz[::stride]
        if pts.shape[0] == 0:
            continue
        t_pts = sw.t_abs[::stride]
        R_t, p_t = traj.at(t_pts)
        p_B = pts.astype(np.float64) @ E.R.T + E.t
        xyz.append((np.einsum("tij,tj->ti", R_t, p_B) + p_t).astype(np.float32))
        inten.append(sw.intensity[::stride])
        sid.append(np.full(pts.shape[0], k, dtype=np.int32))
        if progress is not None and (k // max(sweep_step, 1)) % 400 == 0:
            progress(k, n)

    if not xyz:
        return Cloud(np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                     np.zeros(0, np.int32), traj.label)
    return Cloud(np.concatenate(xyz), np.concatenate(inten),
                 np.concatenate(sid), traj.label)


def pass_of_point(cloud: Cloud, reader: RunReader, traj: DenseTraj) -> np.ndarray:
    """Which flight line each point came from, derived from the trajectory.

    Used only for colouring the close-up, where seeing two passes disagree is
    the whole point of the shot.  Taken from the estimated trajectory's heading,
    not from the truth.
    """
    period = reader.sweep_period
    t = cloud.sweep.astype(np.float64) * period
    R, _ = traj.at(t)
    h = R[:, :, 0]
    ang = np.arctan2(h[:, 1], h[:, 0])
    # Two groups is all the shot needs: outbound and return.
    return (np.cos(ang) < 0).astype(np.int8)


def crop(cloud: Cloud, x_range=None, y_range=None, z_range=None) -> Cloud:
    """Cut a cloud down to a box, keeping the fields aligned."""
    m = np.ones(len(cloud), dtype=bool)
    for axis, rng in enumerate((x_range, y_range, z_range)):
        if rng is None:
            continue
        m &= (cloud.xyz[:, axis] >= rng[0]) & (cloud.xyz[:, axis] <= rng[1])
    return Cloud(cloud.xyz[m], cloud.intensity[m], cloud.sweep[m], cloud.label)


def band(cloud: Cloud, x0: float, half_width: float) -> Cloud:
    """A slab perpendicular to the flight lines, for the cross-section."""
    return crop(cloud, x_range=(x0 - half_width, x0 + half_width))
