"""Undistorting a sweep, which has to happen before anything else touches it.

A sweep takes 100 ms and the aircraft covers half a metre in that time, so every
sweep arrives smeared: the points were measured from a moving sensor and stored
as if they had all been taken at once.  Registering smeared clouds puts a blur
floor on the map that no amount of optimisation afterwards will lift, because
the error is already inside the measurement the optimiser is fitting.

Undoing it needs only the *relative* motion across 100 ms, which the IMU gives
accurately even when its absolute solution has drifted metres away.

One judgement call is worth stating.  Deskewing needs the extrinsic in order to
rotate the body motion into sensor coordinates, and the extrinsic is exactly
what is not known yet.  Using the nominal one is fine here: a 2.2 degree error
applied to at most 0.8 m of intra-sweep motion misplaces a point by under 4 cm,
which is well inside the 20 cm voxel that registration downsamples to.  It would
not be fine for the map, so :mod:`lidar_demo.map.project` uses the recovered
extrinsic when it places points in the world.
"""

from __future__ import annotations

import numpy as np

from ..frames import interp_poses
from ..io import DenseTraj, Extrinsic, Sweep


def deskew_to_sensor(sweep: Sweep, traj: DenseTraj, E: Extrinsic,
                     t_ref: float | None = None) -> np.ndarray:
    """Move every point into the sensor frame at one reference instant.

    ``p_{S0} = E^-1 (T_WB(t_ref)^-1 T_WB(t_p)) E p_S``: take the point out of the
    sensor frame at its own time into the body, into the world, back into the
    body at the reference time, and back into the sensor.  Only the relative
    body motion survives, so a trajectory that is metres wrong in absolute terms
    still deskews correctly.
    """
    if len(sweep) == 0:
        return sweep.xyz.astype(np.float64)

    t_ref = sweep.t_start if t_ref is None else float(t_ref)
    t_pts = sweep.t_abs

    R_t, p_t = traj.at(t_pts)
    R_ref, p_ref = traj.at(np.array([t_ref]))
    R_ref, p_ref = R_ref[0], p_ref[0]

    # T_rel = T_WB(t_ref)^-1 T_WB(t_p)
    R_rel = np.einsum("ji,tjk->tik", R_ref, R_t)
    p_rel = np.einsum("ji,tj->ti", R_ref, p_t - p_ref)

    p_B = sweep.xyz.astype(np.float64) @ E.R.T + E.t
    p_B_ref = np.einsum("tij,tj->ti", R_rel, p_B) + p_rel
    return (p_B_ref - E.t) @ E.R


def deskew_to_world(sweep: Sweep, traj: DenseTraj, E: Extrinsic) -> np.ndarray:
    """Place every point directly in the world at its own timestamp.

    This is deskewing and mapping in one step, and it is what the map products
    use: there is no reference instant to pick and no intermediate frame, so
    nothing is approximated twice.
    """
    if len(sweep) == 0:
        return np.zeros((0, 3))
    R_t, p_t = traj.at(sweep.t_abs)
    p_B = sweep.xyz.astype(np.float64) @ E.R.T + E.t
    return np.einsum("tij,tj->ti", R_t, p_B) + p_t


def sweep_motion(sweep: Sweep, traj: DenseTraj) -> float:
    """How far the body moved during the sweep, in metres.

    Reported by the deskew test and by the run log, because a sweep with no
    motion in it is a sweep whose deskewing cannot be verified.
    """
    if len(sweep) == 0:
        return 0.0
    t = sweep.t_abs
    _, p = traj.at(np.array([t.min(), t.max()]))
    return float(np.linalg.norm(p[1] - p[0]))
