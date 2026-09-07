"""Inertial propagation, vectorised, plus the bridge from keyframes to a dense path.

Two jobs.  Forward propagation turns a raw IMU stream into a trajectory, which
is what the blue reconstruction is built on and what supplies every initial
guess for registration.  The bridge goes the other way: after the graph has
solved for a pose, velocity and bias at each keyframe, the map still has to be
projected at every point's own timestamp, so those keyframe states have to be
expanded back onto the 400 Hz clock.

The bridge integrates each keyframe interval from its own optimised start state
and then distributes the small mismatch at the far end linearly across the
interval.  That is not a filter and it is not pretending to be one: it is the
cheapest way to get a continuous dense path that agrees with the graph at every
node, and the mismatch it has to absorb is millimetres because the graph was
fitted to the same preintegrated measurements.
"""

from __future__ import annotations

import numpy as np

from agspray.measurement_models import GRAVITY, NavState, imu_propagate

from ..frames import exp_so3_batch, log_so3, orthonormalize
from ..io import DenseTraj


def propagate(t: np.ndarray, accel: np.ndarray, gyro: np.ndarray,
              state: NavState, i0: int, i1: int):
    """Strapdown from ``i0`` to ``i1`` inclusive, returning the sampled path."""
    n = i1 - i0 + 1
    p = np.empty((n, 3))
    v = np.empty((n, 3))
    R = np.empty((n, 3, 3))
    p[0], v[0], R[0] = state.p, state.v, state.R

    cur = state.copy()
    for k in range(1, n):
        dt = float(t[i0 + k] - t[i0 + k - 1])
        cur = imu_propagate(cur, accel[i0 + k - 1], gyro[i0 + k - 1], dt)
        p[k], v[k], R[k] = cur.p, cur.v, cur.R
    return p, v, R, cur


def integrate_interval(t, accel, gyro, i0: int, i1: int, p0, v0, R0, b_a, b_g):
    """Same as :func:`propagate` but taking the state as loose arrays."""
    st = NavState(p=np.asarray(p0, float).copy(), v=np.asarray(v0, float).copy(),
                  R=np.asarray(R0, float).copy(),
                  b_a=np.asarray(b_a, float).copy(),
                  b_g=np.asarray(b_g, float).copy())
    return propagate(t, accel, gyro, st, i0, i1)


def imu_bridge(t: np.ndarray, accel: np.ndarray, gyro: np.ndarray,
               kf_idx: np.ndarray, poses_R, poses_p, vels, biases,
               label: str = "green", extrinsic=None) -> DenseTraj:
    """Expand optimised keyframe states onto the full IMU clock.

    ``poses_R``/``poses_p`` are per-keyframe body-to-world rotations and
    positions, ``vels`` velocities, ``biases`` ``(accel, gyro)`` pairs.  Between
    consecutive keyframes the IMU is integrated from the earlier optimised
    state; whatever gap remains at the later keyframe is spread across the
    interval so the dense path lands exactly on every optimised node.
    """
    kf_idx = np.asarray(kf_idx, dtype=int)
    n = int(t.size)
    p_out = np.empty((n, 3))
    R_out = np.empty((n, 3, 3))
    v_out = np.empty((n, 3))

    for k in range(kf_idx.size - 1):
        i0, i1 = int(kf_idx[k]), int(kf_idx[k + 1])
        b_a, b_g = biases[k]
        p, v, R, _ = integrate_interval(t, accel, gyro, i0, i1,
                                        poses_p[k], vels[k], poses_R[k], b_a, b_g)
        u = np.linspace(0.0, 1.0, i1 - i0 + 1)[:, None]

        dp = np.asarray(poses_p[k + 1]) - p[-1]
        p = p + u * dp
        v = v + u * (np.asarray(vels[k + 1]) - v[-1])

        dR = log_so3(orthonormalize(R[-1]).T @ orthonormalize(np.asarray(poses_R[k + 1])))
        R = np.einsum("tij,tjk->tik", R, exp_so3_batch(u * dR[None, :]))

        p_out[i0:i1 + 1] = p
        v_out[i0:i1 + 1] = v
        R_out[i0:i1 + 1] = R

    # The tails outside the keyframe span.  The head is at most a sweep long,
    # before the aircraft has moved the metre that earns a first keyframe, so it
    # holds the first optimised pose rather than integrating backwards for it.
    # The tail can be longer and is integrated forward properly.
    first, last = int(kf_idx[0]), int(kf_idx[-1])
    if first > 0:
        p_out[:first] = poses_p[0]
        v_out[:first] = vels[0]
        R_out[:first] = poses_R[0]
    if last < n - 1:
        b_a, b_g = biases[-1]
        p, v, R, _ = integrate_interval(t, accel, gyro, last, n - 1,
                                        poses_p[-1], vels[-1], poses_R[-1], b_a, b_g)
        p_out[last:] = p
        v_out[last:] = v
        R_out[last:] = R

    return DenseTraj(t=t, p=p_out, R=R_out, v=v_out, label=label,
                     extrinsic=extrinsic,
                     kf_t=t[kf_idx], kf_idx=kf_idx)


def align_gravity(accel: np.ndarray) -> np.ndarray:
    """Roll and pitch from a stretch of specific force.

    At rest, or in level flight, the accelerometer reads the reaction to
    gravity: body ``z`` maps to world ``z``.  That fixes two of three attitude
    angles; heading has to come from motion.
    """
    f = np.asarray(accel, dtype=float).mean(axis=0)
    zb = f / max(np.linalg.norm(f), 1e-9)          # body z expressed in world
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(zb @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    yb = np.cross(zb, ref)
    yb /= max(np.linalg.norm(yb), 1e-9)
    xb = np.cross(yb, zb)
    return orthonormalize(np.stack([xb, yb, zb], axis=-1))


def apply_heading(R: np.ndarray, heading_xy: np.ndarray) -> np.ndarray:
    """Rotate an attitude about world ``z`` so body ``x`` points along a heading."""
    h = np.asarray(heading_xy, dtype=float)[:2]
    n = float(np.linalg.norm(h))
    if n < 1e-9:
        return R
    want = np.arctan2(h[1], h[0])
    have = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(want - have), np.sin(want - have)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return orthonormalize(Rz @ R)


__all__ = ["propagate", "integrate_interval", "imu_bridge", "align_gravity",
           "apply_heading", "GRAVITY"]
