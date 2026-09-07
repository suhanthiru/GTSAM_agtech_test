"""Frame conventions for the LiDAR demo.

This module is the single source of truth; everything else imports from here and
the run manifest echoes it verbatim.

World ``W``
    Local ENU, right handed, ``z`` up.  Origin at the survey south-west corner.
    Gravity is ``[0, 0, -9.80665]``.

Body ``B``
    FLU: ``x`` forward, ``y`` left, ``z`` up.  This is exactly what
    ``agspray.trajectory._attitude_from_acceleration`` produces (body ``z`` along
    thrust, ``x`` along heading), so ground-truth attitude is reused unchanged.
    Aerospace FRD readers: ``R_FLU_to_FRD = diag(1, -1, -1)``.

Sensor ``S``
    LiDAR frame.  ``z`` is the spin axis, ``x`` points at azimuth zero, ``y``
    left.  A beam at azimuth ``az`` and elevation ``el`` points along
    ``[cos el cos az, cos el sin az, sin el]``; azimuth increases counter
    clockwise about ``+z``.

Extrinsic ``T_BS``
    ``p_B = R_BS p_S + t_BS``, with ``R_BS = Rz(yaw) Ry(pitch) Rx(roll)`` and
    angles in degrees.  Because the body frame is FLU, a *positive* pitch tilts
    the sensor forward and down: ``Ry(+35 deg)`` sends sensor ``+x`` to
    ``(cos 35, 0, -sin 35)``.  That is the opposite sign to the FRD aerospace
    habit, so the manifest says it out loud and no consumer has to guess.
"""

from __future__ import annotations

import numpy as np

GRAVITY_W = np.array([0.0, 0.0, -9.80665])
FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------


def Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_zyx_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """``Rz(yaw) Ry(pitch) Rx(roll)`` from degrees."""
    r, p, y = np.radians([float(roll_deg), float(pitch_deg), float(yaw_deg)])
    return Rz(y) @ Ry(p) @ Rx(r)


def R_to_rpy_zyx(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rpy_zyx_to_R`, in degrees, ordered ``(roll, pitch, yaw)``.

    Gimbal lock at ``|pitch| = 90 deg`` is resolved by putting the whole rotation
    into yaw.  The demo never goes near it.
    """
    R = np.asarray(R, dtype=float)
    sp = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = np.arcsin(sp)
    if abs(sp) > 1.0 - 1e-9:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues, safe as ``|w| -> 0``."""
    w = np.asarray(w, dtype=float)
    th = float(np.linalg.norm(w))
    K = skew(w)
    if th < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + (np.sin(th) / th) * K + ((1.0 - np.cos(th)) / th ** 2) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation vector of ``R``, safe near identity and near pi."""
    R = np.asarray(R, dtype=float)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = float(np.arccos(c))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * v
    if np.pi - th < 1e-6:
        # Near pi the skew part vanishes; read the axis off R + I, whose columns
        # are all parallel to it, and take the sign from what is left of v.
        A = R + np.eye(3)
        j = int(np.argmax(np.linalg.norm(A, axis=0)))
        axis = A[:, j] / np.linalg.norm(A[:, j])
        if float(axis @ v) < 0.0:
            axis = -axis
        return axis * th
    return (th / (2.0 * np.sin(th))) * v


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix; numerical hygiene after long chains of products."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0.0:
        U = U.copy()
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def angle_between_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """``|Log(R1^T R2)|`` in degrees.

    This is the gate-5 metric.  It is free of any Euler convention, so it means
    the same thing no matter how the two rotations were parameterised.
    """
    rel = orthonormalize(np.asarray(R1, dtype=float)).T @ orthonormalize(np.asarray(R2, dtype=float))
    return float(np.degrees(np.linalg.norm(log_so3(rel))))


# ---------------------------------------------------------------------------
# SE(3) as (R, t) pairs
# ---------------------------------------------------------------------------


def compose(R1, t1, R2, t2):
    """``T1 * T2``."""
    R1 = np.asarray(R1, dtype=float)
    return R1 @ np.asarray(R2, dtype=float), R1 @ np.asarray(t2, dtype=float) + np.asarray(t1, dtype=float)


def invert(R, t):
    """``T^-1``."""
    Rt = np.asarray(R, dtype=float).T
    return Rt, -Rt @ np.asarray(t, dtype=float)


def transform_points(R, t, pts) -> np.ndarray:
    """Apply one pose to ``(N, 3)`` points."""
    return np.asarray(pts, dtype=float) @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float)


def to_matrix(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def from_matrix(T):
    T = np.asarray(T, dtype=float)
    return T[:3, :3].copy(), T[:3, 3].copy()


# ---------------------------------------------------------------------------
# beams
# ---------------------------------------------------------------------------


def beam_dirs(az_rad: np.ndarray, el_rad: np.ndarray) -> np.ndarray:
    """Unit beam directions in the sensor frame, shaped ``(n_el, n_az, 3)``."""
    az = np.asarray(az_rad, dtype=float)[None, :]
    el = np.asarray(el_rad, dtype=float)[:, None]
    ce, se = np.cos(el), np.sin(el)
    x = ce * np.cos(az)
    y = ce * np.sin(az)
    z = np.broadcast_to(se, x.shape)
    return np.stack([x, y, z], axis=-1)


# ---------------------------------------------------------------------------
# batched SO(3), used by the pose interpolator
# ---------------------------------------------------------------------------


def exp_so3_batch(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    th = np.linalg.norm(w, axis=1)
    K = np.zeros((w.shape[0], 3, 3))
    K[:, 0, 1] = -w[:, 2]
    K[:, 0, 2] = w[:, 1]
    K[:, 1, 0] = w[:, 2]
    K[:, 1, 2] = -w[:, 0]
    K[:, 2, 0] = -w[:, 1]
    K[:, 2, 1] = w[:, 0]
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1.0 - th ** 2 / 6.0, np.sin(ths) / ths)
    b = np.where(small, 0.5 - th ** 2 / 24.0, (1.0 - np.cos(ths)) / ths ** 2)
    return np.eye(3) + a[:, None, None] * K + b[:, None, None] * (K @ K)


def log_so3_batch(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(c)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2],
                  R[:, 0, 2] - R[:, 2, 0],
                  R[:, 1, 0] - R[:, 0, 1]], axis=1)
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    scale = np.where(small, 0.5 + th ** 2 / 12.0, th / (2.0 * np.sin(ths)))
    return v * scale[:, None]


# ---------------------------------------------------------------------------
# pose interpolation on a dense clock
# ---------------------------------------------------------------------------


def interp_poses(t_ref: np.ndarray, p_ref: np.ndarray, R_ref: np.ndarray,
                 t_query: np.ndarray):
    """Interpolate a dense pose sequence at arbitrary times.

    Position is linear.  Rotation walks along the manifold,
    ``R_i exp(u Log(R_i^T R_{i+1}))``, which is exactly SLERP and is cheap for
    the small increments a 400 Hz clock produces.  Queries outside ``t_ref``
    clamp to the end poses rather than extrapolating, which keeps sweep edges
    well defined at the very start and end of a run.
    """
    t_ref = np.asarray(t_ref, dtype=float)
    tq = np.atleast_1d(np.asarray(t_query, dtype=float))
    n = int(t_ref.size)
    if n < 2:
        R = np.broadcast_to(R_ref[0], (tq.size, 3, 3)).copy()
        p = np.broadcast_to(p_ref[0], (tq.size, 3)).copy()
        return R, p

    idx = np.clip(np.searchsorted(t_ref, tq, side="right") - 1, 0, n - 2)
    t0 = t_ref[idx]
    dt = t_ref[idx + 1] - t0
    u = np.clip((tq - t0) / np.where(dt > 0.0, dt, 1.0), 0.0, 1.0)

    p = p_ref[idx] + u[:, None] * (p_ref[idx + 1] - p_ref[idx])

    R0 = R_ref[idx]
    rel = np.einsum("tji,tjk->tik", R0, R_ref[idx + 1])
    w = log_so3_batch(rel) * u[:, None]
    R = np.einsum("tij,tjk->tik", R0, exp_so3_batch(w))
    return R, p
