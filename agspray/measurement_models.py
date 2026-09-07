"""Measurement models shared by every estimator on the ladder.

This module exists so the comparison is about *estimation strategy* and nothing
else.  The EKF and the factor graph both project landmarks with the function
below, both consume GNSS through the same residual, and both use IMU noise
densities that are literally the same numbers.  If the EKF used a slightly
different camera model the study would be comparing implementations.

Conventions, chosen to match GTSAM so the two paths line up exactly:

* World frame is x along-row, y across-row, z up.  Gravity is ``[0, 0, -g]``.
* Body frame: z along thrust (up), x roughly along the heading.  This is what
  ``trajectory._attitude_from_acceleration`` produces.
* Camera is rigidly mounted looking straight down, at the body origin.  Its
  optical axis is ``-z_body``; image x runs along ``+y_body`` and image y along
  ``+x_body``.  That is :data:`R_BC`.
* Pose perturbations are *right* increments in GTSAM's ordering, ``T -> T *
  Exp([omega; v])`` with rotation first.
* The EKF error state is ``[dp, dv, dtheta, dba, dbg]`` with ``R = R_hat *
  Exp(dtheta)``, i.e. a body-frame attitude error, again matching GTSAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.80665])

# Body -> camera.  Columns are the camera axes written in body coordinates.
R_BC = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics plus the image bounds used for visibility."""

    fx: float
    fy: float
    u0: float
    v0: float
    width: int
    height: int
    pixel_sigma: float

    @staticmethod
    def from_cfg(cam_cfg) -> "Camera":
        return Camera(
            fx=cam_cfg.fx,
            fy=cam_cfg.fy,
            u0=0.5 * cam_cfg.width,
            v0=0.5 * cam_cfg.height,
            width=cam_cfg.width,
            height=cam_cfg.height,
            pixel_sigma=cam_cfg.pixel_sigma,
        )

    @property
    def pixel_cov(self) -> np.ndarray:
        return (self.pixel_sigma ** 2) * np.eye(2)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix, batched over any leading dimensions."""
    v = np.asarray(v, dtype=float)
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] = v[..., 1]
    out[..., 1, 0] = v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] = v[..., 0]
    return out


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues exponential for a single rotation vector."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1e-9:
        return np.eye(3) + K + 0.5 * (K @ K)
    return (np.eye(3) + (np.sin(theta) / theta) * K
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K))


def log_so3(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`exp_so3`, single rotation."""
    R = np.asarray(R, dtype=float)
    c = 0.5 * (np.trace(R) - 1.0)
    c = float(np.clip(c, -1.0, 1.0))
    theta = float(np.arccos(c))
    if theta < 1e-9:
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return 0.5 * v
    if abs(np.pi - theta) < 1e-6:
        # Near pi the antisymmetric part vanishes; recover the axis from R + I.
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(axis))
        if axis[k] > 0:
            axis = A[:, k] / axis[k]
        sgn = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(axis, sgn) < 0:
            axis = -axis
        return theta * axis / max(np.linalg.norm(axis), 1e-12)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return (theta / (2.0 * np.sin(theta))) * v


def right_jacobian_so3(w: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3), used for the gyro-bias block."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1e-6:
        return np.eye(3) - 0.5 * K + (1.0 / 6.0) * (K @ K)
    t2 = theta * theta
    return (np.eye(3)
            - ((1.0 - np.cos(theta)) / t2) * K
            + ((theta - np.sin(theta)) / (t2 * theta)) * (K @ K))


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project a nearly-rotation matrix back onto SO(3)."""
    u, _, vt = np.linalg.svd(R)
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------


def body_to_camera(R_wb: np.ndarray, p_wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera pose from body pose.  The camera sits at the body origin."""
    return R_wb @ R_BC, np.asarray(p_wb, dtype=float)


def project(R_wc: np.ndarray, p_wc: np.ndarray, points: np.ndarray, cam: Camera):
    """Project world points into the image.

    Returns ``(uv, pc, valid)`` where ``pc`` is the point in camera coordinates
    and ``valid`` marks points in front of the camera and inside the frame.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    pc = (pts - np.asarray(p_wc, dtype=float)[None, :]) @ R_wc          # (N, 3)
    z = pc[:, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv = np.empty((pts.shape[0], 2))
    uv[:, 0] = cam.fx * pc[:, 0] / safe + cam.u0
    uv[:, 1] = cam.fy * pc[:, 1] / safe + cam.v0
    valid = (z > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) \
        & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    return uv, pc, valid


def d_uv_d_pc(pc: np.ndarray, cam: Camera) -> np.ndarray:
    """(N, 2, 3) derivative of the pixel with respect to the camera-frame point."""
    pc = np.atleast_2d(np.asarray(pc, dtype=float))
    z = pc[:, 2]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    n = pc.shape[0]
    J = np.zeros((n, 2, 3))
    J[:, 0, 0] = cam.fx / z
    J[:, 0, 2] = -cam.fx * pc[:, 0] / (z * z)
    J[:, 1, 1] = cam.fy / z
    J[:, 1, 2] = -cam.fy * pc[:, 1] / (z * z)
    return J


def projection_jacobians(R_wc: np.ndarray, pc: np.ndarray, cam: Camera):
    """Jacobians of the pixel wrt the *camera* pose and the landmark.

    ``H_cam`` is (N, 2, 6) in GTSAM's ``[omega; v]`` right-perturbation order;
    ``H_l`` is (N, 2, 3) in world coordinates.
    """
    J = d_uv_d_pc(pc, cam)
    H_cam = np.zeros((J.shape[0], 2, 6))
    H_cam[:, :, 0:3] = J @ skew(pc)
    H_cam[:, :, 3:6] = -J
    H_l = J @ R_wc.T[None, :, :]
    return H_cam, H_l


def camera_to_body_jacobian(H_cam: np.ndarray) -> np.ndarray:
    """Push a camera-pose Jacobian through the fixed body-camera extrinsic.

    ``T_wc = T_wb * T_bc`` with zero lever arm, so the adjoint of ``T_bc^-1`` is
    block diagonal in ``R_bc^T``.
    """
    Ad = np.zeros((6, 6))
    Ad[0:3, 0:3] = R_BC.T
    Ad[3:6, 3:6] = R_BC.T
    return H_cam @ Ad


def projection_error_state_jacobian(H_body: np.ndarray, R_wb: np.ndarray) -> np.ndarray:
    """Convert a body-pose Jacobian to the EKF error state ``[dp, dtheta]``.

    Returns (N, 2, 6) with the *position* block first, matching the error-state
    layout used in :mod:`agspray.estimators.ekf`.
    """
    out = np.zeros_like(H_body)
    out[:, :, 0:3] = H_body[:, :, 3:6] @ R_wb.T[None, :, :]
    out[:, :, 3:6] = H_body[:, :, 0:3]
    return out


def backproject_to_plane(R_wc: np.ndarray, p_wc: np.ndarray, uv: np.ndarray,
                         cam: Camera, plane_z: float = 0.0) -> np.ndarray:
    """Initialise a landmark by intersecting the pixel ray with the canopy plane.

    A new feature has no depth, and the only prior available in the air is that
    the crop is roughly at ground level.  That is exactly what a real monocular
    front end does over a field, and it puts a real (and honest) error on new
    landmarks, which is part of what the estimators have to sort out.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    rays_c = np.column_stack([
        (uv[:, 0] - cam.u0) / cam.fx,
        (uv[:, 1] - cam.v0) / cam.fy,
        np.ones(uv.shape[0]),
    ])
    rays_w = rays_c @ R_wc.T
    p_wc = np.asarray(p_wc, dtype=float)
    denom = rays_w[:, 2]
    denom = np.where(np.abs(denom) < 1e-6, -1e-6, denom)
    s = (plane_z - p_wc[2]) / denom
    s = np.clip(s, 0.0, 1e4)
    return p_wc[None, :] + s[:, None] * rays_w


# --------------------------------------------------------------------------
# GNSS
# --------------------------------------------------------------------------


def gnss_residual(p_hat: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    return np.asarray(measurement, dtype=float) - np.asarray(p_hat, dtype=float)


def gnss_jacobian(n_state: int) -> np.ndarray:
    """Position rows of the error-state Jacobian."""
    H = np.zeros((3, n_state))
    H[:, 0:3] = np.eye(3)
    return H


# --------------------------------------------------------------------------
# IMU
# --------------------------------------------------------------------------

# Error-state layout.
I_P, I_V, I_TH, I_BA, I_BG = 0, 3, 6, 9, 12
NAV_DIM = 15


@dataclass
class NavState:
    """Nominal navigation state propagated by the strapdown equations."""

    p: np.ndarray
    v: np.ndarray
    R: np.ndarray
    b_a: np.ndarray
    b_g: np.ndarray

    def copy(self) -> "NavState":
        return NavState(self.p.copy(), self.v.copy(), self.R.copy(),
                        self.b_a.copy(), self.b_g.copy())


def imu_propagate(state: NavState, accel: np.ndarray, gyro: np.ndarray,
                  dt: float) -> NavState:
    """One strapdown step.  Mid-point on rotation is not needed at 200 Hz."""
    f = np.asarray(accel, dtype=float) - state.b_a
    w = np.asarray(gyro, dtype=float) - state.b_g
    a_w = state.R @ f + GRAVITY
    p = state.p + state.v * dt + 0.5 * a_w * dt * dt
    v = state.v + a_w * dt
    R = state.R @ exp_so3(w * dt)
    return NavState(p, v, R, state.b_a.copy(), state.b_g.copy())


def imu_error_transition(state: NavState, accel: np.ndarray, gyro: np.ndarray,
                         dt: float, estimate_bias: bool = True):
    """Discrete error-state transition ``F`` for the 15-state model."""
    f = np.asarray(accel, dtype=float) - state.b_a
    w = np.asarray(gyro, dtype=float) - state.b_g
    dR = exp_so3(w * dt)

    F = np.eye(NAV_DIM)
    F[I_P:I_P + 3, I_V:I_V + 3] = dt * np.eye(3)
    F[I_V:I_V + 3, I_TH:I_TH + 3] = -state.R @ skew(f) * dt
    F[I_V:I_V + 3, I_BA:I_BA + 3] = -state.R * dt
    F[I_TH:I_TH + 3, I_TH:I_TH + 3] = dR.T
    if estimate_bias:
        F[I_TH:I_TH + 3, I_BG:I_BG + 3] = -right_jacobian_so3(w * dt) * dt
    else:
        F[I_V:I_V + 3, I_BA:I_BA + 3] = 0.0
    return F


def imu_process_noise(imu_cfg, dt: float, scale: float = 1.0) -> np.ndarray:
    """Discrete process noise from the continuous densities in the config.

    The same four densities feed GTSAM's ``PreintegrationCombinedParams``, so
    the filter and the graph are driven by one noise model.
    """
    sa2 = imu_cfg.accel_noise_density ** 2
    sg2 = imu_cfg.gyro_noise_density ** 2
    sba2 = imu_cfg.accel_bias_rw ** 2
    sbg2 = imu_cfg.gyro_bias_rw ** 2

    Q = np.zeros((NAV_DIM, NAV_DIM))
    I3 = np.eye(3)
    # Position picks up the double integral of the accelerometer white noise.
    Q[I_P:I_P + 3, I_P:I_P + 3] = sa2 * (dt ** 3) / 3.0 * I3
    Q[I_P:I_P + 3, I_V:I_V + 3] = sa2 * (dt ** 2) / 2.0 * I3
    Q[I_V:I_V + 3, I_P:I_P + 3] = sa2 * (dt ** 2) / 2.0 * I3
    Q[I_V:I_V + 3, I_V:I_V + 3] = sa2 * dt * I3
    Q[I_TH:I_TH + 3, I_TH:I_TH + 3] = sg2 * dt * I3
    Q[I_BA:I_BA + 3, I_BA:I_BA + 3] = sba2 * dt * I3
    Q[I_BG:I_BG + 3, I_BG:I_BG + 3] = sbg2 * dt * I3
    return scale * Q


def inject_error(state: NavState, dx: np.ndarray) -> NavState:
    """Apply an error-state correction to the nominal state."""
    p = state.p + dx[I_P:I_P + 3]
    v = state.v + dx[I_V:I_V + 3]
    R = orthonormalize(state.R @ exp_so3(dx[I_TH:I_TH + 3]))
    b_a = state.b_a + dx[I_BA:I_BA + 3]
    b_g = state.b_g + dx[I_BG:I_BG + 3]
    return NavState(p, v, R, b_a, b_g)


def pose_error(R_hat: np.ndarray, p_hat: np.ndarray,
               R_true: np.ndarray, p_true: np.ndarray) -> np.ndarray:
    """Local pose error in GTSAM order ``[omega; v]``, for NEES and ATE."""
    dR = R_hat.T @ np.asarray(R_true, dtype=float)
    omega = log_so3(dR)
    v = R_hat.T @ (np.asarray(p_true, dtype=float) - np.asarray(p_hat, dtype=float))
    return np.concatenate([omega, v])
