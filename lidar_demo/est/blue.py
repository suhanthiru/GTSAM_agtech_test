"""The drone's own belief about where it was.

A loosely-coupled inertial/GNSS filter: strapdown propagation at 400 Hz with an
error-state correction at each 1 Hz position fix.  This is what an aircraft
actually knows *in flight*, and it is the honest opponent for the smoother.

It is worth being clear about why this is a filter and not something simpler.
A position-only nudge cannot work at all here.  Attitude error leaks gravity
straight into horizontal acceleration -- three degrees of tilt is half a metre
per second squared -- so a loop that corrects position while leaving attitude
alone diverges by hundreds of metres over a survey.  Any real system estimates
tilt and bias from the position innovation, so this one does too, using the
error-state transition and process noise already defined in
``agspray.measurement_models``.  The filter and the factor graph are therefore
driven by the same four noise densities.

What the filter cannot do is the point of the whole piece.  It runs forward
only, so an outage is paid for at the time and never repaid.  And the sensor
mounting angle is not in its state vector, so there is no mechanism by which any
measurement could ever move it: the blue map is built with the nominal
extrinsic, and it would still be built with the nominal extrinsic if the
trajectory were perfect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agspray.measurement_models import (I_BA, I_BG, I_P, I_TH, I_V, NAV_DIM,
                                        NavState, imu_error_transition,
                                        imu_process_noise, imu_propagate,
                                        inject_error)

from ..config import BlueCfg, ImuCfg
from ..io import DenseTraj, Extrinsic
from .strapdown import align_gravity, apply_heading


@dataclass
class BlueResult:
    traj: DenseTraj
    n_fixes_used: int
    n_fixes_dropped: int
    longest_outage_s: float
    cfg: BlueCfg

    def summary(self) -> str:
        return (f"blue: {len(self.traj)} samples, {self.n_fixes_used} GNSS fixes "
                f"applied, {self.n_fixes_dropped} lost, longest outage "
                f"{self.longest_outage_s:.0f} s")


def initial_state(t, accel, gyro, t_gnss, p_gnss, valid, cfg: BlueCfg) -> NavState:
    """Coarse alignment from the first few seconds of the run.

    Position from the first valid fix, roll and pitch from the direction of
    specific force, heading from the direction the aircraft is seen to move.
    Biases start at zero, which is what a system with no calibration history
    assumes.
    """
    ok = np.flatnonzero(valid)
    if ok.size == 0:
        raise ValueError("no valid GNSS fix in the run; cannot initialise")

    dt = float(t[1] - t[0])
    n_static = max(int(cfg.align_static_s / dt), 1)
    R0 = align_gravity(accel[:n_static])

    late = ok[t_gnss[ok] <= t_gnss[ok[0]] + cfg.align_heading_s]
    if late.size >= 2:
        d = p_gnss[late[-1]] - p_gnss[late[0]]
        v0 = d / max(float(t_gnss[late[-1]] - t_gnss[late[0]]), 1e-6)
        R0 = apply_heading(R0, d[:2])
    else:
        v0 = np.zeros(3)

    return NavState(p=p_gnss[ok[0]].copy(), v=v0, R=R0,
                    b_a=np.zeros(3), b_g=np.zeros(3))


def initial_covariance(cfg: BlueCfg, imu: ImuCfg) -> np.ndarray:
    P = np.zeros((NAV_DIM, NAV_DIM))
    P[I_P:I_P + 3, I_P:I_P + 3] = np.diag([1.0, 1.0, 4.0])
    P[I_V:I_V + 3, I_V:I_V + 3] = np.eye(3) * 0.5 ** 2
    P[I_TH:I_TH + 3, I_TH:I_TH + 3] = np.diag(
        [np.radians(2.0) ** 2, np.radians(2.0) ** 2, np.radians(10.0) ** 2])
    P[I_BA:I_BA + 3, I_BA:I_BA + 3] = np.eye(3) * (4.0 * imu.accel_bias_init) ** 2
    P[I_BG:I_BG + 3, I_BG:I_BG + 3] = np.eye(3) * (4.0 * imu.gyro_bias_init) ** 2
    return P


def blue_trajectory(t, accel, gyro, t_gnss, p_gnss, valid, gnss_index,
                    cfg: BlueCfg, imu: ImuCfg, sigma_gnss,
                    extrinsic: Extrinsic | None = None) -> BlueResult:
    """Loosely-coupled inertial/GNSS filter on the IMU clock.

    The covariance is propagated on a decimated clock rather than every IMU
    sample, because a 15x15 triple product at 400 Hz is most of the run time and
    buys nothing.  It cannot be decimated all the way to the GNSS rate, though:
    the attitude block of the transition is a rotation through the interval, and
    over a whole second of a turn that is ninety degrees, which is nowhere near
    the product of the four hundred small rotations it is standing in for.  A
    few tens of hertz keeps the approximation honest and still costs little.
    """
    n = int(t.size)
    p_out = np.empty((n, 3))
    v_out = np.empty((n, 3))
    R_out = np.empty((n, 3, 3))

    state = initial_state(t, accel, gyro, t_gnss, p_gnss, valid, cfg)
    P = initial_covariance(cfg, imu)
    R_meas = np.diag(np.asarray(sigma_gnss, dtype=float) ** 2)
    H = np.zeros((3, NAV_DIM))
    H[:, I_P:I_P + 3] = np.eye(3)

    p_out[0], v_out[0], R_out[0] = state.p, state.v, state.R
    fix_at = {int(i): k for k, i in enumerate(np.asarray(gnss_index, int))}

    used = dropped = 0
    last_cov_i = 0
    last_fix_t = float(t[0])
    longest = 0.0
    stride = max(int(round(imu.rate_hz / cfg.cov_rate_hz)), 1)

    def advance_cov(P, i_from, i_to, state):
        i = i_from
        while i < i_to:
            j = min(i + stride, i_to)
            span = float(t[j] - t[i])
            if span > 0.0:
                F = imu_error_transition(state, accel[i], gyro[i], span)
                P = F @ P @ F.T + imu_process_noise(imu, span)
            i = j
        return P

    for i in range(1, n):
        dt = float(t[i] - t[i - 1])
        state = imu_propagate(state, accel[i - 1], gyro[i - 1], dt)
        p_out[i], v_out[i], R_out[i] = state.p, state.v, state.R

        k = fix_at.get(i)
        if k is None:
            continue
        if not valid[k]:
            dropped += 1
            continue

        P = advance_cov(P, last_cov_i, i, state)
        last_cov_i = i

        y = p_gnss[k] - state.p
        S = H @ P @ H.T + R_meas
        K = np.linalg.solve(S.T, (P @ H.T).T).T
        state = inject_error(state, K @ y)
        A = np.eye(NAV_DIM) - K @ H
        P = A @ P @ A.T + K @ R_meas @ K.T
        P = 0.5 * (P + P.T)

        p_out[i], v_out[i], R_out[i] = state.p, state.v, state.R
        longest = max(longest, float(t[i]) - last_fix_t)
        last_fix_t = float(t[i])
        used += 1

    traj = DenseTraj(t=t, p=p_out, R=R_out, v=v_out, label="blue",
                     extrinsic=extrinsic)
    return BlueResult(traj=traj, n_fixes_used=used, n_fixes_dropped=dropped,
                      longest_outage_s=longest, cfg=cfg)


def trajectory_error(traj: DenseTraj, p_true: np.ndarray,
                     R_true: np.ndarray | None = None) -> dict:
    """Position and attitude error against a reference.  Diagnostics only."""
    d = traj.p - np.asarray(p_true)
    xy = np.linalg.norm(d[:, :2], axis=1)
    out = {
        "xy_rms_m": float(np.sqrt((xy ** 2).mean())),
        "xy_p95_m": float(np.percentile(xy, 95)),
        "xy_max_m": float(xy.max()),
        "z_rms_m": float(np.sqrt((d[:, 2] ** 2).mean())),
        "z_p95_m": float(np.percentile(np.abs(d[:, 2]), 95)),
        "z_max_m": float(np.abs(d[:, 2]).max()),
    }
    if R_true is not None:
        from ..frames import log_so3_batch
        rel = np.einsum("tji,tjk->tik", traj.R, np.asarray(R_true))
        ang = np.degrees(np.linalg.norm(log_so3_batch(rel), axis=1))
        out["att_rms_deg"] = float(np.sqrt((ang ** 2).mean()))
        out["att_max_deg"] = float(ang.max())
    return out
