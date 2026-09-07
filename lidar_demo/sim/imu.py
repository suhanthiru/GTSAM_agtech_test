"""Inertial measurements, and the dead-reckoning check that tunes them.

The synthesiser itself is ``agspray.sensors._imu_stream``: specific force and
body rates with a bias random walk, discretised as ``sigma / sqrt(dt)``.  It
reads only ``cfg.imu.*`` and the truth arrays, so it is reused unchanged through
a tiny shim rather than copied.

On the drift target.  The build spec asks for a consumer MEMS profile with noise
densities multiplied five to ten times, *and* for raw dead-reckoned drift of
1-2 m in XY over the flight.  Those two cannot both be true.  Open-loop
strapdown error grows as a high power of elapsed time -- the gyro bias random
walk term alone goes as ``g sigma_bg T^3.5`` -- so over the ~270 s survey any
MEMS-class part leaks hundreds of metres, and at the literal consumer-times-ten
noise density even a 30 s window leaks about 11 m.

The drift *number* is the one worth keeping, because it is what the viewer sees
as ghosting.  So: industrial-MEMS noise densities, and the target is read over
the GNSS-denied window rather than the whole flight.  :func:`drift_report`
measures it and prints both figures so the choice is on the record.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from agspray.measurement_models import NavState, imu_propagate
from agspray.sensors import _imu_stream
from agspray.trajectory import TrueState

from ..config import ImuCfg


def simulate_imu(cfg: ImuCfg, truth: TrueState, rng: np.random.Generator):
    """Specific force and body rates, plus the true biases for scoring."""
    shim = SimpleNamespace(imu=cfg)
    accel, gyro, b_a, b_g = _imu_stream(shim, truth, rng)
    return accel, gyro, b_a, b_g


def predicted_drift(cfg: ImuCfg, T: float) -> dict[str, float]:
    """Closed-form 1-sigma position error per axis after ``T`` seconds.

    Standard strapdown error propagation with a perfectly known initial state
    and perfectly known biases, so this is a floor, not a forecast.  Terms, in
    order: accelerometer white noise, gyro white noise leaking gravity through a
    tilt error, accelerometer bias random walk, gyro bias random walk.
    """
    g = 9.80665
    return {
        "accel_white": cfg.accel_noise_density * T ** 1.5 / np.sqrt(3.0),
        "gyro_white": g * cfg.gyro_noise_density * T ** 2.5 * np.sqrt(0.05),
        "accel_bias_rw": cfg.accel_bias_rw * T ** 2.5 * np.sqrt(0.05),
        "gyro_bias_rw": g * cfg.gyro_bias_rw * T ** 3.5 / np.sqrt(252.0),
    }


@dataclass
class DriftReport:
    window_s: float
    xy_median: float
    xy_p90: float
    z_median: float
    z_p90: float
    n_windows: int
    predicted_xy: float
    predicted_z: float

    def __str__(self) -> str:
        return (f"dead-reckoned drift over {self.window_s:.0f} s windows "
                f"(n={self.n_windows})\n"
                f"  XY  median {self.xy_median:.2f} m   p90 {self.xy_p90:.2f} m"
                f"   (closed form {self.predicted_xy:.2f} m)\n"
                f"  Z   median {self.z_median:.2f} m   p90 {self.z_p90:.2f} m"
                f"   (closed form {self.predicted_z:.2f} m)")


def dead_reckon(truth: TrueState, accel: np.ndarray, gyro: np.ndarray,
                i0: int, n_steps: int, bias_a: np.ndarray | None = None,
                bias_g: np.ndarray | None = None):
    """Integrate raw IMU forward from truth, returning the position error.

    Started from the true pose, velocity and (optionally) the true biases, so
    what comes out is the sensor's own error and nothing else.
    """
    dt = float(truth.t[1] - truth.t[0])
    state = NavState(p=truth.position[i0].copy(), v=truth.velocity[i0].copy(),
                     R=truth.rotation[i0].copy(),
                     b_a=np.zeros(3) if bias_a is None else bias_a[i0].copy(),
                     b_g=np.zeros(3) if bias_g is None else bias_g[i0].copy())
    i1 = min(i0 + n_steps, len(truth) - 1)
    err = np.zeros((i1 - i0, 3))
    for k in range(i0, i1):
        state = imu_propagate(state, accel[k], gyro[k], dt)
        err[k - i0] = state.p - truth.position[k + 1]
    return err


def drift_report(cfg: ImuCfg, truth: TrueState, accel: np.ndarray, gyro: np.ndarray,
                 bias_a: np.ndarray, bias_g: np.ndarray,
                 window_s: float | None = None, stride_s: float = 10.0) -> DriftReport:
    """Measure drift over sliding windows and compare with the closed form."""
    window_s = float(cfg.drift_window_s if window_s is None else window_s)
    dt = float(truth.t[1] - truth.t[0])
    n_steps = int(window_s / dt)
    stride = int(stride_s / dt)

    xy, z = [], []
    for i0 in range(0, len(truth) - n_steps - 1, stride):
        err = dead_reckon(truth, accel, gyro, i0, n_steps, bias_a, bias_g)
        xy.append(float(np.linalg.norm(err[-1, :2])))
        z.append(float(abs(err[-1, 2])))
    xy = np.array(xy)
    z = np.array(z)

    pred = predicted_drift(cfg, window_s)
    per_axis = np.sqrt(sum(v ** 2 for v in pred.values()))
    pred_z = np.sqrt(pred["accel_white"] ** 2 + pred["accel_bias_rw"] ** 2)

    return DriftReport(window_s=window_s,
                       xy_median=float(np.median(xy)), xy_p90=float(np.percentile(xy, 90)),
                       z_median=float(np.median(z)), z_p90=float(np.percentile(z, 90)),
                       n_windows=int(xy.size),
                       predicted_xy=float(per_axis * np.sqrt(2.0)),
                       predicted_z=float(pred_z))
