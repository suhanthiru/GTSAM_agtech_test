"""Ground-truth flight: a boustrophedon coverage pattern in SE(3).

The path is built analytically so there is no controller in the loop and no
integration drift in the truth.  Legs are straight and constant speed; row
reversals are quintic Hermite segments matched in position, velocity *and*
acceleration at both ends.  C2 continuity matters here because the accelerometer
reads the second derivative directly - a C1 path would inject a step in specific
force at every turn and hand the estimators a fictitious event.

Wind enters as a bounded second-order excursion about the planned path rather
than as an integrated random walk.  A real aircraft is held on station by a
controller, so gusts push it around by a metre or so and it comes back; the
accelerometer honestly measures that motion, and the attitude tilts to produce
it.  The wind is therefore *not* hidden from the IMU.  What the wind actually
breaks is the world: it moves the canopy, and that is handled in
``degradation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .config import Config
from .degradation import WindField

GRAVITY = np.array([0.0, 0.0, -9.80665])


@dataclass
class TrueState:
    """Ground truth sampled on the IMU clock."""

    t: np.ndarray            # (T,)
    position: np.ndarray     # (T, 3)
    velocity: np.ndarray     # (T, 3)
    acceleration: np.ndarray  # (T, 3)
    rotation: np.ndarray     # (T, 3, 3), body -> world
    omega: np.ndarray        # (T, 3), body rates
    is_turn: np.ndarray      # (T,) bool
    leg_id: np.ndarray       # (T,) int, which flight line
    pass_id: np.ndarray      # (T,) int
    spray_gate: np.ndarray   # (T,) bool, geometry says spraying is allowed

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def heading(self) -> np.ndarray:
        """Unit horizontal direction of travel, (T, 3)."""
        h = self.velocity.copy()
        h[:, 2] = 0.0
        n = np.linalg.norm(h, axis=1, keepdims=True)
        return h / np.maximum(n, 1e-9)


# --------------------------------------------------------------------------
# quintic Hermite
# --------------------------------------------------------------------------


def _quintic(u: np.ndarray, T: float, p0, v0, a0, p1, v1, a1):
    """Position, velocity and acceleration of a quintic Hermite segment.

    ``u`` is normalised time in [0, 1]; the boundary conditions are given in
    physical units and scaled internally.
    """
    u = np.asarray(u, dtype=float)
    u2, u3, u4, u5 = u * u, u ** 3, u ** 4, u ** 5

    h0 = 1 - 10 * u3 + 15 * u4 - 6 * u5
    h1 = u - 6 * u3 + 8 * u4 - 3 * u5
    h2 = 0.5 * u2 - 1.5 * u3 + 1.5 * u4 - 0.5 * u5
    h3 = 0.5 * u3 - u4 + 0.5 * u5
    h4 = -4 * u3 + 7 * u4 - 3 * u5
    h5 = 10 * u3 - 15 * u4 + 6 * u5

    d0 = -30 * u2 + 60 * u3 - 30 * u4
    d1 = 1 - 18 * u2 + 32 * u3 - 15 * u4
    d2 = u - 4.5 * u2 + 6 * u3 - 2.5 * u4
    d3 = 1.5 * u2 - 4 * u3 + 2.5 * u4
    d4 = -12 * u2 + 28 * u3 - 15 * u4
    d5 = 30 * u2 - 60 * u3 + 30 * u4

    e0 = -60 * u + 180 * u2 - 120 * u3
    e1 = -36 * u + 96 * u2 - 60 * u3
    e2 = 1 - 9 * u + 18 * u2 - 10 * u3
    e3 = 3 * u - 12 * u2 + 10 * u3
    e4 = -24 * u + 84 * u2 - 60 * u3
    e5 = 60 * u - 180 * u2 + 120 * u3

    T2 = T * T
    pos = h0 * p0 + h1 * T * v0 + h2 * T2 * a0 + h3 * T2 * a1 + h4 * T * v1 + h5 * p1
    vel = (d0 * p0 + d1 * T * v0 + d2 * T2 * a0 + d3 * T2 * a1 + d4 * T * v1 + d5 * p1) / T
    acc = (e0 * p0 + e1 * T * v0 + e2 * T2 * a0 + e3 * T2 * a1 + e4 * T * v1 + e5 * p1) / T2
    return pos, vel, acc


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


@dataclass
class FlightPlan:
    """The commanded pattern: which lines, flown in which order."""

    row_y: np.ndarray        # (n_lines,) lateral position of each line
    x_min: float
    x_max: float
    altitude: float
    speed: float
    turn_time: float
    lead_in: float           # m of straight flight before the first line starts

    @staticmethod
    def boustrophedon(cfg: Config, row_y: np.ndarray | None = None) -> "FlightPlan":
        der = cfg.derived
        if row_y is None:
            row_y = np.arange(cfg.field_.n_rows, dtype=float) * cfg.field_.row_spacing
        return FlightPlan(
            row_y=np.asarray(row_y, dtype=float),
            x_min=0.0,
            x_max=cfg.field_.length_x,
            altitude=cfg.flight.altitude,
            speed=cfg.flight.cruise_speed,
            turn_time=der.turn_time,
            lead_in=cfg.flight.start_pad * cfg.flight.cruise_speed,
        )

    def duration(self) -> float:
        n = len(self.row_y)
        leg = (self.x_max - self.x_min) / self.speed
        return n * leg + (n - 1) * self.turn_time + self.lead_in / self.speed


def _sample_plan(plan: FlightPlan, t: np.ndarray):
    """Evaluate the nominal path at the given times.

    Returns position, velocity, acceleration, is_turn, leg_id.
    """
    n_lines = len(plan.row_y)
    v = plan.speed
    leg_time = (plan.x_max - plan.x_min) / v
    lead_time = plan.lead_in / v

    pos = np.zeros((t.size, 3))
    vel = np.zeros((t.size, 3))
    acc = np.zeros((t.size, 3))
    is_turn = np.zeros(t.size, dtype=bool)
    leg_id = np.zeros(t.size, dtype=int)

    cursor = 0.0
    for i in range(n_lines):
        forward = (i % 2 == 0)
        this_leg = leg_time + (lead_time if i == 0 else 0.0)
        x_start = (plan.x_min - plan.lead_in) if (i == 0 and forward) else (
            plan.x_min if forward else plan.x_max)
        x_end = plan.x_max if forward else plan.x_min

        m = (t >= cursor) & (t < cursor + this_leg)
        if i == n_lines - 1:
            m |= (t >= cursor + this_leg)
        if m.any():
            tau = t[m] - cursor
            s = 1.0 if forward else -1.0
            pos[m, 0] = x_start + s * v * tau
            pos[m, 1] = plan.row_y[i]
            pos[m, 2] = plan.altitude
            vel[m, 0] = s * v
            leg_id[m] = i
        cursor += this_leg

        if i < n_lines - 1:
            m = (t >= cursor) & (t < cursor + plan.turn_time)
            if m.any():
                u = (t[m] - cursor) / plan.turn_time
                s = 1.0 if forward else -1.0
                px, vx, ax = _quintic(u, plan.turn_time, x_end, s * v, 0.0, x_end, -s * v, 0.0)
                py, vy, ay = _quintic(u, plan.turn_time, plan.row_y[i], 0.0, 0.0,
                                      plan.row_y[i + 1], 0.0, 0.0)
                pos[m, 0], pos[m, 1], pos[m, 2] = px, py, plan.altitude
                vel[m, 0], vel[m, 1] = vx, vy
                acc[m, 0], acc[m, 1] = ax, ay
                is_turn[m] = True
                leg_id[m] = i
            cursor += plan.turn_time

    return pos, vel, acc, is_turn, leg_id


# --------------------------------------------------------------------------
# wind response and attitude
# --------------------------------------------------------------------------


def _wind_excursion(cfg: Config, t: np.ndarray, wind: WindField | None):
    """Bounded second-order response of the airframe to gusts.

    Returns displacement, its first and second derivatives.  Only the gust
    component is used: a controller trims out steady wind, so a constant
    crosswind should not show up as a constant position error.
    """
    n = t.size
    d = np.zeros((n, 3))
    dd = np.zeros((n, 3))
    ddd = np.zeros((n, 3))
    if wind is None or wind.severity <= 0.0 or n < 2:
        return d, dd, ddd

    deg = cfg.degradation
    omega = 1.0 / max(deg.airframe_tau, 1e-3)
    zeta = 0.7
    gain = deg.airframe_gain * wind.severity

    gust = wind.sample(t) - wind.mean_speed * wind.direction[None, :]
    dt = np.diff(t, prepend=t[0])

    x = np.zeros(3)
    xd = np.zeros(3)
    for i in range(n):
        forcing = gain * gust[i]
        xdd = forcing - 2.0 * zeta * omega * xd - omega * omega * x
        ddd[i] = xdd
        d[i] = x
        dd[i] = xd
        h = dt[i]
        if h > 0:
            xd = xd + h * xdd
            x = x + h * xd
    return d, dd, ddd


def _attitude_from_acceleration(acc: np.ndarray, heading_xy: np.ndarray) -> np.ndarray:
    """Multirotor attitude implied by the commanded acceleration.

    Body z lines up with thrust, which must supply ``a - g``; the remaining
    freedom is taken up by the heading.  This is the standard differential-
    flatness construction and it means gusts visibly tilt the aircraft.
    """
    thrust = acc - GRAVITY[None, :]
    zb = thrust / np.maximum(np.linalg.norm(thrust, axis=1, keepdims=True), 1e-9)

    xc = np.zeros_like(zb)
    xc[:, 0] = heading_xy[:, 0]
    xc[:, 1] = heading_xy[:, 1]

    yb = np.cross(zb, xc)
    nrm = np.linalg.norm(yb, axis=1, keepdims=True)
    # Degenerate only if the heading is parallel to body z, which cannot happen
    # for a level-ish multirotor, but guard anyway.
    bad = (nrm[:, 0] < 1e-6)
    if bad.any():
        yb[bad] = np.array([0.0, 1.0, 0.0])
        nrm[bad] = 1.0
    yb = yb / nrm
    xb = np.cross(yb, zb)
    return np.stack([xb, yb, zb], axis=-1)


def _body_rates(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Body angular rates from the rotation sequence, central differences."""
    n = R.shape[0]
    omega = np.zeros((n, 3))
    if n < 2:
        return omega
    rel = np.einsum("tji,tjk->tik", R[:-1], R[1:])          # R_k^T R_{k+1}
    dtheta = Rotation.from_matrix(rel).as_rotvec()
    dt = np.diff(t)[:, None]
    rates = dtheta / np.maximum(dt, 1e-12)
    omega[:-1] = rates
    omega[1:-1] = 0.5 * (rates[:-1] + rates[1:])
    omega[-1] = rates[-1]
    return omega


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_truth(cfg: Config, wind: WindField | None = None,
                plans: list[FlightPlan] | None = None,
                t_grid: np.ndarray | None = None) -> TrueState:
    """Ground-truth states on the IMU clock for one or more passes."""
    if plans is None:
        plans = [FlightPlan.boustrophedon(cfg)]

    dt = 1.0 / cfg.imu.rate_hz
    gap = cfg.flight.return_gap_s

    segments = []
    t_offset = 0.0
    for pi, plan in enumerate(plans):
        dur = plan.duration()
        n = int(round(dur / dt))
        t_local = np.arange(n) * dt
        segments.append((pi, plan, t_offset + t_local, t_local))
        t_offset += dur + (gap if pi + 1 < len(plans) else 0.0)

    t_all = np.concatenate([s[2] for s in segments]) if t_grid is None else t_grid

    pos = np.zeros((t_all.size, 3))
    vel = np.zeros((t_all.size, 3))
    acc = np.zeros((t_all.size, 3))
    is_turn = np.zeros(t_all.size, dtype=bool)
    leg_id = np.zeros(t_all.size, dtype=int)
    pass_id = np.zeros(t_all.size, dtype=int)
    gate = np.zeros(t_all.size, dtype=bool)

    idx = 0
    for pi, plan, t_global, t_local in segments:
        n = t_local.size
        sl = slice(idx, idx + n)
        p, v, a, turn, leg = _sample_plan(plan, t_local)
        pos[sl], vel[sl], acc[sl], is_turn[sl], leg_id[sl] = p, v, a, turn, leg
        pass_id[sl] = pi
        inside = (p[:, 0] >= plan.x_min) & (p[:, 0] <= plan.x_max)
        gate[sl] = inside & (~turn if cfg.spray.gate_on_turns else True)
        idx += n

    d, dd, ddd = _wind_excursion(cfg, t_all, wind)
    pos = pos + d
    vel = vel + dd
    acc = acc + ddd

    heading = vel[:, :2].copy()
    nrm = np.linalg.norm(heading, axis=1, keepdims=True)
    heading = heading / np.maximum(nrm, 1e-9)
    R = _attitude_from_acceleration(acc, heading)
    omega = _body_rates(R, t_all)

    return TrueState(t=t_all, position=pos, velocity=vel, acceleration=acc,
                     rotation=R, omega=omega, is_turn=is_turn, leg_id=leg_id,
                     pass_id=pass_id, spray_gate=gate)
