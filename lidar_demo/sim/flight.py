"""The survey flight.

A list of straight legs joined by quintic Hermite transitions, evaluated
analytically at any time.  There is no controller in the loop and no integration
in the truth, so the ground-truth pose at an arbitrary instant is exact.  That
matters more here than in a normal simulator: every LiDAR column is cast from
the true pose at its own sub-sweep timestamp, and a truth built by integration
would smear that.

Two properties of the pattern are load-bearing.

**Adjacent passes fly opposite directions.**  Without that, the same mounting
error rotates every pass the same way, all the swaths agree with each other, and
the boresight is unobservable no matter how good the optimiser is.  The
reversals break the degeneracy: the survey pattern is accidentally a calibration
pattern.  :func:`plan_survey` enforces the alternation and the tests assert it.

**One pass runs along the treeline and one closes across the lines.**  The first
puts a GNSS-denied stretch somewhere the graph has to lean on scan matching; the
second is a cheap, very strong global constraint.

Reused from :mod:`agspray.trajectory`: the quintic segment, the
differential-flatness attitude, the body-rate differencing and the
:class:`~agspray.trajectory.TrueState` container.  Not reused: ``FlightPlan``
and ``build_truth``, which hard-wire legs along ``x`` with a zero-extent U-turn,
cannot express a perpendicular crossing, and need a full spray ``Config``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agspray.trajectory import (_attitude_from_acceleration, _body_rates,
                                _quintic, TrueState)

from ..config import FlightCfg, SceneCfg


@dataclass
class Leg:
    """One straight, constant-speed line."""

    p0: np.ndarray           # (2,)
    p1: np.ndarray           # (2,)
    name: str
    pass_id: int

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def direction(self) -> np.ndarray:
        d = self.p1 - self.p0
        return d / max(np.linalg.norm(d), 1e-9)


@dataclass
class LegPath:
    """Legs plus the transitions between them, sampled analytically."""

    legs: list[Leg]
    transition_times: list[float]
    altitude: float
    speed: float
    gust_amp: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gust_periods: tuple[float, float, float] = (6.5, 11.0, 17.0)
    gust_phase: tuple[float, float, float] = (0.4, 1.9, 3.3)

    # ---- timing ----

    def _schedule(self):
        """Start time of each leg and of each transition."""
        t = 0.0
        leg_start, trans_start = [], []
        for i, leg in enumerate(self.legs):
            leg_start.append(t)
            t += leg.length / self.speed
            if i < len(self.legs) - 1:
                trans_start.append(t)
                t += self.transition_times[i]
        return np.array(leg_start), np.array(trans_start), t

    @property
    def duration(self) -> float:
        return self._schedule()[2]

    # ---- sampling ----

    def sample(self, t: np.ndarray):
        """Position, velocity, acceleration and labels at arbitrary times.

        Returns ``(pos, vel, acc, is_turn, leg_id, pass_id)``.  Gusts are added
        on top as an analytic multi-sine, so their derivatives are exact and the
        accelerometer honestly measures the wobble instead of being handed a
        finite-difference artefact.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        leg_start, trans_start, total = self._schedule()

        pos = np.zeros((t.size, 3))
        vel = np.zeros((t.size, 3))
        acc = np.zeros((t.size, 3))
        is_turn = np.zeros(t.size, dtype=bool)
        leg_id = np.zeros(t.size, dtype=int)
        pass_id = np.zeros(t.size, dtype=int)

        for i, leg in enumerate(self.legs):
            t0 = leg_start[i]
            dur = leg.length / self.speed
            m = (t >= t0) & (t < t0 + dur)
            if i == 0:
                m |= t < t0
            if i == len(self.legs) - 1:
                m |= t >= t0 + dur
            if m.any():
                tau = np.clip(t[m] - t0, 0.0, dur)[:, None]
                d = leg.direction
                pos[m, :2] = leg.p0 + self.speed * tau * d
                vel[m, :2] = self.speed * d
                leg_id[m] = i
                pass_id[m] = leg.pass_id

        for i, t0 in enumerate(trans_start):
            T = self.transition_times[i]
            m = (t >= t0) & (t < t0 + T)
            if not m.any():
                continue
            u = (t[m] - t0) / T
            a, b = self.legs[i], self.legs[i + 1]
            va = self.speed * a.direction
            vb = self.speed * b.direction
            for ax in range(2):
                p, v, ac = _quintic(u, T, a.p1[ax], va[ax], 0.0, b.p0[ax], vb[ax], 0.0)
                pos[m, ax], vel[m, ax], acc[m, ax] = p, v, ac
            is_turn[m] = True
            leg_id[m] = i
            pass_id[m] = a.pass_id

        pos[:, 2] = self.altitude
        g, gd, gdd = self._gust(t)
        return pos + g, vel + gd, acc + gdd, is_turn, leg_id, pass_id

    def _gust(self, t: np.ndarray):
        """Deterministic airframe wobble.

        Not decoration: a perfectly level trajectory makes some components of
        the boresight harder to see, because the sensor never presents the scene
        from a second attitude.  A degree or two of wander is enough.
        """
        g = np.zeros((t.size, 3))
        gd = np.zeros((t.size, 3))
        gdd = np.zeros((t.size, 3))
        for ax in range(3):
            A = self.gust_amp[ax]
            if A == 0.0:
                continue
            w = 2.0 * np.pi / self.gust_periods[ax]
            ph = self.gust_phase[ax]
            g[:, ax] = A * np.sin(w * t + ph)
            gd[:, ax] = A * w * np.cos(w * t + ph)
            gdd[:, ax] = -A * w * w * np.sin(w * t + ph)
        return g, gd, gdd

    def pose(self, t: np.ndarray):
        """``(R_WB, p_WB)`` at arbitrary times, body FLU."""
        pos, vel, acc, _, _, _ = self.sample(t)
        heading = vel[:, :2].copy()
        n = np.linalg.norm(heading, axis=1, keepdims=True)
        heading = heading / np.maximum(n, 1e-9)
        R = _attitude_from_acceleration(acc, heading)
        return R, pos


# ---------------------------------------------------------------------------
# the pattern
# ---------------------------------------------------------------------------


def plan_survey(cfg: FlightCfg, scene: SceneCfg) -> LegPath:
    """Boustrophedon with alternating headings, closed by a crossing pass."""
    legs: list[Leg] = []
    x0, x1 = 0.0, scene.survey_x

    # Flight lines are laid from the far edge inward, so the treeline pass (the
    # GNSS-denied one) lands late in the flight.  That leaves a clean stretch at
    # the start for initialisation and a clean stretch afterwards.
    ys = [scene.survey_y - i * cfg.spacing for i in range(cfg.n_passes)]

    for i, y in enumerate(ys):
        forward = (i % 2 == 0)
        a = np.array([x0 - (cfg.lead_in if i == 0 else 0.0), y]) if forward else np.array([x1, y])
        b = np.array([x1, y]) if forward else np.array([x0, y])
        legs.append(Leg(p0=a, p1=b, name=f"pass{i}", pass_id=i))

    # The closing pass: perpendicular to every line, running past both ends of
    # the survey so it constrains the whole field rather than just the middle.
    cx = cfg.crossing_x
    legs.append(Leg(p0=np.array([cx, -cfg.crossing_pad]),
                    p1=np.array([cx, scene.survey_y + cfg.crossing_pad]),
                    name="crossing", pass_id=cfg.n_passes))

    # Transition durations: a fixed time for the row reversals, and a longer,
    # distance-scaled one for the swing onto the crossing line.
    trans = []
    for i in range(len(legs) - 1):
        if i == len(legs) - 2:
            gap = float(np.linalg.norm(legs[i + 1].p0 - legs[i].p1))
            trans.append(max(1.5 * gap / cfg.speed, cfg.turn_time))
        else:
            trans.append(cfg.turn_time)

    return LegPath(legs=legs, transition_times=trans, altitude=cfg.altitude,
                   speed=cfg.speed, gust_amp=tuple(cfg.gust_amp),
                   gust_periods=tuple(cfg.gust_periods))


def truth_on_clock(path: LegPath, rate_hz: float) -> TrueState:
    """Ground truth sampled on the IMU clock."""
    dt = 1.0 / rate_hz
    n = int(np.floor(path.duration / dt))
    t = np.arange(n) * dt
    pos, vel, acc, is_turn, leg_id, pass_id = path.sample(t)

    heading = vel[:, :2].copy()
    nrm = np.linalg.norm(heading, axis=1, keepdims=True)
    heading = heading / np.maximum(nrm, 1e-9)
    R = _attitude_from_acceleration(acc, heading)
    omega = _body_rates(R, t)

    return TrueState(t=t, position=pos, velocity=vel, acceleration=acc,
                     rotation=R, omega=omega, is_turn=is_turn, leg_id=leg_id,
                     pass_id=pass_id, spray_gate=np.zeros(n, dtype=bool))


def pass_headings(path: LegPath) -> np.ndarray:
    """Unit heading of each numbered pass, for the alternation check."""
    seen: dict[int, np.ndarray] = {}
    for leg in path.legs:
        seen.setdefault(leg.pass_id, leg.direction)
    return np.stack([seen[k] for k in sorted(seen)])
