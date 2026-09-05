"""Degradation models, all of them tied to the flight pattern.

The design rule from the brief: degradation must correlate with the coverage
pattern rather than be i.i.d.  White noise is easy for every estimator and
separates nothing.  What separates estimators is disturbance that recurs on
every row, or that hits the motion model and the visual measurements at the
same instant.

One wind realisation drives four of the axes:

* the airframe is pushed around by it (second-order response, so the excursion
  is bounded and correlated rather than a random walk),
* the canopy sways with it, which is the static-world violation every method
  here rests on,
* it decides whether spray drift blows back under the camera,
* and it sets the direction the drift plume travels.

The remaining two axes key off geometry instead: GNSS multipath recurs in a
band along the treeline, so it fires on every turn at that edge, and sun
washout only hits the leg heading into the sun, never the reciprocal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config
from .world import World


@dataclass
class WindField:
    """Ornstein-Uhlenbeck gusts about a prevailing wind.

    Sampled once onto a fixed time grid so that every estimator in a paired
    trial sees the identical realisation, and so Tier 2 can replay it.
    """

    t: np.ndarray          # (T,) time grid
    velocity: np.ndarray   # (T, 3) wind velocity in the world frame, m/s
    direction: np.ndarray  # (3,) unit vector, prevailing direction of travel
    mean_speed: float
    severity: float

    def sample(self, t_query: np.ndarray) -> np.ndarray:
        """Linear interpolation of the wind onto arbitrary times."""
        tq = np.atleast_1d(np.asarray(t_query, dtype=float))
        out = np.empty((tq.size, 3))
        for k in range(3):
            out[:, k] = np.interp(tq, self.t, self.velocity[:, k])
        return out

    def speed(self, t_query: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.sample(t_query), axis=1)


def build_wind(cfg: Config, t_grid: np.ndarray, rng: np.random.Generator) -> WindField:
    deg = cfg.degradation
    sev = float(np.clip(deg.wind, 0.0, 1.0))
    heading = math.radians(deg.wind_direction_deg)
    direction = np.array([math.cos(heading), math.sin(heading), 0.0])

    mean_speed = deg.wind_mean_speed * sev
    sigma = deg.wind_gust_sigma * sev
    tau = max(deg.wind_gust_tau, 1e-3)

    n = t_grid.size
    vel = np.zeros((n, 3))
    if sev <= 0.0 or n == 0:
        return WindField(t=t_grid, velocity=vel, direction=direction,
                         mean_speed=0.0, severity=0.0)

    dt = float(np.mean(np.diff(t_grid))) if n > 1 else 0.0
    # Exact discretisation of dx = -x/tau dt + sqrt(2/tau) sigma dW.
    a = math.exp(-dt / tau) if dt > 0 else 0.0
    q = sigma * math.sqrt(max(1.0 - a * a, 0.0))
    gust = np.zeros((n, 3))
    noise = rng.normal(size=(n, 3)) * np.array([1.0, 1.0, 0.35])
    gust[0] = sigma * noise[0] * np.array([1.0, 1.0, 0.35])
    for i in range(1, n):
        gust[i] = a * gust[i - 1] + q * noise[i]

    vel = mean_speed * direction[None, :] + gust
    return WindField(t=t_grid, velocity=vel, direction=direction,
                     mean_speed=mean_speed, severity=sev)


@dataclass
class CanopyMotion:
    """Plants swaying as a travelling wave, driven by the same wind.

    This is the disturbance that breaks the static-world assumption underneath
    every estimator on the ladder, so it is modelled at the source: the
    landmark positions themselves move.  Tier 2 displaces its canopy mesh with
    this same expression, which is why the wind bands on screen are literally
    the corruption in the measurements.
    """

    wind: WindField
    amplitude: float       # m at unit gust
    frequency: float       # Hz
    wavelength: float      # m
    severity: float

    def displacement(self, points: np.ndarray, t: float) -> np.ndarray:
        """Displacement of ``points`` (..., 3) at time ``t``."""
        pts = np.asarray(points, dtype=float)
        if self.severity <= 0.0 or self.amplitude <= 0.0:
            return np.zeros_like(pts)
        w = self.wind.sample(np.array([t]))[0]
        speed = float(np.linalg.norm(w[:2]))
        if speed < 1e-9:
            return np.zeros_like(pts)
        w_hat = np.array([w[0] / speed, w[1] / speed, 0.0])
        phase = 2.0 * math.pi * (pts[..., :2] @ w_hat[:2]) / self.wavelength
        phase = phase - 2.0 * math.pi * self.frequency * t
        # Sway grows with the instantaneous gust, not just the mean.
        amp = self.amplitude * self.severity * speed / max(self.wind.mean_speed, 1.0)
        s = amp * np.sin(phase)
        out = np.zeros_like(pts)
        out[..., 0] = s * w_hat[0]
        out[..., 1] = s * w_hat[1]
        # Heads dip as they bend over.
        out[..., 2] = -0.25 * np.abs(s)
        return out


class DriftOcclusion:
    """Self-inflicted degradation: the actuator fouls its own sensor.

    A decaying memory of recent spray output stands in for the plume hanging
    under the aircraft.  It only reaches the camera when the wind is carrying
    it along with the aircraft rather than off to one side, so the same leg
    flown into a crosswind is clean and flown downwind is not.
    """

    def __init__(self, cfg: Config, wind: WindField):
        self.severity = float(np.clip(cfg.degradation.drift_occlusion, 0.0, 1.0))
        self.tau = max(cfg.degradation.drift_decay_s, 1e-3)
        self.max_dropout = cfg.degradation.drift_max_dropout
        self.wind = wind
        self.load = 0.0

    def update(self, dt: float, spraying: bool) -> None:
        decay = math.exp(-dt / self.tau)
        self.load = self.load * decay + (1.0 - decay) * (1.0 if spraying else 0.0)

    def extra_dropout(self, t: float, heading: np.ndarray) -> float:
        if self.severity <= 0.0 or self.load <= 0.0:
            return 0.0
        w = self.wind.sample(np.array([t]))[0][:2]
        speed = float(np.linalg.norm(w))
        if speed < 1e-6:
            return 0.0
        h = np.asarray(heading, dtype=float)[:2]
        hn = float(np.linalg.norm(h))
        if hn < 1e-6:
            return 0.0
        alignment = float(np.clip(np.dot(w / speed, h / hn), 0.0, 1.0))
        return self.severity * self.max_dropout * self.load * alignment


class SunWashout:
    """Feature quality lost on one heading and not on the reciprocal.

    Because the pattern is a boustrophedon, this splits the dataset cleanly in
    two: every other leg is degraded.  A method that only works when the
    measurements are good gets exactly half a field.
    """

    def __init__(self, cfg: Config, world: World):
        self.severity = float(np.clip(cfg.degradation.sun_washout, 0.0, 1.0))
        self.max_dropout = cfg.degradation.sun_max_dropout
        sun = np.asarray(world.sun_dir, dtype=float)[:2]
        n = float(np.linalg.norm(sun))
        self.sun_horizontal = sun / n if n > 1e-9 else np.array([1.0, 0.0])

    def extra_dropout(self, heading: np.ndarray) -> float:
        if self.severity <= 0.0:
            return 0.0
        h = np.asarray(heading, dtype=float)[:2]
        hn = float(np.linalg.norm(h))
        if hn < 1e-6:
            return 0.0
        into_sun = float(np.clip(np.dot(h / hn, self.sun_horizontal), 0.0, 1.0))
        return self.severity * self.max_dropout * into_sun


class MultipathZone:
    """GNSS degradation clustered along the treeline.

    Keyed to position rather than to time, so it recurs on every row turn at
    that edge instead of at random moments.  A filter that leans on GNSS gets
    hurt in the same place fifteen times.
    """

    def __init__(self, cfg: Config, world: World, rng: np.random.Generator):
        self.severity = float(np.clip(cfg.degradation.multipath, 0.0, 1.0))
        self.radius = cfg.gnss.multipath_radius
        self.bias_scale = cfg.gnss.multipath_bias
        self.outage_prob = cfg.gnss.outage_prob_in_zone
        self.treeline_y = world.treeline_y
        # A slowly varying bias direction, fixed for the session: multipath from
        # a fixed reflector is not zero-mean over a pass.
        ang = rng.uniform(0.0, 2.0 * math.pi)
        self.bias_dir = np.array([math.cos(ang), math.sin(ang), 0.4])

    def proximity(self, position: np.ndarray) -> float:
        """1 at the treeline, 0 at the edge of the zone and beyond."""
        if self.severity <= 0.0 or self.radius <= 0.0:
            return 0.0
        d = abs(float(position[1]) - self.treeline_y)
        return float(np.clip(1.0 - d / self.radius, 0.0, 1.0))

    def is_outage(self, position: np.ndarray, rng: np.random.Generator) -> bool:
        p = self.proximity(position)
        return bool(rng.random() < self.outage_prob * p * self.severity)

    def bias(self, position: np.ndarray, t: float) -> np.ndarray:
        p = self.proximity(position)
        if p <= 0.0:
            return np.zeros(3)
        # Wander slowly so it is not a constant the estimator could absorb.
        wobble = 0.6 + 0.4 * math.sin(0.11 * t)
        return self.bias_scale * self.severity * p * wobble * self.bias_dir


@dataclass
class Degradation:
    """The full set, built once per trial from one rng stream."""

    wind: WindField
    canopy: CanopyMotion
    drift: DriftOcclusion
    sun: SunWashout
    multipath: MultipathZone


def build_degradation(cfg: Config, world: World, t_grid: np.ndarray,
                      rng: np.random.Generator) -> Degradation:
    wind = build_wind(cfg, t_grid, rng)
    deg = cfg.degradation
    canopy = CanopyMotion(
        wind=wind,
        amplitude=deg.canopy_amplitude,
        frequency=deg.canopy_freq,
        wavelength=deg.canopy_wavelength,
        severity=float(np.clip(deg.canopy, 0.0, 1.0)),
    )
    return Degradation(
        wind=wind,
        canopy=canopy,
        drift=DriftOcclusion(cfg, wind),
        sun=SunWashout(cfg, world),
        multipath=MultipathZone(cfg, world, rng),
    )
