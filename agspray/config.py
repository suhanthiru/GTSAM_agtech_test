"""Configuration objects for the spray-drone estimation study.

Everything the harness needs is described by :class:`Config`.  Nested
dataclasses keep the yaml readable and let sweeps override a single leaf with a
dotted key (``"spray.swath_radius"``) without touching the rest.

Derived quantities live in :class:`Derived`.  In particular the three named
fixed-lag windows are *computed from flight parameters* rather than hardcoded,
so changing speed or row length moves them with the physics.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------
# leaves
# --------------------------------------------------------------------------


@dataclass
class FieldCfg:
    """Rectangular field with crop rows running along +x."""

    length_x: float = 120.0         # m, along-row
    row_spacing: float = 8.0        # m between adjacent flight lines
    n_rows: int = 9
    plant_pitch: float = 3.0        # m, landmark grid pitch along a row
    plant_jitter: float = 0.12      # m, lateral scatter of plant positions
    treeline_offset: float = 6.0    # m outside the y_min boundary
    sun_azimuth_deg: float = 20.0   # from +x; near-along-row so legs differ
    sun_elevation_deg: float = 22.0


@dataclass
class FlightCfg:
    altitude: float = 15.0          # m AGL
    cruise_speed: float = 8.0       # m/s
    turn_duration: float = 4.0      # s for a row-to-row reversal
    passes: int = 1                 # 1 or 2
    return_gap_s: float = 0.0       # >0 means pass 2 is a later session
    start_pad: float = 3.0          # s of straight flight before spraying


@dataclass
class SprayCfg:
    swath_radius: float = 5.0       # m, disc radius on the ground
    rate_lpm: float = 12.0          # litres per minute at full rate
    gate_on_turns: bool = True      # stop spraying while turning
    grid_res_frac: float = 0.125    # raster cell size as a fraction of the swath radius
    throttle_on_covariance: bool = False  # second experiment (causal version of Q2)
    throttle_sigma_frac: float = 0.35     # sigma/r at which the rate is fully cut
    retreatment_cost_per_m2: float = 3.0  # cost multiplier applied to gap area
    tick_hz: float = 20.0                 # boom on/off decision rate
    coverage_gate_frac: float = 0.35      # section control: spray only if this much
                                          # of the swath is *believed* untreated
    over_dose_frac: float = 1.5           # dose above this counts as double-dosed
    under_dose_frac: float = 0.5          # dose below this counts as a gap


@dataclass
class ImuCfg:
    rate_hz: float = 200.0
    accel_noise_density: float = 0.02     # m/s^2/sqrt(Hz)
    gyro_noise_density: float = 0.0015    # rad/s/sqrt(Hz)
    accel_bias_rw: float = 0.0008         # m/s^3/sqrt(Hz)
    gyro_bias_rw: float = 5.0e-5          # rad/s^2/sqrt(Hz)
    accel_bias_init: float = 0.05         # m/s^2, 1-sigma at t=0
    gyro_bias_init: float = 0.002         # rad/s, 1-sigma at t=0


@dataclass
class GnssCfg:
    rate_hz: float = 5.0
    sigma_horizontal: float = 0.6         # m
    sigma_vertical: float = 1.1           # m
    multipath_radius: float = 20.0        # m from the treeline where it degrades
    multipath_bias: float = 2.5           # m, slowly varying offset in that zone
    outage_prob_in_zone: float = 0.75     # chance a fix is dropped in the zone


@dataclass
class CameraCfg:
    rate_hz: float = 10.0
    width: int = 640
    height: int = 480
    fx: float = 480.0
    fy: float = 480.0
    pixel_sigma: float = 1.2
    max_features: int = 40
    detect_prob: float = 0.9              # nominal per-plant detection rate


@dataclass
class DegradationCfg:
    """Each axis is a severity in [0, 1]; 0 disables that axis entirely."""

    wind: float = 0.6
    canopy: float = 0.6
    drift_occlusion: float = 0.5
    multipath: float = 0.7
    sun_washout: float = 0.5
    row_aliasing: float = 0.8             # descriptor similarity between rows

    wind_mean_speed: float = 5.0          # m/s at severity 1
    wind_direction_deg: float = 210.0     # direction the wind blows *towards*
    wind_gust_tau: float = 6.0            # s, Ornstein-Uhlenbeck correlation time
    wind_gust_sigma: float = 2.2          # m/s at severity 1
    airframe_tau: float = 1.6             # s, airframe response to a gust
    airframe_gain: float = 0.09           # accel per (m/s) of gust at severity 1
    canopy_amplitude: float = 0.18        # m of plant sway at severity 1
    canopy_freq: float = 0.55             # Hz
    canopy_wavelength: float = 14.0       # m, travelling wave in the canopy
    drift_decay_s: float = 6.0            # s, memory of recent spray for occlusion
    drift_max_dropout: float = 0.8        # extra dropout probability at severity 1
    sun_max_dropout: float = 0.6          # extra dropout on the into-sun heading


@dataclass
class EstimatorCfg:
    keyframe_hz: float = 5.0
    robust: bool = True                   # ablation axis 1
    estimate_bias: bool = True            # ablation axis 2
    huber_k_pixel: float = 1.345
    process_noise_scale: float = 1.0      # tuning knob
    isam2_relin_thresh: float = 0.05
    isam2_relin_skip: int = 5
    lag_short: float | None = None        # None -> derived from flight parameters
    lag_medium: float | None = None
    lag_long: float | None = None
    assoc_gate_chi2: float = 9.21         # 2 dof, 99%
    assoc_descriptor_thresh: float = 0.55
    assoc_new_per_frame: int = 12         # cap on tracks initialised per frame
    assoc_min_separation: float = 0.8     # m, below this a new track is a duplicate
    lm_sigma0: float = 1.5                # m, 1-sigma of a plane-backprojected feature
    landmark_lifetime_scale: float = 1.0  # lifetime = scale x row period, same for
                                          # every method: only the *pose* horizon varies
    max_live_landmarks: int = 320
    latency_tol_m: float = 0.25           # convergence band for correction latency


@dataclass
class RunCfg:
    seed: int = 0
    duration_s: float | None = None       # None -> the whole coverage pattern
    record_history_hz: float = 5.0


@dataclass
class Config:
    field_: FieldCfg = field(default_factory=FieldCfg)
    flight: FlightCfg = field(default_factory=FlightCfg)
    spray: SprayCfg = field(default_factory=SprayCfg)
    imu: ImuCfg = field(default_factory=ImuCfg)
    gnss: GnssCfg = field(default_factory=GnssCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    degradation: DegradationCfg = field(default_factory=DegradationCfg)
    estimator: EstimatorCfg = field(default_factory=EstimatorCfg)
    run: RunCfg = field(default_factory=RunCfg)

    # ---------------- io ----------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        cfg = cls()
        for key, val in raw.items():
            name = "field_" if key == "field" else key
            if not hasattr(cfg, name):
                raise KeyError(f"unknown config section {key!r}")
            section = getattr(cfg, name)
            known = {f.name for f in fields(section)}
            for sub, v in (val or {}).items():
                if sub not in known:
                    raise KeyError(f"unknown config key {key}.{sub!r}")
                setattr(section, sub, v)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["field"] = out.pop("field_")
        return out

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    def override(self, **dotted: Any) -> "Config":
        """Return a copy with ``section.key=value`` overrides applied."""
        cfg = self.copy()
        for key, val in dotted.items():
            sec, _, leaf = key.partition(".")
            if not leaf:
                raise KeyError(f"override {key!r} must be 'section.key'")
            sec = "field_" if sec == "field" else sec
            section = getattr(cfg, sec, None)
            if section is None or not is_dataclass(section):
                raise KeyError(f"unknown config section in {key!r}")
            if leaf not in {f.name for f in fields(section)}:
                raise KeyError(f"unknown config key {key!r}")
            setattr(section, leaf, val)
        return cfg

    # ---------------- derived ----------------

    @property
    def derived(self) -> "Derived":
        return Derived.of(self)


@dataclass(frozen=True)
class Derived:
    """Quantities implied by the config.  Never set directly."""

    leg_time: float
    turn_time: float
    row_period: float
    pass_time: float
    swath_traverse_time: float
    occlusion_time: float
    lag_short: float
    lag_medium: float
    lag_long: float
    grid_res: float
    row_spacing_over_swath: float
    field_width: float
    field_area: float
    landmark_lifetime: float
    target_dose: float

    @staticmethod
    def of(cfg: Config) -> "Derived":
        v = cfg.flight.cruise_speed
        r = cfg.spray.swath_radius
        leg_time = cfg.field_.length_x / v
        turn_time = cfg.flight.turn_duration
        row_period = leg_time + turn_time

        # Time to fly through the sprayer's own footprint.  This is the natural
        # latency yardstick: a correction landing later than this arrives after
        # the aircraft has already moved past everything it could have fixed.
        swath_traverse = 2.0 * r / v

        # How long the aircraft typically goes without usable GNSS: crossing the
        # multipath zone on the way in, the turn itself, and back out again.
        occlusion = 2.0 * cfg.gnss.multipath_radius / v + turn_time

        est = cfg.estimator
        kf_dt = 1.0 / est.keyframe_hz
        lag_short = est.lag_short if est.lag_short is not None else max(4.0 * kf_dt, 0.5 * swath_traverse)
        lag_medium = est.lag_medium if est.lag_medium is not None else occlusion
        lag_long = est.lag_long if est.lag_long is not None else row_period

        n_rows = cfg.field_.n_rows
        pass_time = n_rows * leg_time + (n_rows - 1) * turn_time + cfg.flight.start_pad
        width = (n_rows - 1) * cfg.field_.row_spacing

        return Derived(
            leg_time=leg_time,
            turn_time=turn_time,
            row_period=row_period,
            pass_time=pass_time,
            swath_traverse_time=swath_traverse,
            occlusion_time=occlusion,
            lag_short=lag_short,
            lag_medium=lag_medium,
            lag_long=lag_long,
            grid_res=cfg.spray.grid_res_frac * r,
            row_spacing_over_swath=cfg.field_.row_spacing / (2.0 * r),
            field_width=width,
            field_area=cfg.field_.length_x * width,
            landmark_lifetime=est.landmark_lifetime_scale * row_period,
            # Litres per square metre laid down by one straight pass at cruise.
            target_dose=(cfg.spray.rate_lpm / 60.0) / (v * 2.0 * r),
        )


def load(path: str | Path | None = None) -> Config:
    """Load a config, falling back to the packaged defaults."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    return Config.from_yaml(path)
