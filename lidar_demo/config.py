"""Configuration for the LiDAR reconstruction demo.

Same shape as ``agspray.config``: nested dataclasses, yaml on top, dotted
overrides for one leaf at a time.  Nothing here is derived from anything else
except through :class:`Derived`, so a value read in the config is the value the
simulator used.

The numbers that matter to the story are in :class:`ExtrinsicCfg`.  The
simulator mounts the LiDAR at ``nominal + true_offset``; the reconstruction is
handed ``nominal`` and nothing else.  That single discrepancy is what the factor
graph has to discover.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------


@dataclass
class SceneCfg:
    """Terrain, canopy and props.

    A flat plane reconstructs to a flat plane and shows nothing, so the terrain
    carries relief at three scales: a constant slope, metre-scale rolling
    macro relief, and centimetre furrows at row spacing.  The drainage ditch is
    the one hard edge in the scene and it anchors the eye in a cross-section.
    """

    survey_x: float = 120.0          # m, survey extent along the flight lines
    survey_y: float = 84.0           # m, across
    context: float = 40.0            # m of surrounding terrain beyond the survey
    grid_dx: float = 0.25            # m, heightfield sample pitch

    slope_deg: float = 1.5           # overall tilt of the field
    slope_azimuth_deg: float = 20.0  # direction of steepest ascent, from +x
    relief_amp: float = 2.0          # m, half-range of the macro relief
    relief_lambda: tuple[float, float] = (45.0, 110.0)   # m, bump wavelengths
    n_relief_modes: int = 5

    ditch_p0: tuple[float, float] = (0.0, 10.0)
    ditch_p1: tuple[float, float] = (120.0, 70.0)
    ditch_depth: float = 1.5
    ditch_halfwidth: float = 2.5

    row_spacing: float = 2.0         # m, crop rows (wide on purpose, see spec 2.2)
    furrow_amp: tuple[float, float] = (0.05, 0.15)   # m ridge height range
    roughness: float = 0.05          # m, clod-scale soil texture
    roughness_lambda: tuple[float, float] = (0.5, 3.0)   # m

    # Variation in canopy height at the scale of a clump of heads.  Not
    # decoration, and the amplitude is not free.  The crop rows run parallel to
    # the flight lines, so the furrows constrain a scan matcher across track and
    # not at all along it; if the canopy is modelled as a smooth sheet, the
    # along-track direction has no geometry in it and registration slides metres
    # while still fitting.  Real wheat is nothing like smooth at this scale --
    # lodged patches, uneven emergence and clump-to-clump height differences run
    # to 10-15 cm -- and that texture has to survive the 20 cm registration
    # voxel to be of any use, which fixes both the amplitude and the wavelength.
    canopy_texture: float = 0.14     # m, 1-sigma of the clump-scale variation
    canopy_texture_lambda: tuple[float, float] = (0.6, 3.0)   # m

    canopy_healthy: float = 1.2      # m
    canopy_stressed: float = 0.6
    block_nx: int = 3                # canopy blocks across x
    block_ny: int = 4
    n_bare_blocks: int = 2
    n_stressed_blocks: int = 4
    canopy_ditch_clearance: float = 3.0   # m of bare ground either side of the ditch

    treeline_y: float = -6.0         # m; also the GNSS multipath edge
    treeline_x: tuple[float, float] = (-20.0, 140.0)
    tree_pitch: tuple[float, float] = (4.0, 6.0)
    tree_height: tuple[float, float] = (6.0, 10.0)

    windmill_xy: tuple[float, float] = (132.0, 40.0)
    windmill_height: float = 12.0
    farmhouse_xy: tuple[float, float] = (-18.0, 55.0)
    outbuilding_xy: tuple[float, float] = (-22.0, 70.0)
    scarecrow_xy: tuple[float, float] = (45.0, 33.0)

    post_x0: float = 10.0
    post_dx: float = 20.0
    n_post_x: int = 6
    post_y: tuple[float, float] = (24.0, 60.0)
    post_jitter: float = 3.0
    post_height: float = 1.8

    coverage_window: float = 40.0    # m, window used by vertical_coverage()
    min_verticals: int = 2           # spec 2.3: two verticals in any close-up


# ---------------------------------------------------------------------------
# flight
# ---------------------------------------------------------------------------


@dataclass
class FlightCfg:
    """Boustrophedon with alternating headings, then one crossing pass.

    ``spacing`` is set for 40-50% LiDAR sidelap at ``altitude``.  Adjacent legs
    fly opposite directions, which is what makes the boresight observable at
    all: flown all one way, the same mounting error biases every pass
    identically and cancels out of every inter-pass comparison.
    """

    altitude: float = 15.0           # m AGL
    speed: float = 5.0               # m/s
    spacing: float = 12.0            # m between flight lines
    n_passes: int = 8
    turn_time: float = 5.0           # s per row reversal (quintic, C2)
    lead_in: float = 15.0            # m of straight flight before the first line
    treeline_pass: int = 7           # index of the pass that runs the treeline
    crossing_x: float = 60.0         # m, the perpendicular closing pass
    crossing_pad: float = 12.0       # m beyond the survey at each end of it

    gust_amp: tuple[float, float, float] = (0.3, 0.3, 0.15)   # m
    gust_periods: tuple[float, float, float] = (6.5, 11.0, 17.0)  # s


# ---------------------------------------------------------------------------
# sensors
# ---------------------------------------------------------------------------


@dataclass
class LidarCfg:
    """Spinning LiDAR.  Geometry is clean; all the damage is navigation and mount.

    ``max_scan_angle_deg`` is the one number here that is a processing choice
    rather than a datasheet figure, and it is worth being explicit about.

    A 360-degree spinner tilted 35 degrees forward does physically return from
    everywhere: the beams near the side of the fan point almost horizontally and
    hit the ground fifty metres away.  Airborne survey practice discards those
    by scan-angle rank, because at that obliquity the footprint is elongated,
    the path through the canopy is long, and the range is unreliable.  Keeping
    them here would be worse than unrealistic: every pass would see the whole
    field, the swath-to-swath disagreement would average away, and the
    one-ridge-per-flight-line structure the boresight error is supposed to
    produce would vanish.  So returns are clipped at a fixed angle from body
    nadir, which is a clean aperture with no statistical tail.  Fifty degrees
    gives a usable half swath of about 12 m at 15 m AGL, and therefore the
    40-50 percent sidelap the flight plan is built around.
    """

    n_rings: int = 64
    n_cols: int = 512                # azimuth columns per sweep
    rate_hz: float = 10.0
    vfov_deg: tuple[float, float] = (-20.0, 20.0)
    min_range: float = 1.0
    max_range: float = 100.0
    range_sigma: float = 0.02        # m
    canopy_extinction: float = 1.1   # 1/m, Beer-Lambert through the crop
    crown_extinction: float = 1.6    # 1/m, through a tree crown
    # Mean depth into the vegetation at which a stopped return comes back.  Kept
    # small on purpose: a first-return sensor triggers near the top of the
    # canopy, and if this is allowed to grow past the canopy's own height
    # texture it buries the surface shape under noise and scan matching over the
    # crop stops working.
    canopy_penetration: float = 0.05  # m, mean depth of a return stopped in crop
    crown_penetration: float = 0.30  # m, same for tree crowns
    canopy_dropout: float = 0.15
    crown_dropout: float = 0.35
    base_dropout: float = 0.02
    max_scan_angle_deg: float = 50.0  # hard aperture, see the note below
    grazing_dropout: float = 0.6      # coefficient on (1 - cos incidence)^4


@dataclass
class ExtrinsicCfg:
    """Sensor-to-body mount.

    ``nominal`` is what the reconstruction is told, as if read off a CAD model.
    ``true_rpy_offset`` is the error actually built into the airframe.  Two
    degrees is not nothing: at 15 m AGL with a tilted mount, returns land 15-25 m
    out, and one degree at 20 m displaces a point 35 cm.
    """

    nominal_rpy_deg: tuple[float, float, float] = (0.0, 35.0, 0.0)
    nominal_t: tuple[float, float, float] = (0.10, 0.0, -0.05)
    true_rpy_offset_deg: tuple[float, float, float] = (0.8, 1.5, 2.0)
    true_t_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ImuCfg:
    """Industrial-MEMS grade.

    The spec asks for consumer MEMS multiplied 5-10x *and* for 1-2 m of drift.
    Those two cannot both hold: at sigma_g ~1e-3 rad/s/sqrt(Hz) a 30 s
    dead-reckoned window leaks ~11 m, and over the whole 265 s flight any MEMS
    part leaks hundreds of metres.  The drift target is the one worth keeping,
    so these are industrial-grade numbers (~0.27 deg/sqrt(h) ARW) and the drift
    target is read over the GNSS-denied window rather than the whole flight.
    Field names match ``agspray.config.ImuCfg`` so its synthesiser is reused.
    """

    rate_hz: float = 400.0
    accel_noise_density: float = 0.011    # m/s^2/sqrt(Hz)
    gyro_noise_density: float = 8.0e-5    # rad/s/sqrt(Hz)
    accel_bias_rw: float = 2.0e-4         # m/s^3/sqrt(Hz)
    gyro_bias_rw: float = 5.0e-6          # rad/s^2/sqrt(Hz)
    accel_bias_init: float = 0.02         # m/s^2, 1-sigma at t=0
    gyro_bias_init: float = 0.001         # rad/s, 1-sigma at t=0
    drift_window_s: float = 30.0          # window the drift gate is read over


@dataclass
class GnssCfg:
    """Never isotropic: vertical is always the worse axis."""

    rate_hz: float = 1.0
    sigma_horizontal: float = 0.5    # m
    sigma_vertical: float = 1.0      # m
    dropout_radius: float = 9.0      # m from the treeline
    random_dropout: float = 0.02


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------


@dataclass
class BlueCfg:
    """The drone's own belief: a loosely-coupled inertial/GNSS filter.

    How much drift survives into the blue map is set by the sensors and by the
    length of the GNSS outage, not by a tuning knob: the filter is a plain
    error-state update against the measured position noise.  What it cannot do
    is revisit the past, and the mounting angle is not in its state at all.
    """

    cov_rate_hz: float = 40.0        # covariance propagation rate (see blue.py)
    align_static_s: float = 0.5      # s of specific force used for initial roll/pitch
    align_heading_s: float = 3.0     # s of GNSS motion used for initial yaw


@dataclass
class GraphCfg:
    keyframe_trans: float = 1.0      # m
    keyframe_rot_deg: float = 10.0

    # Each keyframe's submap gathers the sweeps collected within this much
    # travel either side of it, so consecutive submaps overlap heavily.  A
    # single keyframe's own metre of flying is a thin crescent of ground with
    # too little extent to register against anything; a few metres of it spans
    # real terrain and usually a prop.
    submap_radius: float = 4.0       # m of travel either side
    submap_max_points: int = 80000

    voxel: float = 0.2               # m, registration downsample
    max_corr: float = 0.6            # m, fine stage
    voxel_coarse: float = 0.4        # keep the crop texture: at a 1 m voxel the
    max_corr_coarse: float = 1.5     # canopy averages to a plane and GICP slides
    min_inlier_frac: float = 0.3
    # How far a registration is allowed to move from its initial guess before
    # it is thrown away.  A field of crop rows is close to self-similar along
    # track, so a matcher handed a poor start will happily slide several metres
    # and report an excellent fit; the resulting measurement then looks like a
    # mounting-angle error and the solve chases it.
    max_seq_shift: tuple[float, float] = (1.5, 6.0)     # m, deg from init
    max_cross_shift: tuple[float, float] = (2.5, 8.0)

    overlap_radius: float = 10.0     # m
    overlap_min_dt: float = 5.0      # s
    overlap_exclude_seq: int = 2     # keyframe index distance kept out of it
    overlap_max_per_kf: int = 3

    huber_k: float = 1.345
    cross_kernel: str = "huber"      # or "cauchy"
    sigma_rot_bounds: tuple[float, float] = (0.1, 5.0)     # deg
    sigma_trans_bounds: tuple[float, float] = (0.02, 1.0)  # m
    fallback_seq: tuple[float, float] = (0.3, 0.05)        # deg, m
    fallback_cross: tuple[float, float] = (0.5, 0.10)

    prior_E_rot_deg: float = 5.0
    prior_E_trans: float = 0.02
    lm_max_iters: int = 60
    lm_stage_a_iters: int = 20
    rounds: int = 6
    converge_E_deg: float = 0.02


@dataclass
class MapCfg:
    cell: float = 0.10               # m, raster grid
    ground_pct: float = 5.0
    canopy_pct: float = 95.0
    min_pts_per_cell: int = 3
    stride: int = 1                  # point decimation when projecting


@dataclass
class RenderCfg:
    blue: str = "#2E6FD6"
    green: str = "#1D9E75"
    background: str = "#05070A"
    truth: str = "#FFFFFF"
    point_size: float = 1.5
    point_alpha: float = 0.35
    width: int = 1920
    height: int = 1080
    fps: int = 30
    section_x: float = 60.0          # m, where the cross-section band is cut
    section_halfwidth: float = 0.5   # m, half of the 1 m band
    morph_frames: int = 45
    morph_stagger: float = 0.3


@dataclass
class RunCfg:
    seed: int = 0
    name: str = "run01"
    out_root: str = "data"
    backend: str = "open3d"          # or "heightfield"
    workers: int = 1
    write_sweep_truth: bool = True


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


@dataclass
class DemoConfig:
    scene: SceneCfg = field(default_factory=SceneCfg)
    flight: FlightCfg = field(default_factory=FlightCfg)
    lidar: LidarCfg = field(default_factory=LidarCfg)
    extrinsic: ExtrinsicCfg = field(default_factory=ExtrinsicCfg)
    imu: ImuCfg = field(default_factory=ImuCfg)
    gnss: GnssCfg = field(default_factory=GnssCfg)
    blue: BlueCfg = field(default_factory=BlueCfg)
    graph: GraphCfg = field(default_factory=GraphCfg)
    map: MapCfg = field(default_factory=MapCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    run: RunCfg = field(default_factory=RunCfg)

    # ---------------- io ----------------

    @classmethod
    def from_yaml(cls, path: str | Path, strict: bool = True) -> "DemoConfig":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw, strict=strict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], strict: bool = True) -> "DemoConfig":
        cfg = cls()
        unknown = []
        for key, val in raw.items():
            if not hasattr(cfg, key):
                if strict:
                    raise KeyError(f"unknown config section {key!r}")
                unknown.append(key)
                continue
            section = getattr(cfg, key)
            known = {f.name for f in fields(section)}
            for sub, v in (val or {}).items():
                if sub not in known:
                    if strict:
                        raise KeyError(f"unknown config key {key}.{sub!r}")
                    unknown.append(f"{key}.{sub}")
                    continue
                if isinstance(getattr(section, sub), tuple) and isinstance(v, list):
                    v = tuple(v)
                setattr(section, sub, v)
        if unknown:
            import warnings
            warnings.warn(f"ignoring config keys not in the current schema: "
                          f"{', '.join(unknown)}", stacklevel=2)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(_plain(self.to_dict()), fh, sort_keys=False)

    def copy(self) -> "DemoConfig":
        return copy.deepcopy(self)

    def override(self, **dotted: Any) -> "DemoConfig":
        """Return a copy with ``section.key=value`` overrides applied."""
        cfg = self.copy()
        for key, val in dotted.items():
            sec, _, leaf = key.partition(".")
            if not leaf:
                raise KeyError(f"override {key!r} must be 'section.key'")
            section = getattr(cfg, sec, None)
            if section is None or not is_dataclass(section):
                raise KeyError(f"unknown config section in {key!r}")
            if leaf not in {f.name for f in fields(section)}:
                raise KeyError(f"unknown config key {key!r}")
            setattr(section, leaf, val)
        return cfg

    @property
    def derived(self) -> "Derived":
        return Derived.of(self)


def _plain(obj: Any) -> Any:
    """Tuples to lists so the yaml round-trips through ``safe_dump``."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


@dataclass(frozen=True)
class Derived:
    """Quantities implied by the config.  Never set directly."""

    leg_time: float
    pass_period: float
    survey_duration: float
    n_sweeps: int
    sweep_period: float
    rays_per_sweep: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    footprint_radius: float
    sidelap_frac: float
    max_grazing_deg: float = 70.0

    @staticmethod
    def of(cfg: "DemoConfig") -> "Derived":
        f = cfg.flight
        s = cfg.scene
        leg_time = s.survey_x / f.speed
        pass_period = leg_time + f.turn_time

        # Passes, turns, the lead-in, the transition onto the crossing line and
        # the crossing line itself.  A rough figure, refined by the planner.
        cross_len = s.survey_y + 2.0 * f.crossing_pad
        duration = (f.n_passes * leg_time + (f.n_passes - 1) * f.turn_time
                    + f.lead_in / f.speed + 1.5 * f.crossing_x / f.speed
                    + cross_len / f.speed)

        sweep_period = 1.0 / cfg.lidar.rate_hz

        # The sensor spins through a full 360 degrees, so the footprint is not a
        # fan-limited strip: it is bounded by the usable scan angle and by range.
        # This is an upper bound; the achieved half swath is measured by gate 1,
        # because the beam fan does not reach every part of the aperture.
        import numpy as np

        r_graze = f.altitude * np.tan(np.radians(cfg.lidar.max_scan_angle_deg))
        r_range = np.sqrt(max(cfg.lidar.max_range ** 2 - f.altitude ** 2, 0.0))
        footprint = float(min(r_graze, r_range))
        sidelap = float(np.clip(1.0 - f.spacing / max(2.0 * footprint, 1e-6), 0.0, 1.0))

        return Derived(
            leg_time=leg_time,
            pass_period=pass_period,
            survey_duration=duration,
            n_sweeps=int(duration * cfg.lidar.rate_hz),
            sweep_period=sweep_period,
            rays_per_sweep=cfg.lidar.n_rings * cfg.lidar.n_cols,
            x_min=0.0,
            x_max=s.survey_x,
            y_min=0.0,
            y_max=s.survey_y,
            footprint_radius=footprint,
            sidelap_frac=sidelap,
        )


def load(path: str | Path | None = None) -> DemoConfig:
    """Load a config, falling back to the packaged defaults."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "lidar_demo.yaml"
    return DemoConfig.from_yaml(path)


def load_run_config(run_dir: str | Path) -> DemoConfig:
    """Read the config a recorded run was made with.

    Tolerant of keys the current schema no longer has: an archived run is a
    record of what the simulator did, and it should stay readable after the
    estimator's own settings have moved on.  Anything dropped is warned about
    rather than silently ignored.
    """
    return DemoConfig.from_yaml(Path(run_dir) / "config_resolved.yaml", strict=False)
