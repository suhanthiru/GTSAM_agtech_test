"""Synthetic sensor streams.

Everything random is drawn *once per seed*, before any estimator runs, and the
whole bundle is then replayed identically for all six methods.  That is what
makes the seeds paired: two methods differ because of what they do with the
data, never because they saw different data.

One wrinkle needs care.  Spray drift occlusion is *method dependent* - it
depends on how much that method's controller has been spraying - so the set of
features a method sees is not fixed in advance.  Drawing a fresh random number
at that point would break pairing.  Instead each candidate observation carries a
uniform ``u`` drawn up front, and dropout is a threshold on it:

    keep  <=>  u < detect_prob * (1 - sun_dropout) * (1 - drift_dropout)

Raising the dropout only ever removes observations, and it removes the same ones
for every method that reaches the same drift level.  Pairing survives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .degradation import Degradation, build_degradation
from .measurement_models import Camera, body_to_camera, project
from .trajectory import FlightPlan, TrueState, build_truth
from .world import World, build_world


@dataclass
class CameraFrame:
    """One image worth of point features.

    ``landmark_id`` is the true identity and is used only for scoring; no
    estimator is allowed to read it.
    """

    t: float
    kf: int                      # keyframe index
    imu_index: int               # sample in the IMU stream this frame lands on
    uv: np.ndarray               # (K, 2) noisy pixels
    descriptors: np.ndarray      # (K, D) noisy unit descriptors
    landmark_id: np.ndarray      # (K,) true landmark index
    dropout_u: np.ndarray        # (K,) pre-drawn uniforms
    base_keep: float             # detection probability before drift occlusion
    heading: np.ndarray          # (3,) true horizontal heading, for the drift model
    pass_id: int

    def visible(self, extra_dropout: float = 0.0) -> np.ndarray:
        """Mask of features that survive dropout at the given drift level."""
        return self.dropout_u < self.base_keep * (1.0 - float(extra_dropout))


@dataclass
class SensorData:
    """Every measurement for one trial, on one shared clock."""

    t: np.ndarray                # (T,) IMU / truth clock
    accel: np.ndarray            # (T, 3) specific force in body frame
    gyro: np.ndarray             # (T, 3) body rates
    t_gnss: np.ndarray           # (G,)
    gnss_index: np.ndarray       # (G,) index into the IMU clock
    gnss_pos: np.ndarray         # (G, 3)
    gnss_valid: np.ndarray       # (G,) bool, False during an outage
    gnss_sigma: np.ndarray       # (3,) reported 1-sigma (deliberately optimistic
                                 #      in the multipath zone: the receiver does
                                 #      not know it is being reflected)
    frames: list[CameraFrame]
    t_kf: np.ndarray             # (K,) keyframe times
    kf_index: np.ndarray         # (K,) index into the IMU clock
    pass_id: np.ndarray          # (T,) which pass each IMU sample belongs to
    bias_accel: np.ndarray       # (T, 3) true accelerometer bias, for scoring
    bias_gyro: np.ndarray        # (T, 3)

    @property
    def n_kf(self) -> int:
        return int(self.t_kf.size)


@dataclass
class Trial:
    """A fully specified trial: world, truth, degradation, measurements."""

    cfg: Config
    seed: int
    world: World
    truth: TrueState
    degradation: Degradation
    sensors: SensorData
    camera: Camera


# --------------------------------------------------------------------------
# individual streams
# --------------------------------------------------------------------------


def _imu_stream(cfg: Config, truth: TrueState, rng: np.random.Generator):
    """Specific force and body rates with a bias random walk."""
    imu = cfg.imu
    n = len(truth)
    dt = 1.0 / imu.rate_hz

    b_a = np.empty((n, 3))
    b_g = np.empty((n, 3))
    b_a[0] = rng.normal(scale=imu.accel_bias_init, size=3)
    b_g[0] = rng.normal(scale=imu.gyro_bias_init, size=3)
    step_a = rng.normal(scale=imu.accel_bias_rw * np.sqrt(dt), size=(n, 3))
    step_g = rng.normal(scale=imu.gyro_bias_rw * np.sqrt(dt), size=(n, 3))
    b_a[1:] = b_a[0] + np.cumsum(step_a[1:], axis=0)
    b_g[1:] = b_g[0] + np.cumsum(step_g[1:], axis=0)

    from .measurement_models import GRAVITY
    specific = np.einsum("tji,tj->ti", truth.rotation, truth.acceleration - GRAVITY)

    sigma_a = imu.accel_noise_density / np.sqrt(dt)
    sigma_g = imu.gyro_noise_density / np.sqrt(dt)
    accel = specific + b_a + rng.normal(scale=sigma_a, size=(n, 3))
    gyro = truth.omega + b_g + rng.normal(scale=sigma_g, size=(n, 3))
    return accel, gyro, b_a, b_g


def _gnss_stream(cfg: Config, truth: TrueState, deg: Degradation,
                 rng: np.random.Generator):
    """Low-rate position fixes, degraded in a band along the treeline.

    Outages and biases are keyed to *position*, so they recur at the same field
    edge on every row rather than at random moments.  A filter that leans on
    GNSS is therefore hurt in the same place once per row, which is exactly the
    correlated failure the study is about.
    """
    gcfg = cfg.gnss
    step = max(int(round(cfg.imu.rate_hz / gcfg.rate_hz)), 1)
    idx = np.arange(0, len(truth), step)
    t = truth.t[idx]
    pos = truth.position[idx]

    prox = np.clip(1.0 - np.abs(pos[:, 1] - deg.multipath.treeline_y)
                   / max(deg.multipath.radius, 1e-9), 0.0, 1.0)
    prox = prox * deg.multipath.severity
    wobble = 0.6 + 0.4 * np.sin(0.11 * t)
    bias = (gcfg.multipath_bias * prox * wobble)[:, None] * deg.multipath.bias_dir[None, :]

    sigma = np.array([gcfg.sigma_horizontal, gcfg.sigma_horizontal, gcfg.sigma_vertical])
    noise = rng.normal(size=(idx.size, 3)) * sigma[None, :]
    valid = rng.random(idx.size) >= gcfg.outage_prob_in_zone * prox

    return t, idx, pos + bias + noise, valid, sigma


def _frame_times(cfg: Config, truth: TrueState):
    step = max(int(round(cfg.imu.rate_hz / cfg.estimator.keyframe_hz)), 1)
    idx = np.arange(0, len(truth), step)
    return truth.t[idx], idx


def _camera_stream(cfg: Config, truth: TrueState, world: World, deg: Degradation,
                   cam: Camera, kf_t: np.ndarray, kf_idx: np.ndarray,
                   rng: np.random.Generator) -> list[CameraFrame]:
    """Bearing measurements to crop landmarks, from a down-looking camera.

    The landmarks move: :class:`~agspray.degradation.CanopyMotion` displaces
    them with the wind before they are projected.  That breaks the static-world
    assumption at the source rather than by inflating a noise term, which is the
    point - a larger sigma is something every estimator can absorb, a moving
    world is not.
    """
    ccfg = cfg.camera
    frames: list[CameraFrame] = []
    lm = world.landmarks
    desc = world.descriptors

    # Only landmarks near the footprint can possibly project, and the footprint
    # is small compared with the field.  Pre-filtering keeps this loop cheap.
    reach = 1.2 * cfg.flight.altitude * max(ccfg.width / cam.fx, ccfg.height / cam.fy)

    for k, (t, i) in enumerate(zip(kf_t, kf_idx)):
        R_wb = truth.rotation[i]
        p_wb = truth.position[i]
        heading = truth.heading[i]

        near = (np.abs(lm[:, 0] - p_wb[0]) < reach) & (np.abs(lm[:, 1] - p_wb[1]) < reach)
        cand = np.flatnonzero(near)
        if cand.size == 0:
            frames.append(CameraFrame(t=float(t), kf=k, imu_index=int(i),
                                      uv=np.zeros((0, 2)), descriptors=np.zeros((0, desc.shape[1])),
                                      landmark_id=np.zeros(0, dtype=int),
                                      dropout_u=np.zeros(0), base_keep=0.0,
                                      heading=heading, pass_id=int(truth.pass_id[i])))
            continue

        pts = lm[cand] + deg.canopy.displacement(lm[cand], float(t))
        R_wc, p_wc = body_to_camera(R_wb, p_wb)
        uv, _, valid = project(R_wc, p_wc, pts, cam)
        sel = cand[valid]
        uv = uv[valid]

        if sel.size > ccfg.max_features:
            keep = rng.permutation(sel.size)[: ccfg.max_features]
            keep.sort()
            sel, uv = sel[keep], uv[keep]

        sun_drop = deg.sun.extra_dropout(heading)
        base_keep = ccfg.detect_prob * (1.0 - sun_drop)

        uv = uv + rng.normal(scale=cam.pixel_sigma, size=uv.shape)
        # Washed-out features are not just rarer, they are less distinctive.
        d_sigma = 0.02 + 0.08 * sun_drop
        d = desc[sel] + rng.normal(scale=d_sigma, size=(sel.size, desc.shape[1]))
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)

        frames.append(CameraFrame(
            t=float(t), kf=k, imu_index=int(i), uv=uv, descriptors=d,
            landmark_id=sel, dropout_u=rng.random(sel.size), base_keep=float(base_keep),
            heading=heading, pass_id=int(truth.pass_id[i]),
        ))
    return frames


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_plans(cfg: Config) -> list[FlightPlan]:
    """Flight plans for the configured number of passes.

    A second pass is flown on the same lines but in the opposite order, which is
    what an operator does when returning to a field: you do not want the second
    application to sit exactly on top of the first one's turn artefacts.
    """
    base = FlightPlan.boustrophedon(cfg)
    plans = [base]
    for _ in range(max(cfg.flight.passes, 1) - 1):
        plans.append(FlightPlan.boustrophedon(cfg, row_y=base.row_y[::-1].copy()))
    return plans


def synthesize(cfg: Config, seed: int) -> Trial:
    """Build one complete trial.  Deterministic in ``(cfg, seed)``."""
    rng = np.random.default_rng(seed)

    world = build_world(cfg, rng)
    plans = build_plans(cfg)

    # Truth is needed to build the wind grid, and the wind is needed to build
    # truth (it pushes the airframe around).  Resolve it by laying down the time
    # grid first, sampling the wind on it, then building the path.
    dt = 1.0 / cfg.imu.rate_hz
    total = sum(p.duration() for p in plans) + cfg.flight.return_gap_s * (len(plans) - 1)
    t_grid = np.arange(int(round(total / dt))) * dt

    stub = build_truth(cfg, wind=None, plans=plans)
    del t_grid
    deg = build_degradation(cfg, world, stub.t, rng)
    truth = build_truth(cfg, wind=deg.wind, plans=plans)

    cam = Camera.from_cfg(cfg.camera)
    accel, gyro, b_a, b_g = _imu_stream(cfg, truth, rng)
    t_gnss, gnss_idx, gnss_pos, gnss_valid, gnss_sigma = _gnss_stream(cfg, truth, deg, rng)
    kf_t, kf_idx = _frame_times(cfg, truth)
    frames = _camera_stream(cfg, truth, world, deg, cam, kf_t, kf_idx, rng)

    sensors = SensorData(
        t=truth.t, accel=accel, gyro=gyro,
        t_gnss=t_gnss, gnss_index=gnss_idx, gnss_pos=gnss_pos,
        gnss_valid=gnss_valid, gnss_sigma=gnss_sigma,
        frames=frames, t_kf=kf_t, kf_index=kf_idx,
        pass_id=truth.pass_id, bias_accel=b_a, bias_gyro=b_g,
    )
    return Trial(cfg=cfg, seed=seed, world=world, truth=truth,
                 degradation=deg, sensors=sensors, camera=cam)
