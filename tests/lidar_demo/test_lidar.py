"""LiDAR geometry, the rolling shutter, and the storage convention.

The convention test is the one that protects the whole demo.  If sweeps were
ever stored in the world frame, an answer about where the drone was and where
the sensor pointed would be baked into the data, and reprojecting the same
returns through two trajectories -- which is the entire piece -- would be
impossible.
"""

import numpy as np
import pytest

from lidar_demo.config import DemoConfig
from lidar_demo.frames import rpy_zyx_to_R
from lidar_demo.sim.backend import HeightfieldBackend, RayHits
from lidar_demo.sim.flight import plan_survey
from lidar_demo.sim.lidar import (LidarModel, project_sweep_truth,
                                  simulate_sweep)
from lidar_demo.sim.scene import build_scene
from lidar_demo.sim.terrain import sample_bilinear


@pytest.fixture(scope="module")
def cfg():
    # A small, fast scene: the geometry under test does not care about size.
    c = DemoConfig()
    return c.override(**{"lidar.n_rings": 24, "lidar.n_cols": 96})


@pytest.fixture(scope="module")
def scene(cfg):
    return build_scene(cfg.scene, np.random.default_rng(0))


@pytest.fixture(scope="module")
def backend(scene):
    b = HeightfieldBackend(step=0.6)
    b.build(scene)
    return b


def test_beam_table_covers_the_stated_field_of_view(cfg):
    m = LidarModel.from_cfg(cfg.lidar)
    assert np.degrees(m.ring_el).max() == pytest.approx(max(cfg.lidar.vfov_deg))
    assert np.degrees(m.ring_el).min() == pytest.approx(min(cfg.lidar.vfov_deg))
    assert np.degrees(m.col_az).max() < 360.0
    assert m.dirs_S.shape == (cfg.lidar.n_rings, cfg.lidar.n_cols, 3)
    assert np.allclose(np.linalg.norm(m.dirs_S, axis=-1), 1.0)


def test_columns_are_stamped_across_the_sweep(cfg):
    m = LidarModel.from_cfg(cfg.lidar)
    assert m.col_dt[0] == 0.0
    assert m.col_dt[-1] < m.period
    assert np.allclose(np.diff(m.col_dt), m.period / cfg.lidar.n_cols)


def test_ray_march_matches_an_analytic_plane():
    # A flat surface at a known height: marched ranges must match ray-plane
    # intersection to well under the range noise.
    from lidar_demo.sim.terrain import Heightfield

    z0 = 3.0
    n = 40
    hf = Heightfield(x0=-50.0, y0=-50.0, dx=2.0,
                     z_terrain=np.full((n, n), z0, np.float32),
                     z_surface=np.full((n, n), z0, np.float32),
                     canopy_h=np.zeros((n, n), np.float32),
                     class_id=np.ones((n, n), np.uint8),
                     ditch_p0=(0.0, 0.0), ditch_p1=(1.0, 1.0))

    class _S:
        pass

    s = _S()
    s.hf = hf
    b = HeightfieldBackend(step=0.4)
    b.build(s)

    origin = np.array([0.0, 0.0, 20.0])
    el = np.radians(np.array([-90.0, -70.0, -50.0, -30.0]))
    dirs = np.stack([np.cos(el), np.zeros_like(el), np.sin(el)], axis=1)
    hits = b.cast(np.repeat(origin[None], 4, axis=0), dirs)
    expected = (z0 - origin[2]) / dirs[:, 2]
    assert np.abs(hits.t_hit - expected).max() < 1e-3
    assert np.allclose(hits.normal[:, 2], 1.0, atol=1e-6)


def test_points_are_stored_in_the_sensor_frame_not_the_world(cfg, backend):
    path = plan_survey(cfg.flight, cfg.scene)
    m = LidarModel.from_cfg(cfg.lidar)
    R_BS = rpy_zyx_to_R(*cfg.extrinsic.nominal_rpy_deg)
    t_BS = np.array(cfg.extrinsic.nominal_t)
    sw = simulate_sweep(m, backend, path, 400, R_BS, t_BS, cfg.lidar,
                        np.random.default_rng(0))
    assert len(sw.xyz) > 30

    # Every stored point is its range along its own beam direction, in sensor
    # coordinates.  Nothing about the aircraft's world position is in the file.
    d = m.dirs_S[sw.ring.astype(int), sw.col.astype(int)]
    assert np.abs(np.linalg.norm(sw.xyz, axis=1) - sw.range).max() < 1e-3
    unit = sw.xyz / np.linalg.norm(sw.xyz, axis=1, keepdims=True)
    assert np.abs(np.einsum("ij,ij->i", unit, d) - 1.0).max() < 1e-4

    # The aircraft is 15 m up and 60 m out; a world-frame store would show it.
    assert np.abs(sw.xyz).max() < cfg.lidar.max_range
    assert np.abs(sw.xyz.mean(axis=0)).max() < 60.0


def test_projection_through_the_true_extrinsic_lands_on_the_terrain(cfg, backend, scene):
    # Gate 1 in miniature.  Projected through the mount the simulator actually
    # built, the returns have to sit on the surface the rays hit.
    path = plan_survey(cfg.flight, cfg.scene)
    m = LidarModel.from_cfg(cfg.lidar)
    R_true = rpy_zyx_to_R(0.8, 36.5, 2.0)
    t_BS = np.array(cfg.extrinsic.nominal_t)

    d_terr, d_surf, ground = [], [], []
    for k in range(380, 420):
        sw = simulate_sweep(m, backend, path, k, R_true, t_BS, cfg.lidar,
                            np.random.default_rng(k))
        P = project_sweep_truth(sw, R_true, t_BS)
        d_terr.append(P[:, 2] - sample_bilinear(scene.hf, P[:, :2], "terrain"))
        d_surf.append(P[:, 2] - sample_bilinear(scene.hf, P[:, :2], "surface"))
        ground.append(sw.hit_class == 1)
    d_terr = np.concatenate(d_terr)
    d_surf = np.concatenate(d_surf)
    ground = np.concatenate(ground)

    # A shot classified as ground got all the way to the soil, so it sits on the
    # bare-earth surface to within the range noise.  A canopy return stopped
    # inside the crop, so it reads below the canopy top and never above it.
    assert ground.sum() > 50
    assert np.abs(d_terr[ground]).mean() < 0.05
    assert d_surf[~ground].mean() < 0.02


def test_the_nominal_extrinsic_puts_the_points_in_the_wrong_place(cfg, backend, scene):
    # The premise of the whole demo: a couple of degrees of mount error moves
    # the returns much further than the range noise does.
    path = plan_survey(cfg.flight, cfg.scene)
    m = LidarModel.from_cfg(cfg.lidar)
    R_true = rpy_zyx_to_R(0.8, 36.5, 2.0)
    R_nom = rpy_zyx_to_R(*cfg.extrinsic.nominal_rpy_deg)
    t_BS = np.array(cfg.extrinsic.nominal_t)
    sw = simulate_sweep(m, backend, path, 400, R_true, t_BS, cfg.lidar,
                        np.random.default_rng(2))

    P_true = project_sweep_truth(sw, R_true, t_BS)
    P_nom = project_sweep_truth(sw, R_nom, t_BS)
    shift = np.linalg.norm(P_true - P_nom, axis=1)
    assert shift.mean() > 10.0 * cfg.lidar.range_sigma


def test_the_sweep_is_motion_smeared(cfg, backend):
    # Over 100 ms at cruise the aircraft moves half a metre.  The per-column
    # true poses must show that, otherwise there is nothing for deskewing to fix.
    path = plan_survey(cfg.flight, cfg.scene)
    m = LidarModel.from_cfg(cfg.lidar)
    R_BS = rpy_zyx_to_R(*cfg.extrinsic.nominal_rpy_deg)
    sw = simulate_sweep(m, backend, path, 400, R_BS, np.array(cfg.extrinsic.nominal_t),
                        cfg.lidar, np.random.default_rng(3))
    travel = np.linalg.norm(sw.col_p_WB[-1] - sw.col_p_WB[0])
    expected = cfg.flight.speed * m.period * (m.n_cols - 1) / m.n_cols
    assert travel == pytest.approx(expected, rel=0.15)
    assert travel > 0.3


def test_dropout_and_intensity_are_physical(cfg, backend):
    path = plan_survey(cfg.flight, cfg.scene)
    m = LidarModel.from_cfg(cfg.lidar)
    R_BS = rpy_zyx_to_R(*cfg.extrinsic.nominal_rpy_deg)
    sw = simulate_sweep(m, backend, path, 400, R_BS, np.array(cfg.extrinsic.nominal_t),
                        cfg.lidar, np.random.default_rng(4))
    assert (sw.intensity >= 0.0).all() and (sw.intensity <= 1.0).all()
    assert (sw.range >= cfg.lidar.min_range).all()
    assert (sw.range <= cfg.lidar.max_range + 1.0).all()
    # some rays miss or are dropped; the sweep is not a full grid
    assert len(sw.xyz) < m.n_rings * m.n_cols
