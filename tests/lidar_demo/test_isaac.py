"""The Isaac backend.

These need Kit running, so they are skipped everywhere except inside the Isaac
environment.  Run them with::

    scripts/isaacpy -m pytest tests/lidar_demo/test_isaac.py -q

The point of the Isaac backend is not that it traces rays -- three backends do
that -- but that it traces them against a *USD stage* built from the same scene,
so the geometry the beauty pass will render and the geometry the LiDAR measures
are one description rather than two that can drift apart.  So most of what is
checked here is the round trip through USD.
"""

import numpy as np
import pytest

pytest.importorskip("isaacsim")
pytest.importorskip("warp")

try:  # pragma: no cover - depends on how the process was started
    import omni.usd  # noqa: F401
    _KIT = True
except Exception:  # pragma: no cover
    _KIT = False

pytestmark = pytest.mark.skipif(
    not _KIT, reason="needs a running SimulationApp; use scripts/isaacpy")

from lidar_demo.config import DemoConfig                     # noqa: E402
from lidar_demo.sim.isaac import stage as stage_mod          # noqa: E402
from lidar_demo.sim.isaac.backend import IsaacLabBackend     # noqa: E402
from lidar_demo.sim.scene import build_scene                 # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    # A coarse scene: the round trip under test does not care about resolution.
    return DemoConfig().override(**{"scene.grid_dx": 2.0, "scene.context": 12.0})


@pytest.fixture(scope="module")
def scene(cfg):
    return build_scene(cfg.scene, np.random.default_rng(0))


@pytest.fixture(scope="module")
def built(scene):
    b = IsaacLabBackend()
    b.build(scene)
    return b


def test_the_stage_holds_both_layers(built):
    layout = built.layout
    assert layout["surface"] and layout["ground"]
    assert all(p.startswith("/World/Farm/Surface/") for p in layout["surface"])
    assert all(p.startswith("/World/Farm/Ground/") for p in layout["ground"])
    assert built.triangles("surface") > 1000
    assert built.triangles("ground") > 1000


def test_geometry_survives_the_usd_round_trip(scene, built):
    # What is cast against is read back out of the stage, so the two had better
    # describe the same field.
    V, F, cls = stage_mod.read_layer(built.stage, built.layout["surface"])
    Vs, Fs, Cs = scene.merged()
    assert F.shape[0] == Fs.shape[0]
    assert V.shape[0] == Vs.shape[0]
    assert np.allclose(np.sort(V[:, 2])[::100], np.sort(Vs[:, 2])[::100], atol=1e-3)
    assert set(np.unique(cls)) == set(np.unique(Cs))


def test_a_ray_straight_down_hits_the_ground(built, scene, cfg):
    from lidar_demo.sim.terrain import sample_bilinear

    xy = np.array([[40.0, 30.0], [70.0, 55.0], [20.0, 65.0]], np.float32)
    origins = np.c_[xy, np.full(len(xy), 60.0)].astype(np.float32)
    dirs = np.tile(np.array([[0.0, 0.0, -1.0]], np.float32), (len(xy), 1))

    hit = built.cast(origins, dirs, "surface")
    assert np.isfinite(hit.t_hit).all()
    z = origins[:, 2] - hit.t_hit
    expect = sample_bilinear(scene.hf, xy, "surface")
    assert np.abs(z - expect).max() < 0.05
    assert (hit.normal[:, 2] > 0.5).all()
    assert (hit.class_id > 0).all()


def test_the_ground_layer_sits_under_the_surface_layer(built, scene):
    # The bare-earth layer is what a pulse meets after filtering through the
    # crop, so it can never be above the canopy top.
    rng = np.random.default_rng(1)
    xy = rng.uniform([5, 5], [110, 75], size=(200, 2)).astype(np.float32)
    origins = np.c_[xy, np.full(len(xy), 60.0)].astype(np.float32)
    dirs = np.tile(np.array([[0.0, 0.0, -1.0]], np.float32), (len(xy), 1))

    top = built.cast(origins, dirs, "surface")
    bot = built.cast(origins, dirs, "ground")
    ok = np.isfinite(top.t_hit) & np.isfinite(bot.t_hit)
    assert ok.sum() > 100
    assert (bot.t_hit[ok] >= top.t_hit[ok] - 1e-3).all()
    assert (bot.class_id[ok] > 0).all()


def test_a_ray_into_the_sky_misses(built):
    origins = np.array([[60.0, 40.0, 60.0]], np.float32)
    dirs = np.array([[0.0, 0.0, 1.0]], np.float32)
    hit = built.cast(origins, dirs, "surface")
    assert not np.isfinite(hit.t_hit[0])
    assert hit.prim_id[0] == -1
    assert hit.class_id[0] == 0


def test_it_agrees_with_the_offline_caster(built, scene):
    # The check that matters: two independent tracers, one warp against USD and
    # one Embree against the numpy meshes, over the same geometry.
    from lidar_demo.sim.backend import make_backend

    other = make_backend("open3d")
    other.build(scene)

    rng = np.random.default_rng(2)
    xy = rng.uniform([2, 2], [115, 80], size=(4000, 2)).astype(np.float32)
    origins = np.c_[xy, np.full(len(xy), 55.0)].astype(np.float32)
    d = rng.normal(size=(len(xy), 3)).astype(np.float32)
    d[:, 2] = -np.abs(d[:, 2]) - 1.0
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    a = built.cast(origins, d, "surface")
    b = other.cast(origins, d, "surface")
    both = np.isfinite(a.t_hit) & np.isfinite(b.t_hit)
    assert both.sum() > 2000
    assert np.abs(a.t_hit[both] - b.t_hit[both]).max() < 0.01
    # and they agree about what was hit, not only about how far away it was
    assert (a.class_id[both] == b.class_id[both]).mean() > 0.99


def test_the_stage_records_its_units(built):
    from pxr import UsdGeom

    assert UsdGeom.GetStageUpAxis(built.stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(built.stage) == pytest.approx(1.0)
