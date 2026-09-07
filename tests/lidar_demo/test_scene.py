"""Terrain, canopy and props.

The scene has to satisfy two masters at once: it must look like the reference
painting, and it must be geometrically non-degenerate enough that a scan matcher
converges on it.  These tests check the second, which is the one that fails
silently.
"""

import numpy as np
import pytest

from lidar_demo.config import DemoConfig
from lidar_demo.sim import primitives as prim
from lidar_demo.sim.scene import (VERTICAL_MIN_HEIGHT, build_scene,
                                  vertical_coverage)
from lidar_demo.sim.terrain import (build_heightfield, heightfield_to_mesh,
                                    sample_bilinear)


@pytest.fixture(scope="module")
def cfg():
    return DemoConfig()


@pytest.fixture(scope="module")
def scene(cfg):
    return build_scene(cfg.scene, np.random.default_rng(0))


def test_terrain_has_relief_at_every_scale(cfg):
    hf = build_heightfield(cfg.scene, np.random.default_rng(0))
    x, y = hf.x, hf.y
    inside = np.ix_((y >= 0) & (y <= cfg.scene.survey_y),
                    (x >= 0) & (x <= cfg.scene.survey_x))
    z = hf.z_terrain[inside]

    # macro: metres of relief, not centimetres and not tens of metres
    assert 4.0 < float(z.max() - z.min()) < 12.0

    # slope: a consistent trend along the steepest-ascent direction
    phi = np.radians(cfg.scene.slope_azimuth_deg)
    X, Y = np.meshgrid(x[(x >= 0) & (x <= cfg.scene.survey_x)],
                       y[(y >= 0) & (y <= cfg.scene.survey_y)])
    s = X * np.cos(phi) + Y * np.sin(phi)
    fit = np.polyfit(s.ravel(), z.ravel(), 1)[0]
    assert np.degrees(np.arctan(fit)) == pytest.approx(cfg.scene.slope_deg, abs=0.6)

    # furrows: a ridge every row spacing, at centimetre scale.  Sample a column
    # well clear of the ditch, whose 1.5 m step would swamp a 10 cm signal.
    y_in = y[(y >= 0) & (y <= cfg.scene.survey_y)]
    clear = (y_in > 52.0)
    row = z[clear, z.shape[1] // 2]
    detr = row - np.convolve(row, np.ones(41) / 41, mode="same")
    core = detr[40:-40]
    amp = float(np.percentile(core, 97) - np.percentile(core, 3))
    assert 0.05 < amp < 0.45
    period_samples = cfg.scene.row_spacing / hf.dx
    spec = np.abs(np.fft.rfft(core - core.mean()))
    peak = int(np.argmax(spec))
    assert peak == pytest.approx(core.size / period_samples, rel=0.15)


def test_ditch_is_deep_and_local(cfg):
    hf = build_heightfield(cfg.scene, np.random.default_rng(0))
    mid = np.array([[60.0, 40.0]])
    # the ditch runs (0,10)->(120,70), so it passes near (60, 40)
    on = float(sample_bilinear(hf, mid, "terrain")[0])
    off = float(sample_bilinear(hf, mid + np.array([[0.0, 12.0]]), "terrain")[0])
    assert off - on > 1.0
    far = float(sample_bilinear(hf, np.array([[60.0, 5.0]]), "terrain")[0])
    assert abs(far - off) < 4.0     # away from the ditch the surface is ordinary


def test_canopy_has_blocks_including_bare_ground(cfg):
    hf = build_heightfield(cfg.scene, np.random.default_rng(0))
    x, y = hf.x, hf.y
    inside = np.ix_((y >= 0) & (y <= cfg.scene.survey_y),
                    (x >= 0) & (x <= cfg.scene.survey_x))
    c = hf.canopy_h[inside]
    assert (c < 0.05).mean() > 0.10                    # bare patches exist
    assert (c > cfg.scene.canopy_healthy * 0.7).mean() > 0.20   # so do healthy ones
    assert c.max() < cfg.scene.canopy_healthy * 1.3


def test_surface_is_terrain_plus_canopy(cfg):
    hf = build_heightfield(cfg.scene, np.random.default_rng(0))
    assert np.allclose(hf.z_surface, hf.z_terrain + hf.canopy_h, atol=1e-5)


def test_bilinear_sampler_matches_the_grid(cfg):
    hf = build_heightfield(cfg.scene, np.random.default_rng(0))
    j, i = 300, 400
    xy = np.array([[hf.x[i], hf.y[j]]])
    assert sample_bilinear(hf, xy)[0] == pytest.approx(hf.z_terrain[j, i], abs=1e-4)
    # a query off the grid edge clamps instead of exploding
    assert np.isfinite(sample_bilinear(hf, np.array([[-1e4, 1e4]]))[0])


def test_every_close_up_sees_two_verticals(cfg, scene):
    assert vertical_coverage(scene, cfg.scene) >= cfg.scene.min_verticals


def test_props_are_present_and_tall(scene):
    kinds = {p.kind for p in scene.props}
    assert {"tree", "windmill", "building", "scarecrow", "post"} <= kinds
    posts = [p for p in scene.props if p.kind == "post"]
    assert 8 <= len(posts) <= 12
    assert all(p.height >= VERTICAL_MIN_HEIGHT for p in scene.props if p.kind != "post"
               or p.height >= VERTICAL_MIN_HEIGHT)


def test_mesh_is_well_formed(scene):
    V, F, C = scene.merged()
    assert F.min() >= 0 and F.max() < V.shape[0]
    assert C.shape[0] == F.shape[0]
    # no degenerate triangles
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    assert (area > 1e-9).all()


def test_primitives_are_closed_enough_to_stop_a_ray():
    # Every prop primitive must have an even number of triangles crossing any
    # ray through it; a hole would let a beam pass straight through a post.
    for V, F in (prim.cylinder(0.5, 2.0), prim.box(1.0, 2.0, 3.0),
                 prim.ellipsoid(1.0, 1.0, 2.0), prim.frustum(1.0, 0.4, 3.0)):
        edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert (counts == 2).all()


def test_heightfield_mesh_face_classes_line_up(cfg):
    hf = build_heightfield(cfg.scene, np.random.default_rng(0))
    V, F, C = heightfield_to_mesh(hf, "surface", stride=8)
    assert C.shape[0] == F.shape[0]
    assert set(np.unique(C)).issubset({1, 2, 3})
