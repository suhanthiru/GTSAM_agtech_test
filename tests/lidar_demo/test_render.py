"""Render maths, and the properties the film depends on.

Nothing here draws anything.  The parts that matter are the ones that would be
wrong in a way a still frame does not reveal: whether the morph is continuous,
whether it actually ends where it should, and whether the section is measuring
height against the ground rather than against the top of the crop.
"""

import numpy as np
import pytest

# The render layer runs on the Windows host where pyvista lives; the estimator
# side has gtsam and no renderer, so these skip cleanly there.
pytest.importorskip("pyvista")

from lidar_demo.render.camera import CameraKey, edge_on, interp, three_quarter  # noqa: E402
from lidar_demo.render.crosssection import (bare_earth, flat_reference,  # noqa: E402
                                            flatten, slab)
from lidar_demo.render.morph import (morph_colour, morph_positions,  # noqa: E402
                                     smoothness, stagger_phase)
from lidar_demo.render.scene import dissolve, hex_rgb, smoothstep  # noqa: E402


# ---------------------------------------------------------------------------
# the collapse
# ---------------------------------------------------------------------------


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    blue = rng.normal(scale=8.0, size=(4000, 3)).astype(np.float32)
    green = blue + rng.normal(scale=0.4, size=blue.shape).astype(np.float32)
    return blue, green


def test_morph_starts_at_blue_and_ends_at_green(pair):
    blue, green = pair
    phase = stagger_phase(blue.shape[0], 0.3, 0)
    assert np.allclose(morph_positions(blue, green, 0.0, phase), blue, atol=1e-5)
    assert np.allclose(morph_positions(blue, green, 1.0, phase), green, atol=1e-5)


def test_every_point_moves_monotonically(pair):
    # A point that advances and then retreats reads as a wobble, not a collapse.
    blue, green = pair
    phase = stagger_phase(blue.shape[0], 0.3, 1)
    prev = np.zeros(blue.shape[0])
    total = np.linalg.norm(green - blue, axis=1)
    moving = total > 1e-6
    for k in range(41):
        cur = morph_positions(blue, green, k / 40, phase)
        travelled = np.linalg.norm(cur - blue, axis=1)[moving] / total[moving]
        assert (travelled >= prev[moving] - 1e-6).all()
        prev = np.zeros(blue.shape[0])
        prev[moving] = travelled


def test_morph_is_smooth_enough_to_read_as_a_move(pair):
    # Gate 6.  A cloud that covers most of its distance in one frame is a cut
    # with extra steps.
    blue, green = pair
    rep = smoothness(blue, green, 45, 0.3)
    assert rep["pass"], rep
    assert rep["max_frame_fraction"] < 0.2


def test_morph_colour_runs_from_blue_to_green(pair):
    blue, green = pair
    phase = np.zeros(blue.shape[0], dtype=np.float32)
    start = morph_colour("#2E6FD6", "#1D9E75", 0.0, phase)
    end = morph_colour("#2E6FD6", "#1D9E75", 1.0, phase)
    assert np.allclose(start[0], hex_rgb("#2E6FD6"), atol=1)
    assert np.allclose(end[0], hex_rgb("#1D9E75"), atol=1)


def test_stagger_is_fixed_for_a_given_seed(pair):
    blue, _ = pair
    a = stagger_phase(blue.shape[0], 0.3, 7)
    b = stagger_phase(blue.shape[0], 0.3, 7)
    assert np.array_equal(a, b)
    assert 0.0 <= a.min() and a.max() <= 0.3


# ---------------------------------------------------------------------------
# the section
# ---------------------------------------------------------------------------


class _Flat:
    """A heightfield-shaped stand-in with a known, tilted ground surface."""

    def __init__(self, slope=0.05):
        self.slope = slope


def _fake_sample(hf, xy, which="terrain"):
    return hf.slope * np.asarray(xy)[:, 1]


def test_flatten_measures_height_above_the_ground(monkeypatch):
    import lidar_demo.sim.terrain as terrain

    monkeypatch.setattr(terrain, "sample_bilinear", _fake_sample)
    hf = _Flat(0.05)
    y = np.linspace(0.0, 40.0, 200)
    pts = np.stack([np.full_like(y, 60.0), y, 0.05 * y + 0.25], axis=1)
    out = flatten(pts, hf, exaggeration=1.0, x0=60.0)
    # A cloud sitting a constant 25 cm above a sloping ground flattens to a
    # constant 25 cm, which is the whole point of the view.
    assert np.allclose(out[:, 2], 0.25, atol=1e-6)
    assert np.allclose(out[:, 0], 60.0)


def test_flatten_applies_the_exaggeration(monkeypatch):
    import lidar_demo.sim.terrain as terrain

    monkeypatch.setattr(terrain, "sample_bilinear", _fake_sample)
    hf = _Flat(0.0)
    pts = np.array([[60.0, 5.0, 0.3]])
    assert flatten(pts, hf, 6.0, 60.0)[0, 2] == pytest.approx(1.8)


def test_bare_earth_keeps_the_soil_and_drops_the_crop():
    # A slab of ground with a canopy a metre above it: the filter has to return
    # the ground, because a section drawn from the canopy shows nothing.
    rng = np.random.default_rng(0)
    y = rng.uniform(0.0, 30.0, 6000)
    ground = np.stack([np.full(4000, 60.0), y[:4000],
                       rng.normal(0.0, 0.03, 4000)], axis=1)
    canopy = np.stack([np.full(2000, 60.0), y[4000:],
                       1.2 + rng.normal(0.0, 0.08, 2000)], axis=1)
    pts = np.concatenate([ground, canopy])
    keep = bare_earth(pts)
    assert keep.size > 1000
    assert (pts[keep, 2] < 0.4).mean() > 0.97


def test_slab_takes_a_band_and_can_be_windowed():
    rng = np.random.default_rng(1)
    pts = rng.uniform([-50, -50, -5], [150, 150, 5], size=(5000, 3))
    idx = slab(pts, 60.0, 0.5)
    assert (np.abs(pts[idx, 0] - 60.0) < 0.5).all()
    idx2 = slab(pts, 60.0, 0.5, (20.0, 40.0))
    assert idx2.size <= idx.size
    assert ((pts[idx2, 1] >= 20.0) & (pts[idx2, 1] <= 40.0)).all()


def test_flat_reference_is_a_straight_line_at_zero():
    ref = flat_reference(60.0, 0.0, 84.0, 50)
    assert np.allclose(ref[:, 0], 60.0)
    assert np.allclose(ref[:, 2], 0.0)
    assert ref[0, 1] == 0.0 and ref[-1, 1] == 84.0


# ---------------------------------------------------------------------------
# camera and compositing
# ---------------------------------------------------------------------------


def test_camera_interpolation_starts_and_ends_at_rest():
    a = CameraKey((0, 0, 10), (0, 0, 0))
    b = CameraKey((10, 0, 20), (5, 0, 0))
    assert np.allclose(interp([a, b], 0.0).pos, a.pos)
    assert np.allclose(interp([a, b], 1.0).pos, b.pos)
    # smoothstep eases in and out, so the first step is small
    step0 = np.linalg.norm(np.array(interp([a, b], 0.02).pos) - np.array(a.pos))
    step_mid = np.linalg.norm(np.array(interp([a, b], 0.52).pos)
                              - np.array(interp([a, b], 0.50).pos))
    assert step0 < step_mid


def test_edge_on_camera_is_orthographic_and_level():
    key = edge_on((60.0, 42.0, 0.0), 300.0, 5.0)
    assert key.parallel_scale == 5.0
    assert np.allclose(key.up, (0.0, 0.0, 1.0))
    assert key.pos[0] > key.focal[0]        # looking back along -x


def test_three_quarter_camera_looks_down():
    key = three_quarter((60.0, 42.0, 0.0), 100.0, 30.0, -110.0)
    assert key.pos[2] > key.focal[2]
    assert np.linalg.norm(np.array(key.pos) - np.array(key.focal)) == pytest.approx(100.0)


def test_dissolve_is_a_fade_not_a_cut():
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.full((4, 4, 3), 200, np.uint8)
    assert dissolve(a, b, 0.0).max() == 0
    assert dissolve(a, b, 1.0).min() == 200
    assert 80 < int(dissolve(a, b, 0.5).mean()) < 120


def test_smoothstep_is_flat_at_both_ends():
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(-3.0) == 0.0 and smoothstep(4.0) == 1.0
    assert smoothstep(0.5) == pytest.approx(0.5)
