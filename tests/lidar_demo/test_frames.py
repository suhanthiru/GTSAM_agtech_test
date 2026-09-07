"""Frame conventions.

These tests are deliberately about *meaning*, not about arithmetic: a sign error
in the extrinsic convention would produce a demo that looks plausible and is
wrong, and no downstream test would catch it.
"""

import numpy as np
import pytest

from lidar_demo import frames as F


def test_nominal_pitch_tilts_the_sensor_forward_and_down():
    # In FLU, the nominal mount is a 35 degree nadir tilt *forward*.  If this
    # sign flips, the sensor looks backward and up and the whole survey misses.
    R = F.rpy_zyx_to_R(0.0, 35.0, 0.0)
    x_body = R @ np.array([1.0, 0.0, 0.0])
    assert x_body[0] > 0.0            # forward
    assert x_body[2] < 0.0            # and down
    assert x_body[0] == pytest.approx(np.cos(np.radians(35.0)), abs=1e-9)
    assert x_body[2] == pytest.approx(-np.sin(np.radians(35.0)), abs=1e-9)


def test_rpy_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        rpy = rng.uniform([-180, -80, -180], [180, 80, 180])
        back = F.R_to_rpy_zyx(F.rpy_zyx_to_R(*rpy))
        assert np.allclose(F.rpy_zyx_to_R(*back), F.rpy_zyx_to_R(*rpy), atol=1e-9)


def test_true_extrinsic_offset_is_a_couple_of_degrees():
    # The demo's premise: the mounting error is small enough to be believable
    # and large enough to wreck the map.
    nom = F.rpy_zyx_to_R(0.0, 35.0, 0.0)
    true = F.rpy_zyx_to_R(0.8, 36.5, 2.0)
    assert 1.0 < F.angle_between_deg(nom, true) < 4.0


def test_exp_log_so3_roundtrip_including_near_pi():
    rng = np.random.default_rng(1)
    for scale in (1e-9, 1e-4, 1.0, np.pi - 1e-7):
        for _ in range(50):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            w = axis * scale
            back = F.log_so3(F.exp_so3(w))
            assert np.linalg.norm(back - w) < 1e-6 or np.linalg.norm(back + w) < 1e-6


def test_batch_matches_scalar():
    rng = np.random.default_rng(2)
    w = rng.normal(scale=0.4, size=(32, 3))
    Rb = F.exp_so3_batch(w)
    for i in range(w.shape[0]):
        assert np.allclose(Rb[i], F.exp_so3(w[i]), atol=1e-12)
    assert np.allclose(F.log_so3_batch(Rb), w, atol=1e-9)


def test_angle_between_is_convention_free():
    rng = np.random.default_rng(3)
    R1 = F.exp_so3(rng.normal(size=3))
    w = rng.normal(size=3)
    w = 0.05 * w / np.linalg.norm(w)
    R2 = R1 @ F.exp_so3(w)
    # Same answer whichever side the perturbation is composed on.
    assert F.angle_between_deg(R1, R2) == pytest.approx(np.degrees(0.05), abs=1e-9)
    assert F.angle_between_deg(R1, F.exp_so3(w) @ R1) == pytest.approx(np.degrees(0.05), abs=1e-9)


def test_compose_invert_roundtrip():
    rng = np.random.default_rng(4)
    R, t = F.exp_so3(rng.normal(size=3)), rng.normal(size=3)
    Ri, ti = F.invert(R, t)
    Rc, tc = F.compose(R, t, Ri, ti)
    assert np.allclose(Rc, np.eye(3), atol=1e-12)
    assert np.allclose(tc, 0.0, atol=1e-12)


def test_beam_dirs_are_unit_and_ordered():
    az = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    el = np.radians(np.linspace(20.0, -20.0, 8))
    d = F.beam_dirs(az, el)
    assert d.shape == (8, 16, 3)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-12)
    # ring 0 is the top of the fan, azimuth 0 points along sensor +x
    assert d[0, 0, 2] > 0.0
    assert d[-1, 0, 2] < 0.0
    assert d[3, 0, 1] == pytest.approx(0.0, abs=1e-12)


def test_interp_poses_hits_the_knots_and_clamps():
    t = np.linspace(0.0, 1.0, 11)
    p = np.stack([t, 2 * t, -t], axis=1)
    R = np.stack([F.exp_so3(np.array([0.0, 0.0, 0.7 * ti])) for ti in t])
    Rq, pq = F.interp_poses(t, p, R, t)
    assert np.allclose(pq, p, atol=1e-12)
    assert np.allclose(Rq, R, atol=1e-9)

    # midpoint of a constant-rate yaw is the half angle
    Rm, pm = F.interp_poses(t, p, R, np.array([0.05]))
    assert np.allclose(pm[0], [0.05, 0.10, -0.05], atol=1e-12)
    assert F.log_so3(Rm[0])[2] == pytest.approx(0.7 * 0.05, abs=1e-9)

    # outside the range it clamps rather than extrapolating
    _, pe = F.interp_poses(t, p, R, np.array([-1.0, 5.0]))
    assert np.allclose(pe[0], p[0], atol=1e-12)
    assert np.allclose(pe[1], p[-1], atol=1e-12)
