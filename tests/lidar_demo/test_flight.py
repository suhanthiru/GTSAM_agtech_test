"""The survey pattern.

The alternation test is the important one.  If adjacent passes ever fly the same
direction, the boresight error biases them identically, they agree perfectly with
each other, and it becomes unobservable: the optimiser would return the nominal
mounting angle and the demo would quietly have nothing to show.
"""

import numpy as np
import pytest

from lidar_demo.config import DemoConfig
from lidar_demo.sim.flight import plan_survey, pass_headings, truth_on_clock


@pytest.fixture(scope="module")
def cfg():
    return DemoConfig()


@pytest.fixture(scope="module")
def path(cfg):
    return plan_survey(cfg.flight, cfg.scene)


def test_adjacent_passes_fly_opposite_directions(path, cfg):
    h = pass_headings(path)[:cfg.flight.n_passes]
    for i in range(len(h) - 1):
        assert float(h[i] @ h[i + 1]) == pytest.approx(-1.0, abs=1e-9), (
            f"passes {i} and {i + 1} fly the same way; the boresight would be "
            "unobservable")


def test_closing_pass_is_perpendicular(path, cfg):
    h = pass_headings(path)
    assert abs(float(h[0] @ h[-1])) < 1e-9
    crossing = path.legs[-1]
    assert crossing.p0[0] == pytest.approx(cfg.flight.crossing_x)
    # it runs past both ends of the survey, not just across the middle
    assert crossing.p0[1] < 0.0
    assert crossing.p1[1] > cfg.scene.survey_y


def test_line_spacing_gives_the_planned_sidelap(path, cfg):
    ys = [leg.p0[1] for leg in path.legs[:cfg.flight.n_passes]]
    gaps = np.abs(np.diff(ys))
    assert np.allclose(gaps, cfg.flight.spacing)
    assert cfg.derived.sidelap_frac > 0.4


def test_treeline_pass_is_the_one_next_to_the_trees(path, cfg):
    leg = [lg for lg in path.legs if lg.pass_id == cfg.flight.treeline_pass][0]
    dist = abs(float(leg.p0[1]) - cfg.scene.treeline_y)
    assert dist < cfg.gnss.dropout_radius, (
        "the treeline pass must fly inside the GNSS dropout band, otherwise "
        "there is no GNSS-denied stretch in the run")


def test_position_is_continuous_and_c2(path):
    # Sample either side of every transition and check that position, velocity
    # and acceleration all join up.  A C1 path would put a step in specific
    # force at each turn and hand the estimator a fictitious event.
    t = np.linspace(0.0, path.duration, 40001)
    p, v, a, _, _, _ = path.sample(t)
    dt = t[1] - t[0]
    assert np.abs(np.diff(p, axis=0)).max() < 3.0 * path.speed * dt
    assert np.abs(np.diff(v, axis=0)).max() < 0.3
    assert np.abs(np.diff(a, axis=0)).max() < 0.3


def test_velocity_and_acceleration_match_finite_differences(path):
    t = np.linspace(20.0, path.duration - 20.0, 2000)
    h = 1e-4
    p_plus, _, _, _, _, _ = path.sample(t + h)
    p_minus, _, _, _, _, _ = path.sample(t - h)
    _, v, a, _, _, _ = path.sample(t)
    v_fd = (p_plus - p_minus) / (2 * h)
    assert np.abs(v_fd - v).max() < 1e-4

    _, v_plus, _, _, _, _ = path.sample(t + h)
    _, v_minus, _, _, _, _ = path.sample(t - h)
    a_fd = (v_plus - v_minus) / (2 * h)
    assert np.abs(a_fd - a).max() < 1e-3


def test_cruise_speed_is_held_on_the_legs(path, cfg):
    t = np.linspace(0.0, path.duration, 20000)
    _, v, _, is_turn, _, _ = path.sample(t)
    speed = np.linalg.norm(v[~is_turn], axis=1)
    assert np.abs(speed - cfg.flight.speed).max() < 0.5


def test_attitude_wobbles_but_stays_level(path):
    t = np.linspace(5.0, path.duration - 5.0, 4000)
    R, _ = path.pose(t)
    tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0)))
    assert tilt.max() < 25.0          # turns tilt a multirotor, but not absurdly
    on_leg = tilt[tilt < 5.0]
    assert on_leg.std() > 0.05        # the gust is actually doing something


def test_altitude_and_duration_are_sane(path, cfg):
    t = np.linspace(0.0, path.duration, 5000)
    p, _, _, _, _, _ = path.sample(t)
    assert np.abs(p[:, 2] - cfg.flight.altitude).max() < 1.0
    assert 200.0 < path.duration < 400.0


def test_truth_on_clock_is_consistent(path, cfg):
    truth = truth_on_clock(path, cfg.imu.rate_hz)
    assert len(truth) > 10000
    assert truth.t[1] - truth.t[0] == pytest.approx(1.0 / cfg.imu.rate_hz)
    # rotations are proper
    det = np.linalg.det(truth.rotation)
    assert np.allclose(det, 1.0, atol=1e-8)
    # body rates integrate back to the attitude change over a short window
    i0, i1 = 5000, 5040
    R_rel = truth.rotation[i0].T @ truth.rotation[i1]
    from lidar_demo.frames import log_so3
    dt = truth.t[i1] - truth.t[i0]
    assert np.linalg.norm(log_so3(R_rel) - truth.omega[i0:i1].mean(axis=0) * dt) < 5e-3
