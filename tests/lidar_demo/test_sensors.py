"""IMU and GNSS.

The drift band is a design target, not an incidental property: too little and
blue looks like green, too much and the blue cloud is uniform mush with no
readable row structure.  It is asserted here so a config edit cannot silently
walk out of the band.
"""

import numpy as np
import pytest

from lidar_demo.config import DemoConfig
from lidar_demo.sim.flight import plan_survey, truth_on_clock
from lidar_demo.sim.gnss import dropout_summary, simulate_gnss
from lidar_demo.sim.imu import drift_report, predicted_drift, simulate_imu


@pytest.fixture(scope="module")
def bundle():
    cfg = DemoConfig()
    path = plan_survey(cfg.flight, cfg.scene)
    truth = truth_on_clock(path, cfg.imu.rate_hz)
    seeds = np.random.SeedSequence(cfg.run.seed).spawn(4)
    accel, gyro, b_a, b_g = simulate_imu(cfg.imu, truth, np.random.default_rng(seeds[1]))
    gnss = simulate_gnss(cfg.gnss, truth, cfg.scene.treeline_y,
                         np.random.default_rng(seeds[2]))
    return cfg, path, truth, accel, gyro, b_a, b_g, gnss


def test_specific_force_reads_gravity_when_level(bundle):
    cfg, _, truth, accel, _, b_a, _, _ = bundle
    # On a level leg the accelerometer should read about +g along body z: it
    # measures specific force, which is thrust, not acceleration.
    level = np.abs(truth.acceleration).max(axis=1) < 0.1
    a = (accel - b_a)[level]
    assert a[:, 2].mean() == pytest.approx(9.80665, abs=0.15)
    assert abs(a[:, 0].mean()) < 0.3


def test_gyro_tracks_the_true_body_rates(bundle):
    _, _, truth, _, gyro, _, b_g, _ = bundle
    resid = gyro - b_g - truth.omega
    assert np.abs(resid).mean() < 0.01


def test_bias_random_walk_grows(bundle):
    _, _, _, _, _, b_a, b_g, _ = bundle
    for b in (b_a, b_g):
        early = np.abs(b[:2000] - b[0]).mean()
        late = np.abs(b[-2000:] - b[0]).mean()
        assert late > early


def test_dead_reckoned_drift_is_in_the_target_band(bundle):
    cfg, _, truth, accel, gyro, b_a, b_g, _ = bundle
    rep = drift_report(cfg.imu, truth, accel, gyro, b_a, b_g)
    # Spec 4.2: 1-2 m XY and 0.5-1 m Z, read over the GNSS-denied window.
    # The band is widened a little because a single seed's median wanders.
    assert 0.8 <= rep.xy_median <= 2.5, rep
    assert 0.3 <= rep.z_median <= 1.3, rep


def test_closed_form_budget_agrees_with_the_measurement(bundle):
    cfg, _, truth, accel, gyro, b_a, b_g, _ = bundle
    rep = drift_report(cfg.imu, truth, accel, gyro, b_a, b_g)
    # Within a factor of two of the analytic floor; the measurement includes
    # attitude coupling the closed form ignores.
    assert 0.4 < rep.xy_median / rep.predicted_xy < 2.0


def test_whole_flight_dead_reckoning_is_hopeless(bundle):
    cfg, path, _, _, _, _, _, _ = bundle
    # Stated explicitly because it is why the blue trajectory keeps a GNSS pull
    # rather than being pure dead reckoning.
    budget = predicted_drift(cfg.imu, path.duration)
    assert np.sqrt(sum(v ** 2 for v in budget.values())) > 100.0


def test_gnss_vertical_sigma_is_worse_than_horizontal(bundle):
    _, _, _, _, _, _, _, (t, idx, p, valid, sigma) = bundle
    assert sigma[2] >= 2.0 * sigma[0]


def test_gnss_error_matches_its_stated_sigma(bundle):
    _, _, truth, _, _, _, _, (t, idx, p, valid, sigma) = bundle
    err = p[valid] - truth.position[idx][valid]
    assert np.abs(err.std(axis=0) / sigma - 1.0).max() < 0.25


def test_gnss_drops_out_over_the_treeline_and_nowhere_else(bundle):
    cfg, _, truth, _, _, _, _, (t, idx, p, valid, _) = bundle
    y = truth.position[idx][:, 1]
    near = np.abs(y - cfg.scene.treeline_y) < cfg.gnss.dropout_radius
    assert (~valid[near]).all()
    # away from the trees only the small random loss remains
    assert (~valid[~near]).mean() < 3.0 * cfg.gnss.random_dropout + 0.01


def test_the_outage_is_long_enough_to_matter(bundle):
    _, _, _, _, _, _, _, (t, idx, p, valid, _) = bundle
    ds = dropout_summary(t, valid)
    # A whole pass without GNSS: long enough that the graph has to lean on scan
    # matching, which is the point of routing a pass along the treeline.
    assert ds["longest_s"] > 15.0
    assert ds["fraction"] < 0.35
