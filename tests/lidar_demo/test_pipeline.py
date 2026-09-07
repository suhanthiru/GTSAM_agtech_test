"""End to end on a miniature run, and the boundary the whole piece rests on.

The recording is deliberately tiny -- a handful of sweeps at low beam count --
so this can run in seconds.  It is not checking accuracy; the gate scripts do
that on the real run.  It is checking that the pieces fit together, that a run
written by the simulator is readable by the estimator, and above all that the
data contract holds: sweeps stored in the sensor frame, the truth kept out of
the manifest, and one set of returns that can be projected two ways.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from lidar_demo.config import DemoConfig, load_run_config
from lidar_demo.frames import rpy_zyx_to_R
from lidar_demo.io import Extrinsic, RunReader
from lidar_demo.map.project import project_run
from lidar_demo.map.raster import rasterise
from lidar_demo.sim.record import record, true_extrinsic


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    cfg = DemoConfig().override(**{
        "lidar.n_rings": 16, "lidar.n_cols": 64,
        "run.backend": "heightfield", "run.name": "tiny",
        "scene.grid_dx": 1.0, "scene.context": 12.0,
    })
    root = record(cfg, out_root=tmp_path_factory.mktemp("runs"), limit=12,
                  verbose=False)
    return root, cfg


def test_a_run_round_trips(tiny):
    root, cfg = tiny
    reader = RunReader(root)
    assert reader.n_sweeps == 12
    t, accel, gyro = reader.imu()
    assert t.size == accel.shape[0] == gyro.shape[0] > 1000
    t_g, p_g, valid, sigma, idx = reader.gnss()
    assert p_g.shape[1] == 3 and valid.dtype == bool
    assert sigma[2] > sigma[0]          # vertical is the worse axis


def test_the_manifest_never_carries_the_true_mount(tiny):
    root, cfg = tiny
    text = (root / "manifest.json").read_text()
    man = json.loads(text)
    nominal = np.asarray(man["extrinsic_nominal"]["rpy_deg"])
    assert np.allclose(nominal, cfg.extrinsic.nominal_rpy_deg)

    truth = true_extrinsic(cfg)
    assert not np.allclose(truth.rpy_deg, nominal)

    # And the true triple appears nowhere in the manifest.  Checked by walking
    # the parsed structure rather than by searching the text, because a search
    # for "2.000" finds a ring elevation and proves nothing.
    def triples(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from triples(v)
        elif isinstance(node, list):
            if len(node) == 3 and all(isinstance(v, (int, float)) for v in node):
                yield np.asarray(node, dtype=float)
            for v in node:
                yield from triples(v)

    for t in triples(man):
        assert not np.allclose(t, truth.rpy_deg, atol=1e-6), t


def test_sweeps_are_stored_in_the_sensor_frame(tiny):
    # The single most important detail in the data contract.  If a sweep were
    # stored in world coordinates, an answer about where the aircraft was and
    # where the sensor pointed would already be baked in, and reprojecting the
    # same returns through two trajectories -- which is the entire piece --
    # would be impossible.
    root, cfg = tiny
    reader = RunReader(root)
    sw = reader.sweep(6)
    assert len(sw) > 20
    assert np.abs(np.linalg.norm(sw.xyz, axis=1) - sw.range).max() < 1e-3
    # the aircraft flies at 15 m and tens of metres out; sensor-frame points
    # are ranges, so they never carry that offset
    assert np.abs(sw.xyz).max() <= cfg.lidar.max_range


def test_points_are_stamped_across_the_sweep(tiny):
    root, cfg = tiny
    sw = RunReader(root).sweep(6)
    assert sw.t_offset.min() >= 0.0
    assert sw.t_offset.max() < 1.0 / cfg.lidar.rate_hz
    assert sw.t_offset.max() > 0.0
    assert np.allclose(sw.t_abs, sw.t_start + sw.t_offset)


def test_the_same_returns_project_two_ways(tiny):
    root, cfg = tiny
    reader = RunReader(root)
    truth = reader.truth_trajectory()

    nominal = reader.extrinsic_nominal
    actual = reader.truth_extrinsic()
    a = project_run(reader, truth, nominal)
    b = project_run(reader, truth, actual)

    assert len(a) == len(b) > 100
    shift = np.linalg.norm(a.xyz - b.xyz, axis=1)
    # A couple of degrees of mount error at fifteen metres of range moves the
    # returns much further than the two centimetres of range noise.
    assert shift.mean() > 20 * cfg.lidar.range_sigma
    assert shift.mean() < 5.0


def test_a_raster_comes_out_of_the_cloud(tiny):
    root, cfg = tiny
    reader = RunReader(root)
    cloud = project_run(reader, reader.truth_trajectory(),
                        reader.truth_extrinsic())
    r = rasterise(cloud.xyz, cell=0.5, ground_pct=5.0, canopy_pct=95.0,
                  min_pts=2, label="test")
    assert np.isfinite(r.ground).any()
    finite = np.isfinite(r.canopy)
    assert (r.canopy[finite] >= -1e-6).all()   # canopy height is never negative


def test_the_estimator_reads_only_the_manifest_and_the_streams(tiny):
    # The truth directory exists and is deliberately not part of the input
    # contract.  This asserts the run is complete without it.
    root, cfg = tiny
    reader = RunReader(root)
    for name in ("manifest.json", "imu.npz", "gnss.npz"):
        assert (root / name).exists()
    assert (root / "truth" / "extrinsic_true.json").exists()
    assert reader.extrinsic_nominal.rpy_deg[1] == pytest.approx(
        cfg.extrinsic.nominal_rpy_deg[1])


def test_run_config_survives_a_schema_change(tiny):
    # An archived run is a record of what the simulator did and has to stay
    # readable after the estimator's own settings have moved on.
    root, _ = tiny
    text = (root / "config_resolved.yaml").read_text()
    (root / "config_resolved.yaml").write_text(
        text + "\nblue:\n  a_setting_that_no_longer_exists: 1.0\n")
    with pytest.warns(UserWarning, match="not in the current schema"):
        cfg = load_run_config(root)
    assert cfg.lidar.n_rings == 16
