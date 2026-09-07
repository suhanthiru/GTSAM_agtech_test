"""The extrinsic-dependent scan-match factor.

These tests exist because a sign error here fails silently.  The optimiser would
converge, report a tidy result, and hand back something close to the nominal
mounting angle, and every downstream picture would look plausible and be wrong.
Three independent derivations of the same Jacobian are checked against each
other: the chain rule over GTSAM's own derivatives, closed-form adjoints, and
central differences.
"""

import numpy as np
import pytest

gtsam = pytest.importorskip("gtsam")
from gtsam import Pose3, Rot3, Point3                       # noqa: E402
from gtsam.symbol_shorthand import E, X                     # noqa: E402

from lidar_demo.est.factors import (numeric_jacobians,       # noqa: E402
                                    scan_match_error,
                                    scan_match_factor,
                                    scan_match_jacobians_closed_form)


def rand_pose(rng, t_scale=3.0, r_scale=0.5):
    return Pose3(Rot3.Expmap(rng.normal(scale=r_scale, size=3)),
                 Point3(*rng.normal(scale=t_scale, size=3)))


class _Keys:
    """Stands in for the ``this`` argument GTSAM passes to a CustomFactor."""

    def __init__(self, keys):
        self._keys = keys

    def keys(self):
        return self._keys


def evaluate(measured, Ti, Tj, Ee, want_jacobians=True):
    values = gtsam.Values()
    values.insert(X(0), Ti)
    values.insert(X(1), Tj)
    values.insert(E(0), Ee)
    H = [np.zeros((6, 6)) for _ in range(3)] if want_jacobians else None
    r = scan_match_error(measured, _Keys([X(0), X(1), E(0)]), values, H)
    return r, H


def test_residual_is_zero_when_the_measurement_is_consistent():
    rng = np.random.default_rng(0)
    for _ in range(20):
        Ti, Tj, Ee = rand_pose(rng), rand_pose(rng), rand_pose(rng, 0.2, 0.1)
        measured = Ee.inverse().compose(Ti.between(Tj)).compose(Ee)
        r, _ = evaluate(measured, Ti, Tj, Ee)
        assert np.abs(r).max() < 1e-9


def test_chain_rule_matches_central_differences():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(30):
        Ti, Tj, Ee = rand_pose(rng), rand_pose(rng), rand_pose(rng, 0.2, 0.1)
        measured = rand_pose(rng, 0.5, 0.1)
        _, H = evaluate(measured, Ti, Tj, Ee)
        Hn = numeric_jacobians(measured, Ti, Tj, Ee)
        for a, b in zip(H, Hn):
            worst = max(worst, float(np.abs(a - b).max()))
    assert worst < 1e-5, worst


def test_closed_form_matches_the_chain_rule():
    rng = np.random.default_rng(2)
    for _ in range(30):
        Ti, Tj, Ee = rand_pose(rng), rand_pose(rng), rand_pose(rng, 0.2, 0.1)
        measured = rand_pose(rng, 0.5, 0.1)
        _, H = evaluate(measured, Ti, Tj, Ee)
        Hc = scan_match_jacobians_closed_form(measured, Ti, Tj, Ee)
        for a, b in zip(H, Hc):
            assert np.abs(a - b).max() < 1e-8


def test_one_factor_leaves_two_directions_of_the_extrinsic_free():
    # dr/dE is J (I - Ad(B^-1)), and I - Ad(T) is singular on the screw axis of
    # T: a rotation about that axis and a translation along it change nothing.
    # So a single pair always leaves two degrees of freedom of the mounting
    # unconstrained.  That is not a defect, it is the reason the survey pattern
    # matters -- see the next two tests.
    rng = np.random.default_rng(3)
    for _ in range(20):
        Ti, Tj, Ee = rand_pose(rng), rand_pose(rng), rand_pose(rng, 0.2, 0.1)
        measured = rand_pose(rng, 0.5, 0.1)
        _, H = evaluate(measured, Ti, Tj, Ee)
        assert np.linalg.matrix_rank(H[2], tol=1e-8) == 4


NOMINAL = Pose3(Rot3.RzRyRx(0.0, np.radians(35.0), 0.0), Point3(0.1, 0.0, -0.05))
TRUE_MOUNT = Pose3(Rot3.RzRyRx(np.radians(0.8), np.radians(36.5), np.radians(2.0)),
                   Point3(0.1, 0.0, -0.05))


def _toy_survey(alternate, n_legs=4, n=8, step=5.0, spacing=12.0, seed=0):
    """A miniature boustrophedon, optionally with every leg flown the same way."""
    rng = np.random.default_rng(seed)
    poses = []
    for leg in range(n_legs):
        heading = +1 if (not alternate or leg % 2 == 0) else -1
        yaw = 0.0 if heading > 0 else np.pi
        for k in range(n):
            w = rng.normal(scale=0.02, size=3)
            R = Rot3.Expmap(np.array([w[0], w[1], yaw + w[2]]))
            x = heading * step * k + (0.0 if heading > 0 else step * (n - 1))
            poses.append(Pose3(R, Point3(x, leg * spacing, 15.0)))
    return poses


def _solve_toy(alternate, anchor, n_legs=4, n=8):
    """Fit the toy survey and return (mount error in degrees, worst 1-sigma).

    ``anchor`` chooses what pins the trajectory: ``"attitude"`` gives every pose
    a full prior, as the IMU and GNSS together do; ``"position"`` gives position
    only, as GNSS alone does.
    """
    poses = _toy_survey(alternate, n_legs, n)
    seq = [(i, i + 1) for i in range(len(poses) - 1) if (i + 1) % n != 0]
    cross = [(leg * n + k,
              (leg + 1) * n + (k if leg % 2 == 0 else n - 1 - k))
             for leg in range(n_legs - 1) for k in range(n)]

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    for i, pose in enumerate(poses):
        values.insert(X(i), pose)
        if anchor == "attitude":
            graph.add(gtsam.PriorFactorPose3(X(i), pose,
                      gtsam.noiseModel.Diagonal.Sigmas(
                          np.r_[np.full(3, np.radians(0.5)), np.full(3, 0.5)])))
        else:
            graph.add(gtsam.GPSFactor(X(i), pose.translation(),
                      gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 1.0]))))

    sn = gtsam.noiseModel.Diagonal.Sigmas(
        np.r_[np.full(3, np.radians(0.2)), np.full(3, 0.03)])
    for i, j in seq + cross:
        meas = (TRUE_MOUNT.inverse().compose(poses[i].between(poses[j]))
                .compose(TRUE_MOUNT))
        graph.add(scan_match_factor(X(i), X(j), E(0), meas, sn))

    graph.add(gtsam.PriorFactorPose3(E(0), NOMINAL,
              gtsam.noiseModel.Diagonal.Sigmas(
                  np.r_[np.full(3, np.radians(5.0)), np.full(3, 0.02)])))
    values.insert(E(0), NOMINAL)

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(150)
    result = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()

    est = result.atPose3(E(0))
    err = np.degrees(np.linalg.norm(
        Rot3.Logmap(est.rotation().between(TRUE_MOUNT.rotation()))))
    cov = gtsam.Marginals(graph, result).marginalCovariance(E(0))
    sigma = float(np.degrees(np.sqrt(np.diag(cov)[:3])).max())
    return float(err), sigma


NOMINAL_OFFSET_DEG = 2.241     # how far the nominal mount sits from the true one


def test_the_graph_recovers_an_injected_mounting_error():
    err, sigma = _solve_toy(alternate=True, anchor="attitude")
    assert err < 0.05, f"recovered mount is {err:.3f} deg out"
    assert sigma < 0.5


def test_position_measurements_alone_cannot_see_the_mount():
    # The gauge X_k -> X_k (R, 0), E -> (R, 0)^-1 E leaves every predicted
    # measurement unchanged *and* leaves every aircraft position unchanged, so
    # no amount of GNSS can break it.  The solve should come back sitting on its
    # prior, having learned nothing -- which is why the IMU's gravity-referenced
    # attitude is what actually makes the mounting angle observable.
    for alternate in (False, True):
        err, sigma = _solve_toy(alternate=alternate, anchor="position")
        assert err > 0.9 * NOMINAL_OFFSET_DEG
        assert sigma > 4.5


def test_reversing_alternate_passes_sharpens_the_mount():
    # Not an on/off effect once attitude is anchored, but a real one: the
    # reversal gives the inter-pass pairs a different screw axis and roughly
    # halves the uncertainty.  Its bigger job is making the error visible as a
    # washboard, which is measured on the map rather than here.
    err_same, sig_same = _solve_toy(alternate=False, anchor="attitude")
    err_alt, sig_alt = _solve_toy(alternate=True, anchor="attitude")
    assert sig_alt < 0.75 * sig_same, (sig_alt, sig_same)
    assert err_alt <= err_same


def test_factor_matches_the_bare_error_function():
    rng = np.random.default_rng(4)
    Ti, Tj, Ee = rand_pose(rng), rand_pose(rng), rand_pose(rng, 0.2, 0.1)
    measured = rand_pose(rng, 0.5, 0.1)
    noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    f = scan_match_factor(X(0), X(1), E(0), measured, noise)

    values = gtsam.Values()
    values.insert(X(0), Ti)
    values.insert(X(1), Tj)
    values.insert(E(0), Ee)
    r, _ = evaluate(measured, Ti, Tj, Ee)
    assert f.error(values) == pytest.approx(0.5 * float(r @ r) / 0.1 ** 2, rel=1e-9)
