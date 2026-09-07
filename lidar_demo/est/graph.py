"""The factor graph: trajectory, velocities, biases, and one shared mounting angle.

Variables, using GTSAM's shorthand symbols:

``X(i)`` ``Pose3``            body pose at keyframe i
``V(i)`` ``Vector3``          velocity
``B(i)`` ``imuBias.ConstantBias``
``E(0)`` ``Pose3``            the sensor mount -- **one** of these, for the whole flight

Factors:

* preintegrated IMU between consecutive keyframes.  ``PreintegratedImuMeasurements``
  folds a couple of hundred raw samples into a single factor, which is the main
  reason to reach for GTSAM here rather than assemble the same problem by hand.
* a bias random walk between consecutive keyframes.
* GNSS position, wherever a fix was valid.
* scan-match factors, each carrying ``E`` as a third variable, so the mounting
  angle is estimated jointly with the trajectory rather than assumed.
* a loose prior on ``E`` at nominal, and a tight one on its translation.  The
  lever arm is ten centimetres, so the translation is barely observable; saying
  so with a prior is more honest than letting the optimiser wander along a
  direction the data does not constrain.

Robust kernels on the scan-match factors are not optional.  Registration
produces outliers -- a pair that latched onto the wrong ridge, a patch that was
mostly canopy -- and under plain least squares a single bad inter-pass match
drags the whole map with it.

The solve is staged.  Pass one uses only the along-track factors, which fixes
the gross trajectory; pass two adds the inter-pass factors from that better
starting point.  Throwing everything in at once from a drifted seed puts the
inter-pass residuals so far outside the robust kernel's linear region that they
are downweighted into irrelevance before they have a chance to act.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import gtsam
    from gtsam import Pose3, Rot3, Point3
    from gtsam.symbol_shorthand import B, E, V, X
except ImportError:  # pragma: no cover
    gtsam = None

from ..config import GraphCfg, ImuCfg
from ..frames import R_to_rpy_zyx, angle_between_deg
from ..io import Extrinsic


def to_pose3(R, p):
    return Pose3(Rot3(np.asarray(R, dtype=float)), Point3(*np.asarray(p, dtype=float)))


def from_pose3(T):
    return np.asarray(T.rotation().matrix(), dtype=float), np.asarray(T.translation())


@dataclass
class GraphBuild:
    graph: object
    values: object
    n_imu: int = 0
    n_gnss: int = 0
    n_seq: int = 0
    n_cross: int = 0
    kf_used: list = field(default_factory=list)


def preintegration_params(imu: ImuCfg):
    params = gtsam.PreintegrationParams.MakeSharedU(9.80665)
    params.setAccelerometerCovariance(np.eye(3) * imu.accel_noise_density ** 2)
    params.setGyroscopeCovariance(np.eye(3) * imu.gyro_noise_density ** 2)
    params.setIntegrationCovariance(np.eye(3) * 1e-8)
    return params


def _robust(sigmas, cfg: GraphCfg, kernel: str):
    base = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigmas, dtype=float))
    if kernel == "cauchy":
        m = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
    else:
        m = gtsam.noiseModel.mEstimator.Huber.Create(cfg.huber_k)
    return gtsam.noiseModel.Robust.Create(m, base)


def build_graph(kfs, imu_t, accel, gyro, gnss, seq_regs, cross_regs,
                E_init: Extrinsic, cfg: GraphCfg, imu: ImuCfg,
                seed_poses=None, seed_vels=None, seed_biases=None,
                E_prior: Extrinsic | None = None) -> GraphBuild:
    """Assemble the graph.  ``*_regs`` are ``{(i, j): RegResult}``.

    ``E_init`` is where the optimiser starts from and ``E_prior`` is what the
    prior is centred on.  They are different arguments on purpose.  Across
    rounds the starting value should follow the current best estimate, but the
    prior must not: it encodes what the drawing says the mount is, which does
    not change because the last solve moved.  Re-centring it each round removes
    the only absolute anchor on the mounting angle and lets it ratchet away a
    fraction of a degree at a time.
    """
    from .factors import scan_match_factor

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    out = GraphBuild(graph=graph, values=values)

    n = len(kfs)
    poses = seed_poses if seed_poses is not None else [to_pose3(kf.R, kf.p) for kf in kfs]
    vels = (seed_vels if seed_vels is not None
            else [np.zeros(3) if kf.v is None else np.asarray(kf.v, float) for kf in kfs])
    biases = (seed_biases if seed_biases is not None
              else [gtsam.imuBias.ConstantBias() for _ in kfs])

    for i in range(n):
        values.insert(X(i), poses[i])
        values.insert(V(i), vels[i])
        values.insert(B(i), biases[i])

    # ---- priors ----
    graph.add(gtsam.PriorFactorPose3(X(0), poses[0], gtsam.noiseModel.Diagonal.Sigmas(
        np.r_[np.radians([2.0, 2.0, 5.0]), [1.0, 1.0, 2.0]])))
    graph.add(gtsam.PriorFactorVector(V(0), vels[0],
                                      gtsam.noiseModel.Isotropic.Sigma(3, 1.0)))
    graph.add(gtsam.PriorFactorConstantBias(
        B(0), biases[0], gtsam.noiseModel.Diagonal.Sigmas(
            np.r_[np.full(3, 0.1), np.full(3, 0.01)])))

    # ---- IMU ----
    params = preintegration_params(imu)
    dt_nominal = float(np.median(np.diff(imu_t)))
    for i in range(n - 1):
        i0, i1 = kfs[i].imu_idx, kfs[i + 1].imu_idx
        if i1 <= i0:
            continue
        pim = gtsam.PreintegratedImuMeasurements(params, biases[i])
        for k in range(i0, i1):
            dt = float(imu_t[k + 1] - imu_t[k]) if k + 1 < imu_t.size else dt_nominal
            pim.integrateMeasurement(accel[k], gyro[k], dt)
        graph.add(gtsam.ImuFactor(X(i), V(i), X(i + 1), V(i + 1), B(i), pim))
        out.n_imu += 1

        span = max(float(imu_t[i1] - imu_t[i0]), 1e-3)
        rw = np.r_[np.full(3, imu.accel_bias_rw * np.sqrt(span)),
                   np.full(3, imu.gyro_bias_rw * np.sqrt(span))]
        graph.add(gtsam.BetweenFactorConstantBias(
            B(i), B(i + 1), gtsam.imuBias.ConstantBias(),
            gtsam.noiseModel.Diagonal.Sigmas(np.maximum(rw, 1e-6))))

    # ---- GNSS ----
    t_gnss, p_gnss, valid, sigma_gnss, _ = gnss
    kf_t = np.array([kf.t for kf in kfs])
    gnss_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigma_gnss, dtype=float))
    for k in range(t_gnss.size):
        if not valid[k]:
            continue
        i = int(np.argmin(np.abs(kf_t - t_gnss[k])))
        dt = float(t_gnss[k] - kf_t[i])
        if abs(dt) > 1.0:
            continue
        # Carry the fix back to the keyframe's own instant; at 1 Hz the shift is
        # at most half a second of flight and the velocity estimate covers it.
        z = np.asarray(p_gnss[k], dtype=float) - np.asarray(vels[i], dtype=float) * dt
        graph.add(gtsam.GPSFactor(X(i), Point3(*z), gnss_noise))
        out.n_gnss += 1

    # ---- scan matching ----
    E_ref = E_init if E_prior is None else E_prior
    values.insert(E(0), to_pose3(E_init.R, E_init.t))
    graph.add(gtsam.PriorFactorPose3(
        E(0), to_pose3(E_ref.R, E_ref.t), gtsam.noiseModel.Diagonal.Sigmas(
            np.r_[np.full(3, np.radians(cfg.prior_E_rot_deg)),
                  np.full(3, cfg.prior_E_trans)])))

    for regs, kernel, attr in ((seq_regs, "huber", "n_seq"),
                               (cross_regs, cfg.cross_kernel, "n_cross")):
        for (i, j), res in regs.items():
            if not res.ok:
                continue
            R, p = res.T[:3, :3], res.T[:3, 3]
            graph.add(scan_match_factor(X(i), X(j), E(0), to_pose3(R, p),
                                        _robust(res.sigmas, cfg, kernel)))
            setattr(out, attr, getattr(out, attr) + 1)

    return out


def lm_params(cfg: GraphCfg, max_iters: int | None = None, verbose: bool = False):
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(int(cfg.lm_max_iters if max_iters is None else max_iters))
    p.setlambdaInitial(1e-3)
    p.setlambdaFactor(10.0)
    p.setlambdaUpperBound(1e8)
    p.setRelativeErrorTol(1e-6)
    p.setAbsoluteErrorTol(1e-6)
    p.setLinearSolverType("MULTIFRONTAL_CHOLESKY")
    if verbose:
        p.setVerbosityLM("SUMMARY")
    return p


def solve(build: GraphBuild, cfg: GraphCfg, max_iters: int | None = None,
          verbose: bool = False):
    """Run Levenberg-Marquardt and report what happened."""
    e0 = build.graph.error(build.values)
    opt = gtsam.LevenbergMarquardtOptimizer(build.graph, build.values,
                                            lm_params(cfg, max_iters, verbose))
    result = opt.optimize()
    return result, {"error_before": float(e0),
                    "error_after": float(build.graph.error(result)),
                    "iterations": int(opt.iterations())}


def extract(result, n: int):
    """Pull the optimised states back out as plain arrays."""
    Rs, ps, vs, bs = [], [], [], []
    for i in range(n):
        R, p = from_pose3(result.atPose3(X(i)))
        Rs.append(R)
        ps.append(p)
        vs.append(np.asarray(result.atVector(V(i)), dtype=float))
        b = result.atConstantBias(B(i))
        bs.append((np.asarray(b.accelerometer()), np.asarray(b.gyroscope())))
    Ee = result.atPose3(E(0))
    R, p = from_pose3(Ee)
    return np.stack(Rs), np.stack(ps), np.stack(vs), bs, Extrinsic(R, p)


def extrinsic_marginal(build: GraphBuild, result):
    """1-sigma on the mounting angle, in degrees, or NaN if it cannot be had."""
    try:
        cov = gtsam.Marginals(build.graph, result).marginalCovariance(E(0))
        return np.degrees(np.sqrt(np.diag(cov)[:3]))
    except Exception:
        return np.full(3, np.nan)


def extrinsic_report(E_est: Extrinsic, E_nom: Extrinsic, sigma_deg,
                     E_true: Extrinsic | None = None,
                     gate_deg: float = 0.2) -> dict:
    """The title card: nominal, recovered and (for the gate only) true."""
    rep = {
        "nominal_rpy_deg": [float(v) for v in E_nom.rpy_deg],
        "recovered_rpy_deg": [float(v) for v in E_est.rpy_deg],
        "recovered_sigma_deg": [float(v) for v in np.asarray(sigma_deg, dtype=float)],
        "recovered_t_m": [float(v) for v in E_est.t],
        "moved_from_nominal_deg": angle_between_deg(E_nom.R, E_est.R),
        "gate_deg": float(gate_deg),
    }
    if E_true is not None:
        rep["true_rpy_deg"] = [float(v) for v in E_true.rpy_deg]
        rep["error_deg"] = angle_between_deg(E_est.R, E_true.R)
        rep["nominal_error_deg"] = angle_between_deg(E_nom.R, E_true.R)
        rep["pass"] = bool(rep["error_deg"] < gate_deg)
    return rep


def format_report(rep: dict) -> str:
    def row(name, rpy, extra=""):
        return f"  {name:<14}{rpy[0]:8.3f}{rpy[1]:9.3f}{rpy[2]:8.3f}   {extra}"

    lines = ["Boresight  R_BS as roll / pitch / yaw, degrees",
             f"  {'':<14}{'roll':>8}{'pitch':>9}{'yaw':>8}",
             row("nominal", rep["nominal_rpy_deg"]),
             row("recovered", rep["recovered_rpy_deg"],
                 "+/- " + "  ".join(f"{s:.3f}" for s in rep["recovered_sigma_deg"]))]
    if "true_rpy_deg" in rep:
        lines.append(row("true", rep["true_rpy_deg"], "(gate only)"))
        lines.append("")
        lines.append(f"  nominal was {rep['nominal_error_deg']:.3f} deg from the truth; "
                     f"the solve moved {rep['moved_from_nominal_deg']:.3f} deg")
        lines.append(f"  |Log(R_est^T R_true)| = {rep['error_deg']:.3f} deg"
                     f"   gate 5 (< {rep['gate_deg']:.1f}): "
                     f"{'PASS' if rep['pass'] else 'FAIL'}")
    else:
        lines.append("")
        lines.append(f"  the solve moved {rep['moved_from_nominal_deg']:.3f} deg "
                     "from nominal")
    return "\n".join(lines)
