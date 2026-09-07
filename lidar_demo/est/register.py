"""Scan matching between keyframe submaps, and what its covariance is worth.

The transform this module returns is the relative pose between two *sensor*
frames, ``T_{S_i S_j}``, because that is the only thing the point clouds
themselves know about.  Converting it to a statement about body poses needs the
extrinsic, and the whole point of the exercise is that the extrinsic is not
known.  So the conversion is left to the factor, which carries it as a variable;
see :mod:`lidar_demo.est.factors`.

Covariance comes from the registration Hessian, scaled by the residual.  It is
then clamped, because a raw Hessian on this kind of scene is not to be trusted
in every direction at once: a patch of near-planar wheat constrains height and
two tilts well and in-plane translation and heading barely at all, and an
unclamped inverse Hessian would hand the optimiser a wildly overconfident number
on exactly the axes it is worst at.  Degenerate axes are inflated to the cap
rather than dropped, so the pair still contributes what it does know.

Backends: ``small_gicp`` when it is installed, Open3D's generalised ICP
otherwise.  Both are wrapped to return the same thing in the same convention,
and a test pins that convention rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import GraphCfg
from ..frames import log_so3

_BACKEND = None


def available_backend() -> str:
    """Which registration library this environment has."""
    global _BACKEND
    if _BACKEND is None:
        try:
            import small_gicp  # noqa: F401
            _BACKEND = "small_gicp"
        except ImportError:
            try:
                import open3d  # noqa: F401
                _BACKEND = "open3d"
            except ImportError:
                _BACKEND = "none"
    return _BACKEND


@dataclass
class RegResult:
    """One registration attempt."""

    T: np.ndarray            # (4, 4) T_{S_i S_j}: maps points of j into i
    sigmas: np.ndarray       # (6,) [rot x3 (rad), trans x3 (m)] right-perturbation
    converged: bool
    inlier_frac: float
    rmse: float
    degenerate: int          # how many axes had to be inflated
    n_source: int
    n_target: int

    @property
    def ok(self) -> bool:
        return self.converged


def _to_pose(T: np.ndarray):
    return np.asarray(T, dtype=float)


def _shift_from_init(T: np.ndarray, T_init: np.ndarray):
    """How far the solution moved from where it started, in metres and degrees."""
    rel = np.linalg.inv(T_init) @ T
    return (float(np.linalg.norm(rel[:3, 3])),
            float(np.degrees(np.linalg.norm(log_so3(rel[:3, :3])))))


def _cov_from_hessian(H: np.ndarray, rmse: float, n_inliers: int,
                      cfg: GraphCfg, fallback: tuple[float, float]):
    """Per-axis sigmas from a registration Hessian, clamped and de-degenerated.

    ``H`` is in the library's own ordering; both supported backends use
    ``[rotation; translation]``, matching GTSAM's right-perturbation ordering,
    and the convention test pins that.
    """
    lo_r, hi_r = np.radians(cfg.sigma_rot_bounds)
    lo_t, hi_t = cfg.sigma_trans_bounds
    fb = np.r_[np.full(3, np.radians(fallback[0])), np.full(3, fallback[1])]

    if H is None or not np.all(np.isfinite(H)):
        return fb, 6

    dof = max(n_inliers - 6, 1)
    s2 = max(rmse, 1e-4) ** 2 * n_inliers / dof

    Hs = 0.5 * (H + H.T)
    try:
        cov = s2 * np.linalg.inv(Hs + np.eye(6) * max(np.trace(Hs), 1.0) * 1e-9)
    except np.linalg.LinAlgError:
        return fb, 6
    sig = np.sqrt(np.clip(np.diag(cov), 1e-12, None))

    # Degeneracy is judged per block, never across them.  The rotation and
    # translation parts of a registration Hessian are in different units, so
    # their eigenvalues differ by orders of magnitude for reasons that have
    # nothing to do with how well constrained either one is; comparing them
    # flags every pair as degenerate.
    degenerate = int((sig[:3] > hi_r).sum() + (sig[3:] > hi_t).sum())
    sig[:3] = np.clip(sig[:3], lo_r, hi_r)
    sig[3:] = np.clip(sig[3:], lo_t, hi_t)
    return sig, degenerate


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


def _register_small_gicp(target, source, T_init, voxel, max_corr, num_threads):
    import small_gicp

    res = small_gicp.align(target, source, init_T_target_source=T_init,
                           registration_type="GICP",
                           max_correspondence_distance=max_corr,
                           downsampling_resolution=voxel,
                           num_threads=num_threads)
    return (np.asarray(res.T_target_source, dtype=float),
            np.asarray(res.H, dtype=float),
            bool(res.converged), int(res.num_inliers), float(res.error))


def _register_open3d(target, source, T_init, voxel, max_corr):
    import open3d as o3d

    def pc(p):
        c = o3d.geometry.PointCloud()
        c.points = o3d.utility.Vector3dVector(np.ascontiguousarray(p))
        c = c.voxel_down_sample(voxel)
        c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(3.0 * voxel, 0.5), max_nn=20))
        return c

    tgt, src = pc(target), pc(source)
    reg = o3d.pipelines.registration.registration_generalized_icp(
        src, tgt, max_corr, T_init,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src, tgt, max_corr, reg.transformation)
    n = len(reg.correspondence_set)
    return (np.asarray(reg.transformation, dtype=float), np.asarray(info),
            n > 0, n, float(reg.inlier_rmse))


# ---------------------------------------------------------------------------


def register_pair(target_cloud: np.ndarray, source_cloud: np.ndarray,
                  T_init: np.ndarray, cfg: GraphCfg, kind: str = "seq",
                  num_threads: int = 4) -> RegResult:
    """Align ``source`` onto ``target``, both in their own sensor frames.

    Coarse then fine: a wide correspondence radius on a heavily decimated cloud
    to get the pair roughly together, then a tight one at the working
    resolution.  A single tight pass would stall whenever the seed trajectory is
    off by more than the correspondence distance, which over the GNSS-denied
    stretch it certainly is.
    """
    backend = available_backend()
    n_t, n_s = int(target_cloud.shape[0]), int(source_cloud.shape[0])
    fallback = cfg.fallback_seq if kind == "seq" else cfg.fallback_cross
    fail = RegResult(T=np.asarray(T_init, dtype=float),
                     sigmas=np.r_[np.full(3, np.radians(fallback[0])),
                                  np.full(3, fallback[1])],
                     converged=False, inlier_frac=0.0, rmse=np.inf,
                     degenerate=6, n_source=n_s, n_target=n_t)
    if backend == "none":
        raise ImportError("install small_gicp or open3d to register scans")
    if n_t < 200 or n_s < 200:
        return fail

    T = np.asarray(T_init, dtype=float).copy()
    H = None
    inliers, rmse, converged = 0, np.inf, False

    stages = ((cfg.voxel_coarse, cfg.max_corr_coarse), (cfg.voxel, cfg.max_corr))
    for voxel, max_corr in stages:
        try:
            if backend == "small_gicp":
                T, H, converged, inliers, rmse = _register_small_gicp(
                    target_cloud, source_cloud, T, voxel, max_corr, num_threads)
            else:
                T, H, converged, inliers, rmse = _register_open3d(
                    target_cloud, source_cloud, T, voxel, max_corr)
        except Exception:
            return fail
        if not np.all(np.isfinite(T)):
            return fail

    # Measure the overlap directly rather than trusting the backend's inlier
    # count, which is taken after its own internal downsampling and so is not
    # comparable between backends or between pairs of different sizes.
    inlier_frac, rmse = _overlap(target_cloud, source_cloud, T, cfg.max_corr)

    d_m, d_deg = _shift_from_init(T, np.asarray(T_init, dtype=float))
    lim_m, lim_deg = cfg.max_seq_shift if kind == "seq" else cfg.max_cross_shift
    if d_m > lim_m or d_deg > lim_deg:
        return fail
    if inlier_frac < cfg.min_inlier_frac:
        return fail

    sig, degenerate = _cov_from_hessian(H, rmse, max(inliers, 7), cfg, fallback)
    return RegResult(T=T, sigmas=sig, converged=bool(converged),
                     inlier_frac=inlier_frac, rmse=float(rmse),
                     degenerate=degenerate, n_source=n_s, n_target=n_t)


def _overlap(target: np.ndarray, source: np.ndarray, T: np.ndarray,
             max_corr: float, sample: int = 4000):
    """Fraction of source points with a target point nearby, and their RMSE."""
    from scipy.spatial import cKDTree

    src = source @ T[:3, :3].T + T[:3, 3]
    if src.shape[0] > sample:
        idx = np.linspace(0, src.shape[0] - 1, sample).astype(int)
        src = src[idx]
    tree = cKDTree(target)
    d, _ = tree.query(src, k=1, distance_upper_bound=max_corr)
    hit = np.isfinite(d)
    if not hit.any():
        return 0.0, float("inf")
    return float(hit.mean()), float(np.sqrt((d[hit] ** 2).mean()))


def init_from_poses(R_i, p_i, R_j, p_j, E) -> np.ndarray:
    """Seed transform ``T_{S_i S_j}`` implied by two body poses and an extrinsic."""
    from ..frames import compose, invert, to_matrix

    R_si, p_si = compose(R_i, p_i, E.R, E.t)
    R_sj, p_sj = compose(R_j, p_j, E.R, E.t)
    Ri, ti = invert(R_si, p_si)
    return to_matrix(*compose(Ri, ti, R_sj, p_sj))
