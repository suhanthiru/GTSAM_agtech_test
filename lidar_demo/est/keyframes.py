"""Keyframes and the local submaps that get registered.

One graph node per metre of travel or ten degrees of turn, not one per sweep.
At 5 m/s and 10 Hz that is roughly one node every two sweeps, which keeps the
graph tractable and better conditioned: consecutive sweeps 10 cm apart would add
thousands of variables that are nearly linearly dependent on their neighbours
and would tell the optimiser almost nothing it did not already know.

Each keyframe owns the sweeps collected while it was current, deskewed and
gathered into its own sensor frame.  That aggregate is what registration sees.
Aggregating matters: a single sweep from a narrow forward-looking aperture is a
thin crescent of ground, and two crescents from slightly different places have
little to align.  A metre of accumulated travel gives a patch with real extent
and, when the aircraft passes one, a prop standing up out of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..frames import log_so3
from ..io import DenseTraj, Extrinsic, RunReader
from .deskew import deskew_to_sensor


@dataclass
class Keyframe:
    index: int
    t: float                       # sweep-start time of the first owned sweep
    imu_idx: int                   # index into the IMU clock
    sweeps: list[int] = field(default_factory=list)
    R: np.ndarray | None = None    # body-to-world at ``t``, from the seed trajectory
    p: np.ndarray | None = None
    v: np.ndarray | None = None
    pass_id: int = 0
    cloud: np.ndarray | None = None   # (N, 3) in this keyframe's sensor frame

    @property
    def n_points(self) -> int:
        return 0 if self.cloud is None else int(self.cloud.shape[0])


def select_keyframes(traj: DenseTraj, sweep_t: np.ndarray,
                     d_trans: float = 1.0, d_rot_deg: float = 10.0) -> list[Keyframe]:
    """Pick keyframes along a trajectory and assign every sweep to one."""
    sweep_t = np.asarray(sweep_t, dtype=float)
    R_s, p_s = traj.at(sweep_t)

    kfs: list[Keyframe] = []
    anchor_R, anchor_p = None, None
    rot_thresh = np.radians(d_rot_deg)

    for k in range(sweep_t.size):
        new = anchor_p is None
        if not new:
            moved = float(np.linalg.norm(p_s[k] - anchor_p))
            turned = float(np.linalg.norm(log_so3(anchor_R.T @ R_s[k])))
            new = moved >= d_trans or turned >= rot_thresh
        if new:
            i = int(np.searchsorted(traj.t, sweep_t[k]))
            kfs.append(Keyframe(index=len(kfs), t=float(sweep_t[k]),
                                imu_idx=min(i, len(traj.t) - 1),
                                R=R_s[k].copy(), p=p_s[k].copy(),
                                v=None if traj.v is None else traj.v[min(i, len(traj.t) - 1)].copy()))
            anchor_R, anchor_p = R_s[k], p_s[k]
        kfs[-1].sweeps.append(int(k))
    return kfs


def assign_pass_ids(kfs: list[Keyframe], turn_rate_deg: float = 15.0) -> None:
    """Number the passes, so overlap detection can prefer different ones.

    A turn is a stretch where the heading swings quickly; the pass counter ticks
    over once the aircraft settles onto a new heading.  This is derived from the
    estimated trajectory rather than read from the truth, because the
    reconstruction is not allowed to look at the truth.
    """
    if not kfs:
        return
    headings = np.array([kf.R[:, 0][:2] for kf in kfs])
    n = np.linalg.norm(headings, axis=1, keepdims=True)
    headings = headings / np.maximum(n, 1e-9)

    pid = 0
    kfs[0].pass_id = 0
    for k in range(1, len(kfs)):
        dt = max(kfs[k].t - kfs[k - 1].t, 1e-6)
        c = float(np.clip(headings[k] @ headings[k - 1], -1.0, 1.0))
        rate = np.degrees(np.arccos(c)) / dt
        if rate > turn_rate_deg:
            kfs[k].pass_id = -1            # mid-turn, resolved below
        else:
            kfs[k].pass_id = pid
        if kfs[k].pass_id == -1 and kfs[k - 1].pass_id >= 0:
            pid += 1
    # A keyframe caught mid-turn belongs to the pass it is about to join.
    nxt = 0
    for k in range(len(kfs) - 1, -1, -1):
        if kfs[k].pass_id < 0:
            kfs[k].pass_id = nxt
        else:
            nxt = kfs[k].pass_id


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Average the points in each voxel.  Pure numpy, no dependency."""
    if pts.shape[0] == 0:
        return pts
    key = np.floor(pts / voxel).astype(np.int64)
    key -= key.min(axis=0)
    dims = key.max(axis=0) + 1
    flat = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    order = np.argsort(flat, kind="stable")
    flat = flat[order]
    pts = pts[order]
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    sums = np.add.reduceat(pts, starts, axis=0)
    counts = np.diff(np.r_[starts, pts.shape[0]])[:, None]
    return sums / counts


def build_submaps(kfs: list[Keyframe], reader: RunReader, traj: DenseTraj,
                  E: Extrinsic, voxel: float = 0.2, radius: float = 4.0,
                  max_points: int = 80000, subset: list[int] | None = None) -> None:
    """Fill each keyframe's cloud, expressed in its own sensor frame.

    Every sweep within ``radius`` metres of travel is deskewed straight into the
    sensor frame at the keyframe's own time, which folds the intra-sweep motion
    and the sweep-to-keyframe offset into a single transform.  Everything stays
    relative, so a seed trajectory that is metres out in absolute terms still
    produces a clean local patch.

    The window is what makes the patch registrable.  One keyframe's own metre of
    travel yields a thin crescent from a narrow forward aperture, and two such
    crescents have almost nothing to align; several metres of accumulated travel
    spans enough ground, and enough of the props standing up out of it, to pin
    all six degrees of freedom.
    """
    dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(
        np.stack([kf.p for kf in kfs]), axis=0), axis=1))]
    want = range(len(kfs)) if subset is None else subset

    cache: dict[int, np.ndarray] = {}
    for i in want:
        kf = kfs[i]
        lo = int(np.searchsorted(dist, dist[i] - radius, side="left"))
        hi = int(np.searchsorted(dist, dist[i] + radius, side="right"))
        chunks = []
        for j in range(lo, hi):
            for k in kfs[j].sweeps:
                pts = cache.get(k)
                if pts is None:
                    sw = reader.sweep(k)
                    if len(sw) == 0:
                        continue
                    pts = (sw.xyz.astype(np.float64), sw.t_abs)
                    cache[k] = pts
                chunks.append(_deskew_cached(pts, traj, E, kf.t))
        if not chunks:
            kf.cloud = np.zeros((0, 3))
            continue
        pts = voxel_downsample(np.concatenate(chunks, axis=0), voxel)
        if pts.shape[0] > max_points:
            sel = np.random.default_rng(kf.index).choice(
                pts.shape[0], max_points, replace=False)
            pts = pts[sel]
        kf.cloud = pts.astype(np.float64)
        if len(cache) > 4000:
            cache.clear()


def _deskew_cached(cached, traj: DenseTraj, E: Extrinsic, t_ref: float) -> np.ndarray:
    """:func:`deskew_to_sensor` on points already read off disk."""
    xyz, t_pts = cached
    R_t, p_t = traj.at(t_pts)
    R_ref, p_ref = traj.at(np.array([t_ref]))
    R_ref, p_ref = R_ref[0], p_ref[0]
    R_rel = np.einsum("ji,tjk->tik", R_ref, R_t)
    p_rel = np.einsum("ji,tj->ti", R_ref, p_t - p_ref)
    p_B = xyz @ E.R.T + E.t
    p_B_ref = np.einsum("tij,tj->ti", R_rel, p_B) + p_rel
    return (p_B_ref - E.t) @ E.R


def keyframe_arrays(kfs: list[Keyframe]):
    """Poses and metadata as arrays, for saving and for spatial queries."""
    return {
        "index": np.array([kf.index for kf in kfs], dtype=np.int64),
        "t": np.array([kf.t for kf in kfs], dtype=np.float64),
        "imu_idx": np.array([kf.imu_idx for kf in kfs], dtype=np.int64),
        "pass_id": np.array([kf.pass_id for kf in kfs], dtype=np.int64),
        "p": np.stack([kf.p for kf in kfs]),
        "R": np.stack([kf.R for kf in kfs]),
        "n_points": np.array([kf.n_points for kf in kfs], dtype=np.int64),
    }
