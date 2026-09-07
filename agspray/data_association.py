"""The visual front end, shared by every estimator.

Data association is where the row-aliasing failure actually happens, so it has
to be identical across the ladder or the study measures front ends instead of
back ends.  One class, one gate, one scoring rule; the only thing that differs
between methods is the pose and pose covariance they hand in.

That last point is not incidental.  The Mahalanobis gate is sized by the
estimator's *own* covariance, which makes consistency causal rather than merely
observable: an overconfident filter gates too tightly, throws away good matches,
drifts further, and eventually accepts a match one row over because its
descriptor is - by construction in :mod:`agspray.world` - nearly identical.
That is Q2 with teeth, and it needs no extra machinery.

Landmark covariance, by contrast, is a shared proxy that shrinks with the
observation count.  Pulling exact landmark marginals out of iSAM2 every keyframe
would cost more than the rest of the harness put together, and using exact
values for the EKF and approximate ones for the graph would reintroduce the
confound this module exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .measurement_models import (
    Camera,
    backproject_to_plane,
    body_to_camera,
    camera_to_body_jacobian,
    project,
    projection_jacobians,
)


@dataclass
class AssocParams:
    gate_chi2: float = 9.21          # 2 dof, 99 %
    descriptor_thresh: float = 0.55
    new_per_frame: int = 12
    min_separation: float = 0.8      # m; below this a new track is a duplicate
    lm_sigma0: float = 1.5           # m, 1-sigma of a plane-backprojected feature
    lm_sigma_floor: float = 0.05     # m, however many times it has been seen

    @staticmethod
    def from_cfg(cfg) -> "AssocParams":
        e = cfg.estimator
        return AssocParams(
            gate_chi2=e.assoc_gate_chi2,
            descriptor_thresh=e.assoc_descriptor_thresh,
            new_per_frame=e.assoc_new_per_frame,
            min_separation=e.assoc_min_separation,
            lm_sigma0=e.lm_sigma0,
        )


class LandmarkStore:
    """The front end's memory: where each track is, what it looks like, when it
    was last seen.  Positions are refreshed from whichever back end owns them.
    """

    def __init__(self, descriptor_dim: int):
        self.dim = descriptor_dim
        self.key = np.zeros(0, dtype=np.int64)       # estimator-facing key
        self.xyz = np.zeros((0, 3))
        self.desc = np.zeros((0, descriptor_dim))
        self.count = np.zeros(0, dtype=np.int64)
        self.last_seen = np.zeros(0)
        self.first_seen = np.zeros(0)
        self.truth_id = np.zeros(0, dtype=np.int64)  # scoring only
        self._next_key = 0

    def __len__(self) -> int:
        return int(self.key.size)

    def add(self, xyz: np.ndarray, desc: np.ndarray, t: float,
            truth_id: np.ndarray) -> np.ndarray:
        n = xyz.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        keys = np.arange(self._next_key, self._next_key + n, dtype=np.int64)
        self._next_key += n
        self.key = np.concatenate([self.key, keys])
        self.xyz = np.vstack([self.xyz, xyz])
        self.desc = np.vstack([self.desc, desc])
        self.count = np.concatenate([self.count, np.ones(n, dtype=np.int64)])
        self.last_seen = np.concatenate([self.last_seen, np.full(n, t)])
        self.first_seen = np.concatenate([self.first_seen, np.full(n, t)])
        self.truth_id = np.concatenate([self.truth_id, np.asarray(truth_id, dtype=np.int64)])
        return keys

    def touch(self, rows: np.ndarray, desc: np.ndarray, t: float) -> None:
        if rows.size == 0:
            return
        self.count[rows] += 1
        self.last_seen[rows] = t
        # Running mean descriptor, renormalised.
        w = 1.0 / self.count[rows][:, None]
        d = (1.0 - w) * self.desc[rows] + w * desc
        self.desc[rows] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)

    def sigma(self, params: AssocParams) -> np.ndarray:
        """Proxy 1-sigma per landmark, shrinking with the observation count."""
        return np.maximum(params.lm_sigma0 / np.sqrt(np.maximum(self.count, 1)),
                          params.lm_sigma_floor)

    def keep(self, mask: np.ndarray) -> np.ndarray:
        """Retire everything outside ``mask``; returns the retired keys."""
        dropped = self.key[~mask]
        self.key = self.key[mask]
        self.xyz = self.xyz[mask]
        self.desc = self.desc[mask]
        self.count = self.count[mask]
        self.last_seen = self.last_seen[mask]
        self.first_seen = self.first_seen[mask]
        self.truth_id = self.truth_id[mask]
        return dropped

    def retire(self, t: float, lifetime: float, max_live: int) -> np.ndarray:
        """Drop stale tracks, then the least recently seen if still over budget."""
        mask = (t - self.last_seen) <= lifetime
        if int(mask.sum()) > max_live:
            live = np.flatnonzero(mask)
            order = live[np.argsort(-self.last_seen[live], kind="stable")]
            mask = np.zeros_like(mask)
            mask[order[:max_live]] = True
        if mask.all():
            return np.zeros(0, dtype=np.int64)
        return self.keep(mask)


@dataclass
class Association:
    obs_rows: np.ndarray      # index into the frame's surviving observations
    store_rows: np.ndarray    # index into the store
    new_obs_rows: np.ndarray
    new_xyz: np.ndarray


def associate(R_wb: np.ndarray, p_wb: np.ndarray, pose_cov: np.ndarray,
              store: LandmarkStore, uv: np.ndarray, desc: np.ndarray,
              cam: Camera, params: AssocParams) -> Association:
    """Match one frame against the map.

    ``pose_cov`` is the 6x6 body-pose covariance in GTSAM's ``[omega; v]``
    ordering.  Everything else is plain arrays so the same call works from the
    filter and from the graph.
    """
    n_obs = uv.shape[0]
    empty = np.zeros(0, dtype=np.int64)
    if n_obs == 0:
        return Association(empty, empty, empty, np.zeros((0, 3)))

    R_wc, p_wc = body_to_camera(R_wb, p_wb)

    if len(store) == 0:
        new_xyz = backproject_to_plane(R_wc, p_wc, uv, cam)
        take = np.arange(min(n_obs, params.new_per_frame))
        return Association(empty, empty, take, new_xyz[take])

    pred_uv, pc, in_view = project(R_wc, p_wc, store.xyz, cam)
    cand = np.flatnonzero(in_view)
    if cand.size == 0:
        new_xyz = backproject_to_plane(R_wc, p_wc, uv, cam)
        take = np.arange(min(n_obs, params.new_per_frame))
        return Association(empty, empty, take, new_xyz[take])

    H_cam, H_l = projection_jacobians(R_wc, pc[cand], cam)
    H_body = camera_to_body_jacobian(H_cam)

    # S_j = H_pose P_pose H_pose^T + H_l P_l H_l^T + R_pix
    S = np.einsum("nab,bc,ndc->nad", H_body, pose_cov, H_body)
    lm_var = store.sigma(params)[cand] ** 2
    S += np.einsum("nab,ncb->nac", H_l, H_l) * lm_var[:, None, None]
    S += cam.pixel_cov[None, :, :]

    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    inv = np.empty_like(S)
    inv[:, 0, 0] = S[:, 1, 1] / det
    inv[:, 1, 1] = S[:, 0, 0] / det
    inv[:, 0, 1] = -S[:, 0, 1] / det
    inv[:, 1, 0] = -S[:, 1, 0] / det

    r = uv[None, :, :] - pred_uv[cand][:, None, :]                    # (C, O, 2)
    d2 = (r[:, :, 0] * (inv[:, 0, 0][:, None] * r[:, :, 0] + inv[:, 0, 1][:, None] * r[:, :, 1])
          + r[:, :, 1] * (inv[:, 1, 0][:, None] * r[:, :, 0] + inv[:, 1, 1][:, None] * r[:, :, 1]))
    cos = store.desc[cand] @ desc.T                                   # (C, O)

    ok = (d2 < params.gate_chi2) & (cos > params.descriptor_thresh)
    score = np.where(ok, cos, -np.inf)

    # Mutual best match.  Cheap, deterministic, and it degrades the way a real
    # front end does: as the pose covariance grows the gate swallows the
    # neighbouring row and the descriptor can no longer tell them apart.
    best_lm = np.argmax(score, axis=0)
    best_obs = np.argmax(score, axis=1)
    obs_idx = np.arange(n_obs)
    mutual = (best_obs[best_lm] == obs_idx) & np.isfinite(score[best_lm, obs_idx])

    obs_rows = obs_idx[mutual]
    store_rows = cand[best_lm[mutual]]

    free = obs_idx[~mutual]
    new_rows = empty
    new_xyz = np.zeros((0, 3))
    if free.size and params.new_per_frame > 0:
        guess = backproject_to_plane(R_wc, p_wc, uv[free], cam)
        # Suppress duplicates of tracks we already hold.
        d = np.linalg.norm(guess[:, None, :2] - store.xyz[None, :, :2], axis=2)
        fresh = d.min(axis=1) > params.min_separation if len(store) else np.ones(free.size, bool)
        free, guess = free[fresh], guess[fresh]
        take = np.arange(min(free.size, params.new_per_frame))
        new_rows, new_xyz = free[take], guess[take]

    return Association(obs_rows, store_rows, new_rows, new_xyz)
