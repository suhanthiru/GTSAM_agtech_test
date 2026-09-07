"""The one interface every estimator on the ladder implements.

The point of this module is negative: it is what the estimators are *not*
allowed to know.  They never see ground truth, never see a landmark's true
identity, never learn which tier they are running in, and never learn which
rung of the ladder they are.  The runner hands each of them the same
measurements in the same order and asks the same questions.

The only thing that varies down the ladder is how far back in time a method may
still revise, which is expressed entirely through :meth:`Estimator.history`.
A filter returns a history that never changes once written; a smoother returns
one whose recent entries move; the batch method returns one whose entire
contents can move at :meth:`Estimator.finalize`.  Everything downstream - the
believed coverage map, the correction-latency metric, the final map error - is
computed from that single signal, so no estimator needs special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..data_association import Association
from ..measurement_models import Camera


@dataclass
class PoseEstimate:
    """A pose and its uncertainty, in the conventions of :mod:`measurement_models`.

    ``cov`` is the 6x6 covariance of the *right* local perturbation in GTSAM's
    ordering ``[omega; v]``, i.e. the covariance of ``Log(T_hat^-1 T_true)``.
    The front end sizes its gate with this matrix, so an overconfident
    estimator is punished through data association rather than only on paper.
    """

    R: np.ndarray                # (3, 3) body -> world
    p: np.ndarray                # (3,)
    cov: np.ndarray              # (6, 6) in [omega; v]
    v: np.ndarray | None = None  # (3,) world velocity, when the method has one

    def copy(self) -> "PoseEstimate":
        return PoseEstimate(self.R.copy(), self.p.copy(), self.cov.copy(),
                            None if self.v is None else self.v.copy())


@dataclass
class PoseHistory:
    """What an estimator currently believes about the keyframes it has seen.

    ``kf`` is the keyframe index, so the runner can tell which entries changed
    since the last call without assuming anything about the window.
    """

    kf: np.ndarray               # (K,) int keyframe indices, ascending
    R: np.ndarray                # (K, 3, 3)
    p: np.ndarray                # (K, 3)

    @staticmethod
    def empty() -> "PoseHistory":
        return PoseHistory(np.zeros(0, dtype=np.int64), np.zeros((0, 3, 3)),
                           np.zeros((0, 3)))


class Estimator:
    """Base class.  Subclasses override what they support and no more.

    Lifecycle, driven by :mod:`agspray.runner`::

        est = Method(cfg, camera)
        est.initialize(t0, p0, v0, R0)
        for each IMU sample:      est.propagate(t, accel, gyro)
        for each GNSS fix:        est.add_gnss(t, pos, sigma)
        at each keyframe:         est.add_frame(t, kf, uv, desc, assoc)
                                  est.end_keyframe(t, kf)
                                  est.history()
        at the end:               est.finalize()

    ``propagate`` is called for every IMU sample including those between
    keyframes, because the spray controller asks for :meth:`current` at a
    higher rate than keyframes arrive.  A boom decision made at 20 Hz on a
    5 Hz pose would smear the very error the study measures.
    """

    #: Set by subclasses.  Purely informational; nothing branches on it.
    name: str = "base"

    def __init__(self, cfg: Config, camera: Camera, *,
                 robust: bool | None = None,
                 estimate_bias: bool | None = None):
        self.cfg = cfg
        self.camera = camera
        est = cfg.estimator
        self.robust = est.robust if robust is None else bool(robust)
        self.estimate_bias = est.estimate_bias if estimate_bias is None else bool(estimate_bias)
        self.t = 0.0

    # ---------------- lifecycle ----------------

    def initialize(self, t0: float, p0: np.ndarray, v0: np.ndarray,
                   R0: np.ndarray) -> None:
        """Seed the state.

        Every method is given the same launch-point prior - position, velocity
        and attitude with the same uncertainty - so the ladder starts level.
        Bias is *not* given: it is drawn per seed in :mod:`agspray.sensors` and
        each method has to work it out or live without it.
        """
        raise NotImplementedError

    def propagate(self, t: float, accel: np.ndarray, gyro: np.ndarray) -> None:
        raise NotImplementedError

    def add_gnss(self, t: float, pos: np.ndarray, sigma: np.ndarray) -> None:
        """Absorb one position fix.  ``sigma`` is the receiver's *reported*
        1-sigma, which in the multipath zone is optimistic by construction."""
        raise NotImplementedError

    def add_frame(self, t: float, kf: int, uv: np.ndarray, desc: np.ndarray,
                  assoc: Association, keys: np.ndarray,
                  new_keys: np.ndarray) -> None:
        """Absorb one camera frame.

        ``assoc`` came from the shared front end, so every method is handed the
        same matching *algorithm*; the matches themselves differ only because
        the pose and covariance fed to that algorithm differ.  ``keys`` are the
        store keys of ``assoc.store_rows`` and ``new_keys`` those minted for
        ``assoc.new_obs_rows``.
        """
        raise NotImplementedError

    def end_keyframe(self, t: float, kf: int) -> PoseEstimate:
        """Finish the keyframe and return the current pose belief."""
        raise NotImplementedError

    def current(self) -> PoseEstimate:
        """Best pose right now, between keyframes.  Cheap: called at tick rate."""
        raise NotImplementedError

    def history(self) -> PoseHistory:
        """Current belief about past keyframe poses.

        This is the ladder's only degree of freedom.  Returning a history whose
        entries never change *is* what makes something a filter.
        """
        return PoseHistory.empty()

    def landmark_xyz(self, keys: np.ndarray) -> np.ndarray:
        """Positions the method currently believes for the given store keys.

        Returns ``(len(keys), 3)``.  Rows for keys the method does not hold are
        filled with NaN and left alone by the caller.
        """
        return np.full((np.asarray(keys).size, 3), np.nan)

    def drop_landmarks(self, keys: np.ndarray) -> None:
        """The front end retired these tracks; free whatever they own."""
        return None

    def finalize(self) -> None:
        """Last chance to revise.  Only the batch method does anything here."""
        return None

    # ---------------- helpers shared by subclasses ----------------

    @property
    def prior_sigmas(self) -> dict[str, np.ndarray]:
        """Launch-point prior, identical for every method.

        Loose enough to be honest about a real take-off fix, tight enough that
        no method spends the first row recovering from the initialisation.
        """
        return {
            "position": np.array([0.5, 0.5, 0.8]),
            "velocity": np.array([0.2, 0.2, 0.2]),
            "rotation": np.array([0.02, 0.02, 0.05]),
            "accel_bias": np.full(3, self.cfg.imu.accel_bias_init),
            "gyro_bias": np.full(3, self.cfg.imu.gyro_bias_init),
        }


@dataclass
class HistoryRecorder:
    """Tracks how a method's belief about each keyframe evolves.

    Correction latency is not a standard metric and cannot be recovered after
    the fact from a trajectory, so it is measured here as it happens: every
    time :meth:`Estimator.history` is polled, any keyframe whose believed
    position has moved is stamped with the current