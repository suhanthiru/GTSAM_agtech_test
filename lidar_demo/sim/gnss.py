"""GNSS fixes in the local ENU frame.

Two details are deliberate.

Vertical sigma is twice horizontal, never isotropic.  A receiver sees satellites
above it and none below, so the vertical dilution of precision is always the
worse number.  Modelling it isotropic is the classic way to end up with a
solution that is good in XY and visibly wrong in Z.

The dropout is keyed to *position*, not to time: it happens where the treeline
blocks the sky, which is the same place the flight plan sends one whole pass.
So the GNSS-denied stretch recurs at a known part of the field and the graph has
to lean on scan matching exactly there, rather than at a random moment.
"""

from __future__ import annotations

import numpy as np

from agspray.trajectory import TrueState

from ..config import GnssCfg


def simulate_gnss(cfg: GnssCfg, truth: TrueState, treeline_y: float,
                  rng: np.random.Generator):
    """Returns ``(t, imu_index, p_meas, valid, sigma)``.

    Every epoch on the nominal 1 Hz timeline is written, with invalid ones
    flagged rather than dropped, so a consumer sees a regular clock and can tell
    a dropout from a gap in the file.
    """
    dt_imu = float(truth.t[1] - truth.t[0])
    step = max(int(round(1.0 / (cfg.rate_hz * dt_imu))), 1)
    idx = np.arange(0, len(truth), step)
    t = truth.t[idx]
    p_true = truth.position[idx]

    sigma = np.array([cfg.sigma_horizontal, cfg.sigma_horizontal, cfg.sigma_vertical])
    p_meas = p_true + rng.normal(scale=sigma, size=(idx.size, 3))

    blocked = np.abs(p_true[:, 1] - treeline_y) < cfg.dropout_radius
    random_loss = rng.random(idx.size) < cfg.random_dropout
    valid = ~(blocked | random_loss)

    return t, idx, p_meas, valid, sigma


def dropout_summary(t: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Longest and total outage, for the run log."""
    if valid.all():
        return {"longest_s": 0.0, "total_s": 0.0, "fraction": 0.0}
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    longest = run = 0
    for v in valid:
        run = 0 if v else run + 1
        longest = max(longest, run)
    return {"longest_s": longest * dt,
            "total_s": float((~valid).sum()) * dt,
            "fraction": float((~valid).mean())}
