"""The collapse: every point walking from where blue put it to where green does.

This is the payoff of the whole piece and it must never be a cut.  The two
clouds hold the same returns in the same order, so each point has one
destination and the move is a straight interpolation.  What a cross-fade would
show instead is one picture disappearing while another appears, which reads as a
slide change and says nothing about the two being the same measurements.

Points are staggered slightly, so the cloud settles rather than snapping as one
rigid body.  The stagger is a fixed function of the point index, not random per
frame, so a given point's motion is monotone.
"""

from __future__ import annotations

import numpy as np

from .scene import hex_rgb, smoothstep


def stagger_phase(n: int, spread: float, seed: int = 0) -> np.ndarray:
    """A per-point start time in ``[0, spread]``, fixed for the whole move."""
    if spread <= 0.0:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return rng.random(n).astype(np.float32) * float(spread)


def morph_positions(blue: np.ndarray, green: np.ndarray, u: float,
                    phase: np.ndarray) -> np.ndarray:
    """Point positions at global progress ``u`` in ``[0, 1]``."""
    spread = float(phase.max()) if phase.size else 0.0
    local = (u - phase) / max(1.0 - spread, 1e-6)
    w = smoothstep(local)[:, None].astype(np.float32)
    return (1.0 - w) * blue + w * green


def morph_colour(colour_a: str, colour_b: str, u: float, phase: np.ndarray,
                 shade: np.ndarray | None = None) -> np.ndarray:
    """Per-point colour part way through the move."""
    a, b = hex_rgb(colour_a), hex_rgb(colour_b)
    spread = float(phase.max()) if phase.size else 0.0
    w = smoothstep((u - phase) / max(1.0 - spread, 1e-6))[:, None]
    rgb = (1.0 - w) * a[None, :] + w * b[None, :]
    if shade is not None:
        s = np.clip(np.asarray(shade, np.float32), 0.0, 1.0)[:, None]
        rgb = rgb * (0.40 + 0.60 * s)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def smoothness(blue: np.ndarray, green: np.ndarray, n_frames: int,
               spread: float, seed: int = 0) -> dict:
    """Gate 6: does the move read smoothly, or does it pop?

    A cloud that jumps most of the way in one frame reads as a cut with extra
    steps.  The measure is the largest single-frame displacement as a fraction
    of the total distance a point has to travel; anything under about a fifth
    looks continuous at 30 frames per second.
    """
    total = np.linalg.norm(green - blue, axis=1)
    moving = total > 1e-6
    if not moving.any():
        return {"max_frame_fraction": 0.0, "median_travel_m": 0.0, "pass": True}
    phase = stagger_phase(blue.shape[0], spread, seed)

    prev = morph_positions(blue, green, 0.0, phase)
    worst = 0.0
    for k in range(1, n_frames + 1):
        cur = morph_positions(blue, green, k / n_frames, phase)
        step = np.linalg.norm(cur - prev, axis=1)[moving] / total[moving]
        worst = max(worst, float(np.percentile(step, 99)))
        prev = cur
    return {"max_frame_fraction": worst,
            "median_travel_m": float(np.median(total[moving])),
            "p95_travel_m": float(np.percentile(total[moving], 95)),
            "n_frames": int(n_frames),
            "pass": bool(worst < 0.20)}
