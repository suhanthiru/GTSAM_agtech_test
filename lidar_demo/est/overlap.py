"""Finding keyframe pairs that saw the same ground.

The obvious rule -- pair up keyframes whose positions are close -- is wrong for
this sensor, and wrongly enough to matter.  The mount is tilted forward, so the
usable footprint sits about fourteen metres ahead of the aircraft.  Two
keyframes on opposite passes that are ten metres apart are flying *towards each
other*, and their footprints are therefore roughly thirty metres apart: they
overlap hardly at all.  The pairs that actually share ground are the ones whose
footprints coincide, which on opposite headings means the aircraft positions are
tens of metres apart.

So proximity is measured between forward-projected footprint centres rather than
between aircraft.  Everything else is the usual: exclude near neighbours in
time, cap the number of partners per keyframe, and prefer partners on a
different pass, because those are the pairs that carry information about the
mounting angle.
"""

from __future__ import annotations

import numpy as np

from ..config import GraphCfg


def footprint_centres(kfs, look_ahead: float, altitude_hint: float = 15.0) -> np.ndarray:
    """Where on the ground each keyframe was looking, in the world ``xy`` plane."""
    out = np.empty((len(kfs), 2))
    for i, kf in enumerate(kfs):
        h = kf.R[:, 0].copy()
        h[2] = 0.0
        n = float(np.linalg.norm(h))
        h = h / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        out[i] = kf.p[:2] + look_ahead * h[:2]
    return out


def look_ahead_distance(altitude: float, scan_min_deg: float,
                        scan_max_deg: float) -> float:
    """Ground range to the middle of the usable aperture."""
    lo = altitude * np.tan(np.radians(scan_min_deg))
    hi = altitude * np.tan(np.radians(scan_max_deg))
    return float(0.5 * (lo + hi))


def sequential_pairs(kfs, stride: int) -> list[tuple[int, int]]:
    """Along-track pairs far enough apart that their submaps share no sweeps.

    Consecutive keyframes are a metre apart and their submaps are built from
    largely the same sweeps, so registering them returns the seed trajectory
    back and tells the graph nothing it did not already assume.  Stepping by the
    submap window makes the two clouds independent measurements of overlapping
    ground.
    """
    n = len(kfs)
    return [(i, i + stride) for i in range(n - stride)
            if kfs[i].pass_id == kfs[i + stride].pass_id]


def overlap_pairs(kfs, cfg: GraphCfg, look_ahead: float,
                  positions: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Spatially overlapping pairs, preferring different passes."""
    from scipy.spatial import cKDTree

    if len(kfs) < 3:
        return []
    xy = footprint_centres(kfs, look_ahead) if positions is None else positions
    t = np.array([kf.t for kf in kfs])
    pid = np.array([kf.pass_id for kf in kfs])

    tree = cKDTree(xy)
    found: set[tuple[int, int]] = set()
    per_kf = np.zeros(len(kfs), dtype=int)

    for i in range(len(kfs)):
        if per_kf[i] >= cfg.overlap_max_per_kf:
            continue
        cand = np.array(tree.query_ball_point(xy[i], cfg.overlap_radius), dtype=int)
        if cand.size == 0:
            continue
        cand = cand[(np.abs(cand - i) > cfg.overlap_exclude_seq)
                    & (np.abs(t[cand] - t[i]) > cfg.overlap_min_dt)]
        if cand.size == 0:
            continue
        # A different pass first: those are the pairs that see the same ground
        # from a reversed heading, which is where the mounting angle shows up.
        d = np.linalg.norm(xy[cand] - xy[i], axis=1)
        same = pid[cand] == pid[i]
        order = np.lexsort((d, same))
        for j in cand[order]:
            if per_kf[i] >= cfg.overlap_max_per_kf:
                break
            if per_kf[j] >= cfg.overlap_max_per_kf:
                continue
            key = (min(i, int(j)), max(i, int(j)))
            if key in found:
                continue
            found.add(key)
            per_kf[i] += 1
            per_kf[j] += 1
    return sorted(found)


def summarise(kfs, pairs) -> dict:
    pid = np.array([kf.pass_id for kf in kfs])
    if not pairs:
        return {"n": 0, "cross_pass": 0, "same_pass": 0}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    cross = int((pid[a] != pid[b]).sum())
    return {"n": len(pairs), "cross_pass": cross, "same_pass": len(pairs) - cross}
