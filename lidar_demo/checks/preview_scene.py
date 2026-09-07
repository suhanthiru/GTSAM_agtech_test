"""Look at the scene and the flight before spending an hour ray casting it.

Four panels: the terrain from above with the flight lines on it, the canopy, a
cross-section through the survey, and the attitude the aircraft actually flies.
Nothing here feeds the pipeline; it exists so a mistake in the scene is caught
by eye in a second rather than by a gate check twenty minutes later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import DemoConfig, load
from ..sim.flight import plan_survey
from ..sim.scene import build_scene
from ..sim.terrain import sample_bilinear


def preview(cfg: DemoConfig, out: Path) -> Path:
    rng = np.random.default_rng(cfg.run.seed)
    scene = build_scene(cfg.scene, rng)
    path = plan_survey(cfg.flight, cfg.scene)
    hf = scene.hf

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor("white")

    # --- terrain from above, with the flight lines ---
    ax = axes[0, 0]
    im = ax.imshow(hf.z_terrain, origin="lower", cmap="terrain",
                   extent=[hf.x[0], hf.x[-1], hf.y[0], hf.y[-1]])
    plt.colorbar(im, ax=ax, label="bare earth, m")
    t = np.linspace(0.0, path.duration, 6000)
    p, _, _, is_turn, _, pid = path.sample(t)
    ax.plot(p[:, 0], p[:, 1], "-", color="#111111", lw=0.8, alpha=0.9)
    for k in np.unique(pid):
        m = (pid == k) & ~is_turn
        if not m.any():
            continue
        seg = p[m]
        mid = seg[len(seg) // 2]
        d = seg[-1] - seg[0]
        d = d / max(np.linalg.norm(d), 1e-9)
        ax.arrow(mid[0], mid[1], 8 * d[0], 8 * d[1], head_width=2.2,
                 color="#B00020", length_includes_head=True)
    xy = np.array([pr.xy for pr in scene.props])
    ax.plot(xy[:, 0], xy[:, 1], "^", ms=4, color="#222222")
    ax.add_patch(plt.Rectangle((0, 0), cfg.scene.survey_x, cfg.scene.survey_y,
                               fill=False, ec="w", lw=1.5, ls="--"))
    ax.set_title("terrain, flight lines and props  (arrows show heading)")
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")

    # --- canopy ---
    ax = axes[0, 1]
    im = ax.imshow(hf.canopy_h, origin="lower", cmap="YlGn",
                   extent=[hf.x[0], hf.x[-1], hf.y[0], hf.y[-1]])
    plt.colorbar(im, ax=ax, label="canopy height, m")
    ax.set_title("canopy: healthy, stressed and bare blocks")
    ax.set_xlabel("x, m")

    # --- cross-section ---
    ax = axes[1, 0]
    x0 = cfg.render.section_x
    ys = np.linspace(-10.0, cfg.scene.survey_y + 10.0, 1200)
    q = np.stack([np.full_like(ys, x0), ys], axis=1)
    ax.plot(ys, sample_bilinear(hf, q, "terrain"), "-", color="#333333",
            lw=1.5, label="bare earth")
    ax.plot(ys, sample_bilinear(hf, q, "surface"), "-", color="#1D9E75",
            lw=1.0, label="canopy top")
    ax.set_title(f"cross-section at x = {x0:.0f} m  (the gate-3 viewing plane)")
    ax.set_xlabel("y, m")
    ax.set_ylabel("z, m")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # --- attitude ---
    ax = axes[1, 1]
    ts = np.linspace(0.0, path.duration, 4000)
    R, _ = path.pose(ts)
    tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    ax.plot(ts, tilt, lw=0.8, color="#2E6FD6", label="tilt from vertical")
    ax.plot(ts, np.abs(yaw), lw=0.8, color="#B08000", alpha=0.7, label="|yaw|")
    ax.set_title("attitude over the flight")
    ax.set_xlabel("t, s")
    ax.set_ylabel("deg")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    d = cfg.derived
    fig.suptitle(
        f"{cfg.run.name}: {path.duration:.0f} s, {cfg.flight.n_passes} passes + 1 crossing, "
        f"{scene.n_triangles/1e6:.2f} M triangles, sidelap {d.sidelap_frac:.0%}",
        fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/preview_scene.png")
    args = ap.parse_args(argv)
    cfg = load(args.config)
    path = preview(cfg, Path(args.out))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
