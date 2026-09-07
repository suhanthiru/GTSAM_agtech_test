"""Assembling the seven shots into one film.

    1  beauty        the sunlit field, no overlay, let it breathe
    2  reveal        the crop dissolves and the blue cloud fades in
    3  drift         push in on a patch with verticals: rows doubled, post smeared
    4  corrugation   cut to the cross-section: the washboard against the truth
    5  collapse      blue morphs to green, point by point, onto the true terrain
    6  title card    what the mount was, and what the graph found
    7  pull back     the corrected map in three dimensions

Two rules from the brief are followed literally and they matter.  The two
failures get separate beats, because shown together they read as one vague mess
rather than as drift and a systematic mounting error.  And the collapse is a
morph, never a cut: the two clouds hold the same returns, and only a move that
carries each point from one position to the other says so.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..io import RunReader, load_traj
from . import beauty, crosssection as xs
from .camera import CameraKey, apply, interp, three_quarter
from .morph import morph_colour, morph_positions, smoothness, stagger_phase
from .scene import (BACKGROUND, BLUE, GREEN, add_cloud, caption, dissolve,
                    hex_rgb, make_plotter, render, smoothstep)
from .titlecard import load_report, render_card


class Frames:
    """Collects frames and writes them out as video."""

    def __init__(self, path: Path, fps: int, keep_dir: Path | None = None):
        self.path = path
        self.fps = fps
        self.keep_dir = keep_dir
        self.n = 0
        self._writer = None

    def __enter__(self):
        import imageio.v2 as imageio

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Constant-quality rather than a target bitrate.  A point cloud on black
        # is nearly all high-frequency detail, so a bitrate-driven encode either
        # smears the points into mush or produces a file several times larger
        # than the sequence is worth.
        self._writer = imageio.get_writer(
            self.path, fps=self.fps, codec="libx264",
            pixelformat="yuv420p", macro_block_size=1,
            ffmpeg_params=["-crf", "21", "-preset", "slow",
                           "-tune", "grain", "-movflags", "+faststart"])
        if self.keep_dir:
            self.keep_dir.mkdir(parents=True, exist_ok=True)
        return self

    def add(self, img: np.ndarray) -> None:
        img = np.ascontiguousarray(img[:, :, :3].astype(np.uint8))
        self._writer.append_data(img)
        if self.keep_dir:
            import imageio.v2 as imageio
            imageio.imwrite(self.keep_dir / f"frame_{self.n:05d}.png", img)
        self.n += 1

    def hold(self, img: np.ndarray, seconds: float) -> None:
        for _ in range(max(int(round(seconds * self.fps)), 1)):
            self.add(img)

    def __exit__(self, *exc):
        self._writer.close()


def _heightfield(reader: RunReader):
    from ..sim.terrain import Heightfield

    d = reader.heightfield()
    return Heightfield(x0=float(d["x0"]), y0=float(d["y0"]), dx=float(d["dx"]),
                       z_terrain=d["z_terrain"], z_surface=d["z_surface"],
                       canopy_h=d["canopy_h"], class_id=d["class_id"],
                       ditch_p0=tuple(d["ditch_xy"][0]), ditch_p1=tuple(d["ditch_xy"][1]))


def _load_clouds(run_dir: Path, max_points: int, seed: int = 0):
    """Blue and green at two densities, decimated identically.

    The wide shots are decimated to keep frame times sane.  The cross-section
    is not: it keeps only a one-metre slab out of a hundred-and-twenty-metre
    field, so a cloud thinned for the wide shots leaves a few thousand points in
    the frame and the ground stops reading as a surface.  The slab is taken from
    everything and thinned afterwards if it needs to be.
    """
    blue = np.load(run_dir / "map" / "blue_xyz.npy")
    green = np.load(run_dir / "map" / "green_xyz.npy")
    inten = np.load(run_dir / "map" / "blue_intensity.npy").astype(np.float32)
    if blue.shape != green.shape:
        raise AssertionError("blue and green clouds must correspond point for point")
    shade_all = inten / max(float(inten.max()), 1e-6)

    if blue.shape[0] > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(
            blue.shape[0], max_points, replace=False))
        wide = (blue[idx], green[idx], shade_all[idx])
    else:
        wide = (blue, green, shade_all)
    return wide, (blue, green, shade_all)


# ---------------------------------------------------------------------------
# shots
# ---------------------------------------------------------------------------


def shot_beauty(cfg, hf, scene_mesh, truth, size, n_frames, wheat_points):
    """The field, sunlit, with the aircraft flying the pattern."""
    pl = make_plotter(*size)
    beauty.add_sky(pl)
    beauty.add_ground(pl, hf)
    beauty.add_wheat(pl, hf, cfg, n=wheat_points)
    beauty.add_props(pl, scene_mesh)
    beauty.add_sun(pl, cfg)

    # Low and warm, looking across the crop towards the treeline and the
    # windmill, so the horizon is broken by the verticals rather than by nothing.
    centre = (cfg.scene.survey_x * 0.62, cfg.scene.survey_y * 0.42, 3.0)
    keys = [three_quarter(centre, 86.0, 11.0, 168.0),
            three_quarter(centre, 74.0, 15.0, 146.0)]

    out = []
    t_lo, t_hi = 26.0, 26.0 + 0.42 * float(truth.t[-1])
    for k in range(n_frames):
        u = k / max(n_frames - 1, 1)
        apply(pl, interp(keys, u))
        tq = t_lo + u * (t_hi - t_lo)
        R, p = truth.at(np.array([tq]))
        drone = beauty.add_drone(pl, p[0], R[0])
        out.append(render(pl))
        pl.remove_actor(drone)
    pl.close()
    return out


def shot_cloud(xyz, colour, shade, cfg, size, keys, n_frames, point_size, alpha):
    """A cloud under a moving camera."""
    pl = make_plotter(*size)
    add_cloud(pl, xyz, colour, point_size, alpha, shade)
    out = []
    for k in range(n_frames):
        apply(pl, interp(keys, k / max(n_frames - 1, 1)))
        out.append(render(pl))
    pl.close()
    return out


def _section_camera(pl, x0, y_range, half_height):
    key = xs.frame_camera(x0, y_range, half_height)
    pl.camera_position = [key.pos, key.focal, key.up]
    pl.enable_parallel_projection()
    pl.camera.parallel_scale = key.parallel_scale


def section_points(run_dir, cfg, hf, x0, y_range, exaggeration, half_width=3.0):
    """Blue and green bare-earth sections on a shared grid."""
    from ..map import raster as rast

    cell = int(cfg.map.cell * 100)
    out = {}
    for label in ("blue", "green"):
        r = rast.load(run_dir / "map" / f"{label}_raster_{cell}cm.npz")
        out[label] = xs.raster_section(r, hf, x0, half_width, y_range,
                                       exaggeration)
    return xs.align_sections(*out["blue"], *out["green"])


def shot_section(pts, colour, cfg, size, n_frames, x0, y_range, half_height,
                 point_size=2.4, alpha=0.85):
    pl = make_plotter(*size)
    xs.build(pl, pts, colour, x0, y_range, point_size, alpha)
    _section_camera(pl, x0, y_range, half_height)
    out = [render(pl) for _ in range(n_frames)]
    pl.close()
    return out


def shot_morph(pts_blue, pts_green, cfg, size, n_frames, x0, y_range,
               half_height, stagger, seed=0, point_size=2.4, alpha=0.85):
    """The collapse: the bare-earth surface walking onto the truth."""
    phase = stagger_phase(pts_blue.shape[0], stagger, seed)
    pl = make_plotter(*size)
    mesh, _ = add_cloud(pl, pts_blue, BLUE, point_size, alpha)
    from .scene import add_profile_line
    add_profile_line(pl, xs.flat_reference(x0, *y_range), "#FFFFFF", 3.0)
    _section_camera(pl, x0, y_range, half_height)

    out = []
    for k in range(n_frames):
        u = k / max(n_frames - 1, 1)
        mesh.points = morph_positions(pts_blue, pts_green, u, phase)
        mesh["rgb"] = morph_colour(BLUE, GREEN, u, phase)
        out.append(render(pl))
    pl.close()
    return out


def shot_pullback(green, shade, cfg, size, n_frames, point_size, alpha):
    centre = (cfg.scene.survey_x * 0.5, cfg.scene.survey_y * 0.5, 1.5)
    keys = [three_quarter(centre, 70.0, 24.0, -120.0),
            three_quarter(centre, 130.0, 33.0, -95.0),
            three_quarter(centre, 205.0, 46.0, -70.0)]
    return shot_cloud(green, GREEN, shade, cfg, size, keys, n_frames,
                      point_size, alpha)


# ---------------------------------------------------------------------------


def build(run_dir: Path, fps: int = 30, max_points: int = 2_500_000,
          scale: float = 1.0, keep_frames: bool = False,
          wheat_points: int = 260000, exaggeration: float = 6.0,
          verbose: bool = True) -> dict:
    t_start = time.time()
    reader = RunReader(run_dir)
    cfg = load_run_config(run_dir)
    hf = _heightfield(reader)
    truth = reader.truth_trajectory()
    rep = load_report(run_dir)

    width = int(cfg.render.width * scale)
    height = int(cfg.render.height * scale)
    size = (width, height)
    ps = max(cfg.render.point_size * scale, 1.2)
    alpha = cfg.render.point_alpha

    (blue, green, shade), (blue_full, green_full, shade_full) = _load_clouds(
        run_dir, max_points)
    if verbose:
        print(f"clouds: {blue_full.shape[0]/1e6:.2f} M points, "
              f"{blue.shape[0]/1e6:.2f} M in the wide shots, "
              f"{width}x{height} at {fps} fps")

    from ..sim.scene import build_scene
    scene_mesh = build_scene(cfg.scene, np.random.default_rng(cfg.run.seed))

    x0 = cfg.render.section_x
    out = run_dir / "render"
    out.mkdir(parents=True, exist_ok=True)

    gate6 = smoothness(blue, green, cfg.render.morph_frames,
                       cfg.render.morph_stagger)
    gate6["exaggeration"] = float(exaggeration)
    if verbose:
        print(f"gate 6 morph: worst single-frame step is "
              f"{gate6['max_frame_fraction']:.0%} of a point's travel "
              f"({'PASS' if gate6['pass'] else 'FAIL'})")

    def n(seconds):
        return max(int(round(seconds * fps)), 1)

    with Frames(out / "lidar_demo.mp4", fps,
                out / "frames" if keep_frames else None) as vid:
        # 1. beauty
        if verbose:
            print("shot 1: beauty")
        b_frames = shot_beauty(cfg, hf, scene_mesh, truth, size, n(9.0),
                               wheat_points)
        for k, f in enumerate(b_frames):
            vid.add(caption(f, "a drone surveys a wheat field",
                            "32-beam LiDAR, 15 m, eight passes and a crossing line",
                            alpha=smoothstep(min(k / n(1.2), 1.0))))

        # 2. reveal: the crop dissolves, the aircraft's own map fades in
        if verbose:
            print("shot 2: reveal")
        centre = (cfg.scene.survey_x * 0.62, cfg.scene.survey_y * 0.42, 3.0)
        reveal_keys = [three_quarter(centre, 74.0, 15.0, 146.0)]
        cloud_frames = shot_cloud(blue, BLUE, shade, cfg, size, reveal_keys,
                                  n(2.5), ps * 1.3, min(alpha * 1.8, 0.9))
        last = b_frames[-1]
        for k in range(n(2.5)):
            u = smoothstep(k / n(2.5))
            vid.add(caption(dissolve(last, cloud_frames[k], u),
                            "the map the drone thinks it made",
                            "its own trajectory, and the mounting angle off the drawing",
                            alpha=u))
        vid.hold(caption(cloud_frames[-1], "the map the drone thinks it made",
                         "its own trajectory, and the mounting angle off the drawing"),
                 1.6)

        # 3. failure one: drift
        if verbose:
            print("shot 3: drift close-up")
        props = [p for p in reader.props() if p["kind"] in ("post", "scarecrow")
                 and 20 < p["xy"][0] < cfg.scene.survey_x - 20
                 and 20 < p["xy"][1] < cfg.scene.survey_y - 20]
        target = (np.array(props[len(props) // 2]["xy"]) if props
                  else np.array([cfg.scene.survey_x / 2, cfg.scene.survey_y / 2]))
        from ..sim.terrain import sample_bilinear
        tz = float(sample_bilinear(hf, target[None, :], "terrain")[0])
        focus = (float(target[0]), float(target[1]), tz + 1.0)
        push = [three_quarter(focus, 110.0, 46.0, -108.0),
                three_quarter(focus, 46.0, 52.0, -122.0)]
        for f in shot_cloud(blue_full, BLUE, shade_full, cfg, size, push, n(5.0),
                            ps * 1.6, min(alpha * 2.0, 0.9)):
            vid.add(caption(f, "first failure: the trajectory drifted",
                            "rows doubled, the fence post smeared into a streak"))
        vid.hold(caption(f, "first failure: the trajectory drifted",
                         "rows doubled, the fence post smeared into a streak"), 1.2)

        # 4. failure two: corrugation
        if verbose:
            print("shot 4: cross-section")
        y_range = (max(cfg.scene.survey_y - 5.0 * cfg.flight.spacing, 0.0),
                   cfg.scene.survey_y)
        half_h = 0.55 * exaggeration
        sec_b, sec_g = section_points(run_dir, cfg, hf, x0, y_range, exaggeration)
        if verbose:
            print(f"  section: {sec_b.shape[0]} cells, "
                  f"{y_range[0]:.0f}-{y_range[1]:.0f} m across the lines")
        sec_blue = shot_section(sec_b, BLUE, cfg, size, n(0.1), x0, y_range,
                                half_h, ps * 1.8, 0.85)
        cap2 = ("second failure: the LiDAR is not bolted on where the drawing says",
                f"bare-earth surface minus the true ground at x = {x0:.0f} m, "
                f"vertical scale {exaggeration:.0f}x; one ridge per flight line")
        for k in range(n(1.2)):
            vid.add(caption(dissolve(np.zeros_like(sec_blue[0]), sec_blue[0],
                                     smoothstep(k / n(1.2))), *cap2,
                            alpha=smoothstep(k / n(1.2))))
        vid.hold(caption(sec_blue[0], *cap2), 4.2)

        # 5. the collapse
        if verbose:
            print("shot 5: the collapse")
        morph = shot_morph(sec_b, sec_g, cfg, size, cfg.render.morph_frames,
                           x0, y_range, half_h, cfg.render.morph_stagger,
                           point_size=ps * 1.8, alpha=0.85)
        for k, f in enumerate(morph):
            u = k / max(len(morph) - 1, 1)
            vid.add(caption(f, "the same returns, through the solved trajectory "
                               "and the solved mount",
                            "every point walks from where it was to where it belongs",
                            alpha=1.0 - 0.25 * u))
        vid.hold(caption(morph[-1], "the same returns, through the solved "
                                    "trajectory and the solved mount",
                         "the ground is a line again"), 3.0)

        # 6. title card
        if verbose:
            print("shot 6: title card")
        card = render_card(rep, width, height, morph[-1], dim=0.18)
        for k in range(n(0.8)):
            vid.add(dissolve(morph[-1], card, smoothstep(k / n(0.8))))
        vid.hold(card, 5.0)

        # 7. pull back
        if verbose:
            print("shot 7: pull back")
        pull = shot_pullback(green, shade, cfg, size, n(6.0), ps * 1.3,
                             min(alpha * 1.8, 0.9))
        for k in range(n(1.0)):
            vid.add(dissolve(card, pull[0], smoothstep(k / n(1.0))))
        for f in pull:
            vid.add(caption(f, "the corrected map", ""))
        vid.hold(caption(pull[-1], "the corrected map", ""), 1.5)

    summary = {
        "video": str(out / "lidar_demo.mp4"),
        "frames": vid.n,
        "seconds": vid.n / fps,
        "fps": fps,
        "size": [width, height],
        "points": int(blue.shape[0]),
        "vertical_exaggeration": exaggeration,
        "gate6_morph": gate6,
        "wall_seconds": time.time() - t_start,
    }
    (out / "sequence.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"\nwrote {summary['video']}: {vid.n} frames, "
              f"{summary['seconds']:.0f} s  [{summary['wall_seconds']:.0f} s]")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-points", type=int, default=2_500_000)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resolution multiplier; 0.5 for a quick preview")
    ap.add_argument("--wheat-points", type=int, default=260000)
    ap.add_argument("--exaggeration", type=float, default=6.0)
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args(argv)

    s = build(Path(args.run), args.fps, args.max_points, args.scale,
              args.keep_frames, args.wheat_points, args.exaggeration)
    return 0 if s["gate6_morph"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
