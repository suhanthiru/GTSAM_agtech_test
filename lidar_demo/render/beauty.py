"""The opening shot: a sunlit field, before any overlay.

Said plainly, this is a stand-in.  The brief asks for a warm painterly farmland
in late-morning light, and what is here is a stylised approximation of it built
from the same terrain and the same prop positions the simulation used, so the
geometry is honest even though the rendering is not photographic.  It shares
positions with the scene, not materials.

It still has to be genuinely nice to look at.  The reveal that follows lands
against this shot, and if the opening is dull the reveal has nothing to work
against.

Two things are real rather than decorative.  The wheat is instanced on the
canopy the simulator actually built, so the healthy, stressed and bare blocks
are the blocks the LiDAR flew over.  The props stand where the ray caster put
them, so the windmill and the farmhouse that break the horizon here are the same
verticals that make the scan matching converge.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

SKY_TOP = "#2E7BC4"
SKY_HORIZON = "#BFE0F2"
SOIL = "#7A5C3A"
WHEAT = np.array([212, 165, 58], dtype=np.float32)
WHEAT_DARK = np.array([150, 108, 32], dtype=np.float32)
GRASS = "#5E8C42"
ROOF = "#B3402E"
WALL = "#E8E0CE"
TRUNK = "#4A3A28"
CROWN = "#3F6B34"


def add_sky(pl: pv.Plotter) -> None:
    pl.set_background(SKY_HORIZON, top=SKY_TOP)


def add_ground(pl: pv.Plotter, hf, stride: int = 4):
    """Bare earth, warm where the crop is thin and green where it is not."""
    z = hf.z_terrain[::stride, ::stride]
    x = hf.x[::stride]
    y = hf.y[::stride]
    grid = pv.StructuredGrid(*np.meshgrid(x, y), np.asarray(z, np.float32))
    canopy = hf.canopy_h[::stride, ::stride]
    t = np.clip(canopy / max(float(canopy.max()), 1e-6), 0, 1).ravel(order="F")
    # Outside the survey the land is pasture, inside it is worked ground showing
    # between the rows, so the two get different colours and the field reads as
    # a field rather than as a rectangle of the same brown.
    X, Y = np.meshgrid(x, y)
    inside = ((X >= 0) & (X <= hf.x[-1]) & (Y >= 0)).ravel(order="F")
    soil = np.array([132, 104, 66], np.float32)
    pasture = np.array([104, 138, 74], np.float32)
    stubble = np.array([168, 140, 82], np.float32)
    base = np.where(inside[:, None], stubble[None, :], pasture[None, :])
    grid["rgb"] = np.clip(base * (1 - 0.45 * t[:, None])
                          + soil[None, :] * 0.45 * t[:, None],
                          0, 255).astype(np.uint8)
    return pl.add_mesh(grid, scalars="rgb", rgb=True, smooth_shading=True,
                       ambient=0.35, diffuse=0.75, specular=0.05)


def wheat_points(hf, cfg, n: int = 260000, seed: int = 0):
    """Sample the canopy for instancing, denser where the crop is taller."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, cfg.scene.survey_x, n * 2)
    y = rng.uniform(0.0, cfg.scene.survey_y, n * 2)
    from ..sim.terrain import sample_bilinear

    xy = np.stack([x, y], axis=1)
    h = sample_bilinear(hf, xy, "canopy")
    keep = rng.random(x.size) < np.clip(h / max(cfg.scene.canopy_healthy, 1e-6), 0, 1)
    xy, h = xy[keep][:n], h[keep][:n]
    z = sample_bilinear(hf, xy, "terrain") + h
    return np.stack([xy[:, 0], xy[:, 1], z], axis=1), h


def add_wheat(pl: pv.Plotter, hf, cfg, n: int = 260000, seed: int = 0,
              point_size: float = 5.0, opacity: float = 1.0):
    """The crop, as a dense gold stipple.

    Instanced geometry for a quarter of a million stalks costs more than the
    shot is worth; a dense point stipple shaded by height reads as a canopy at
    the distances this sequence uses, and it keeps the frame time in seconds.
    """
    pts, h = wheat_points(hf, cfg, n, seed)
    if pts.shape[0] == 0:
        return None
    # Shade by height within the stand so the canopy has depth: the heads catch
    # the light and the gaps between them fall away.
    t = np.clip(h / max(cfg.scene.canopy_healthy, 1e-6), 0, 1)[:, None]
    rng2 = np.random.default_rng(seed + 1)
    jitter = rng2.normal(0.0, 0.07, (pts.shape[0], 1))
    rgb = WHEAT_DARK[None, :] * (1 - t) + WHEAT[None, :] * t
    rgb = np.clip(rgb * (1.0 + jitter), 0, 255).astype(np.uint8)
    mesh = pv.PolyData(np.ascontiguousarray(pts, np.float32))
    mesh["rgb"] = rgb
    actor = pl.add_mesh(mesh, scalars="rgb", rgb=True, point_size=point_size,
                        render_points_as_spheres=True, opacity=opacity,
                        lighting=False)
    return mesh, actor


def add_props(pl: pv.Plotter, scene_mesh) -> list:
    """The verticals, in flat colour, from the same meshes the LiDAR saw."""
    palette = {
        "tree_trunks": TRUNK, "tree_crowns": CROWN,
        "windmill": "#D8D2C4", "farmhouse_walls": WALL, "farmhouse_roof": ROOF,
        "outbuilding_walls": WALL, "outbuilding_roof": ROOF,
        "scarecrow": "#8A6A45", "posts": "#6B5537",
    }
    out = []
    for part in scene_mesh.parts:
        colour = palette.get(part.name)
        if colour is None or part.F.shape[0] == 0:
            continue
        faces = np.hstack([np.full((part.F.shape[0], 1), 3, np.int64), part.F])
        poly = pv.PolyData(np.ascontiguousarray(part.V, np.float32), faces.ravel())
        out.append(pl.add_mesh(poly, color=colour, smooth_shading=False,
                               ambient=0.30, diffuse=0.80))
    return out


def add_sun(pl: pv.Plotter, cfg) -> None:
    """A low warm key light, from the azimuth the scene was lit at."""
    az = np.radians(cfg.scene.slope_azimuth_deg + 150.0)
    el = np.radians(22.0)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    centre = np.array([cfg.scene.survey_x / 2, cfg.scene.survey_y / 2, 0.0])
    light = pv.Light(position=tuple(centre + 400.0 * d), focal_point=tuple(centre),
                     color="#FFE3B0", intensity=0.95)
    light.positional = False
    pl.add_light(light)
    fill = pv.Light(position=tuple(centre + np.array([-200.0, -150.0, 250.0])),
                    focal_point=tuple(centre), color="#9FC4E8", intensity=0.35)
    fill.positional = False
    pl.add_light(fill)


def add_drone(pl: pv.Plotter, p, R, size: float = 1.6):
    """A small marker at the aircraft, so the opening reads as a survey."""
    body = pv.Cube(center=(0, 0, 0), x_length=size, y_length=size,
                   z_length=size * 0.25)
    body = body.rotate_vector(vector=(0, 0, 1),
                              angle=np.degrees(np.arctan2(R[1, 0], R[0, 0])),
                              inplace=False)
    body.translate(np.asarray(p, float), inplace=True)
    return pl.add_mesh(body, color="#2A2E33", ambient=0.4, diffuse=0.7)
