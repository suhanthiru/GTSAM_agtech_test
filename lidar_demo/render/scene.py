"""Offscreen plotting: clouds, terrain, and how the layers are combined.

Point styling follows the brief.  Small points, moderate alpha, near-black
ground, and density doing the work.  The temptation with a sparse-looking cloud
is to enlarge the points; that trades a cloud that reads as a surface for one
that reads as confetti, so the point size stays at one or two pixels and the
answer to thin coverage is more points.

A note on additive blending.  VTK's point mapper has no additive blend mode, so
it is approximated in two places.  Within a cloud, depth peeling lets many
translucent points accumulate honestly.  Across layers -- the field, the blue
cloud, the green cloud -- each is rendered to its own image over black and the
frames are added in image space, which is exactly additive and costs one extra
render per layer.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

BLUE = "#2E6FD6"
GREEN = "#1D9E75"
BACKGROUND = "#05070A"


def hex_rgb(colour: str) -> np.ndarray:
    c = colour.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def make_plotter(width: int, height: int, background: str = BACKGROUND,
                 peels: int = 6) -> pv.Plotter:
    pl = pv.Plotter(off_screen=True, window_size=(int(width), int(height)))
    pl.set_background(background)
    try:
        pl.enable_depth_peeling(number_of_peels=peels, occlusion_ratio=0.0)
    except Exception:
        pass
    return pl


def add_cloud(pl: pv.Plotter, xyz: np.ndarray, colour, point_size: float = 1.5,
              alpha: float = 0.35, shade: np.ndarray | None = None):
    """A point cloud with a flat hue, optionally shaded by a per-point scalar."""
    mesh = pv.PolyData(np.ascontiguousarray(xyz, np.float32))
    base = hex_rgb(colour) if isinstance(colour, str) else np.asarray(colour, np.float32)
    if shade is None:
        rgb = np.tile(base[None, :], (xyz.shape[0], 1))
    else:
        s = np.clip(np.asarray(shade, np.float32), 0.0, 1.0)[:, None]
        rgb = base[None, :] * (0.40 + 0.60 * s)
    mesh["rgb"] = np.clip(rgb, 0, 255).astype(np.uint8)
    actor = pl.add_mesh(mesh, scalars="rgb", rgb=True, point_size=point_size,
                        render_points_as_spheres=False, opacity=alpha,
                        lighting=False)
    return mesh, actor


def add_terrain(pl: pv.Plotter, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                colour="#6B5637", opacity: float = 1.0, **kwargs):
    """A heightfield as a lit surface."""
    grid = pv.StructuredGrid(*np.meshgrid(x, y), np.asarray(z, np.float32))
    return pl.add_mesh(grid, color=colour, opacity=opacity,
                       smooth_shading=True, **kwargs)


def add_profile_line(pl: pv.Plotter, pts: np.ndarray, colour="#FFFFFF",
                     width: float = 3.0):
    """The truth profile drawn through a cross-section."""
    line = pv.lines_from_points(np.ascontiguousarray(pts, np.float32))
    return pl.add_mesh(line, color=colour, line_width=width, lighting=False)


def render(pl: pv.Plotter) -> np.ndarray:
    """Grab a frame.

    The explicit ``render()`` is load-bearing.  Mutating a mesh in place --
    which is how the morph moves its points and shifts their colour -- does not
    on its own mark the pipeline dirty, so without it every frame of the morph
    comes back identical to the first and the collapse silently becomes a
    still.  Camera moves happen to trigger a redraw by themselves, which is
    exactly why the fault hides in the one shot that does not move the camera.
    """
    pl.render()
    img = pl.screenshot(return_img=True)
    return np.asarray(img)[:, :, :3].astype(np.float32)


def composite(layers, weights=None) -> np.ndarray:
    """Add rendered layers in image space.  This is the additive blend."""
    weights = [1.0] * len(layers) if weights is None else weights
    out = np.zeros_like(layers[0], dtype=np.float32)
    for img, w in zip(layers, weights):
        if w <= 0.0:
            continue
        out += w * img
    return np.clip(out, 0, 255).astype(np.uint8)


def dissolve(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    """Cross-fade two frames.  Never a cut: the reveal has to read as one."""
    a, b = np.asarray(a), np.asarray(b)
    u = float(np.clip(u, 0.0, 1.0))
    return np.clip((1.0 - u) * a.astype(np.float32)
                   + u * b.astype(np.float32), 0, 255).astype(np.uint8)


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def caption(img: np.ndarray, text: str, sub: str = "", alpha: float = 1.0,
            pos=(0.045, 0.90)) -> np.ndarray:
    """Burn a line of text into a frame."""
    from PIL import Image, ImageDraw

    img = np.clip(np.asarray(img), 0, 255).astype(np.uint8)
    if alpha <= 0.01 or not text:
        return img
    pil = Image.fromarray(img).convert("RGB")
    layer = Image.new("RGB", pil.size, (0, 0, 0))
    d = ImageDraw.Draw(layer)
    font, small = _fonts(pil.size[1])
    x = int(pos[0] * pil.size[0])
    y = int(pos[1] * pil.size[1])
    d.text((x, y), text, fill=(238, 238, 238), font=font)
    if sub:
        d.text((x, y + int(pil.size[1] * 0.045)), sub, fill=(165, 175, 185),
               font=small)
    out = np.maximum(np.asarray(pil, np.float32),
                     np.asarray(layer, np.float32) * float(np.clip(alpha, 0, 1)))
    return np.clip(out, 0, 255).astype(np.uint8)


_FONT_CACHE: dict = {}


def _fonts(height: int):
    key = int(height)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    from PIL import ImageFont

    size = max(int(height * 0.030), 14)
    small = max(int(height * 0.021), 11)
    for name in ("segoeui.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            f = (ImageFont.truetype(name, size), ImageFont.truetype(name, small))
            _FONT_CACHE[key] = f
            return f
        except OSError:
            continue
    f = (ImageFont.load_default(), ImageFont.load_default())
    _FONT_CACHE[key] = f
    return f
