"""The title card: what the mount was, and what the graph found.

Small type, held briefly.  The claim it makes is the whole argument, so the
numbers on it come straight from the solve rather than being retyped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (5, 7, 10)
FG = (232, 236, 240)
DIM = (150, 162, 174)
GREEN = (29, 158, 117)
RED = (200, 70, 60)


def _font(px: int):
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _sans(px: int):
    for name in ("segoeui.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def render_card(rep: dict, width: int, height: int,
                background: np.ndarray | None = None,
                dim: float = 0.10) -> np.ndarray:
    """One 'sensor mounting angle' card, optionally over a dimmed frame."""
    if background is None:
        img = Image.new("RGB", (width, height), BG)
    else:
        # Dim the held frame hard and lay a panel over the type, so the card
        # reads as a card rather than as text sitting on top of a picture.
        base = np.clip(background.astype(np.float32) * dim, 0, 255)
        panel = np.zeros_like(base)
        y0, y1 = int(height * 0.13), int(height * 0.87)
        base[y0:y1] = 0.35 * base[y0:y1] + 0.65 * panel[y0:y1]
        img = Image.fromarray(base.astype(np.uint8))
    d = ImageDraw.Draw(img)

    title = _sans(max(int(height * 0.038), 18))
    mono = _font(max(int(height * 0.030), 15))
    small = _sans(max(int(height * 0.022), 12))

    x = int(width * 0.13)
    y = int(height * 0.20)
    d.text((x, y), "sensor mounting angle", fill=FG, font=title)
    y += int(height * 0.085)

    col = [0, int(width * 0.22), int(width * 0.34), int(width * 0.46)]
    for label, hdr in zip(col[1:], ("roll", "pitch", "yaw")):
        d.text((x + label, y), f"{hdr:>8}", fill=DIM, font=mono)
    y += int(height * 0.055)

    rows = [("nominal", rep["nominal_rpy_deg"], DIM, ""),
            ("recovered", rep["recovered_rpy_deg"], FG,
             "+/- " + "  ".join(f"{s:.3f}" for s in rep["recovered_sigma_deg"]))]
    if "true_rpy_deg" in rep:
        rows.append(("true", rep["true_rpy_deg"], DIM, ""))

    for name, rpy, colour, extra in rows:
        d.text((x, y), f"{name}", fill=colour, font=mono)
        for c, v in zip(col[1:], rpy):
            d.text((x + c, y), f"{v:>8.3f}", fill=colour, font=mono)
        if extra:
            d.text((x + col[3] + int(width * 0.10), y), extra, fill=DIM, font=small)
        y += int(height * 0.052)

    y += int(height * 0.045)
    if "error_deg" in rep:
        ok = bool(rep["pass"])
        d.text((x, y),
               f"the drawing was wrong by {rep['nominal_error_deg']:.2f} degrees",
               fill=DIM, font=small)
        y += int(height * 0.038)
        d.text((x, y),
               f"the graph found it to within {rep['error_deg']:.3f} degrees",
               fill=GREEN if ok else RED, font=small)
        y += int(height * 0.055)
    d.text((x, y),
           "no filter can do this: the mounting angle is not in its state, so no",
           fill=DIM, font=small)
    d.text((x, y + int(height * 0.034)),
           "measurement it processes has any path to it.",
           fill=DIM, font=small)
    return np.asarray(img)


def load_report(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "est" / "boresight.json").read_text())
