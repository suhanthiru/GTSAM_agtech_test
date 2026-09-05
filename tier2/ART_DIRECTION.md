# Tier 2 art direction

Reference supplied by the user: a stylised painterly farm landscape (a still of
a ripe grain field in wind, hills and a farmhouse behind it).  Match its
qualities, not photorealism.

## What to match

| Element | Note |
|---|---|
| Foreground | Ripe golden grain filling the lower half, tall enough that the heads read as a *surface* rather than a texture |
| Wind | Coherent travelling bands running through the canopy - whole patches of heads bending together, not per-stalk noise. **The single most important element.** |
| Mid distance | Soft layered rolling green hills, progressively desaturated and cooler with depth |
| Landmark | A small pale farmhouse with a warm-red roof, middle distance, fixed |
| Sky | Bright saturated blue, soft cumulus, low warm sun, long shadows |
| Edges | Windbreak treeline / hedgerow at one field edge - this is also the GNSS multipath source, so the visual and the physical model must agree on where it is |
| Palette | High saturation, warm golds against cool greens and sky blue, soft edges, no crisp PBR microdetail |
| Extras in ref | A scarecrow and a windmill sit at the field edges; a pale track curves through the midground.  Optional set dressing, not required. |

## Why the wind matters beyond looking good

The canopy motion *is* the static-world violation the whole study is about.
Tier 1 models plant sway as a travelling wave

```
displacement(p, t) = A(t) * w_hat * sin(2*pi * (p . w_hat) / lambda - 2*pi*f*t)
```

(`agspray/degradation.py`, `CanopyMotion`).  Tier 2 must displace the canopy
mesh vertices with **the same equation and the same wind realisation**, so the
bands a viewer sees are literally the disturbance corrupting the visual
measurements.  Do not substitute a decorative shader noise for this.

## Overlays

Coverage layers sit above the canopy as two translucent sheets:

* `believed_coverage` - high-chroma **magenta**
* `true_coverage` - high-chroma **cyan**

Both must read against saturated gold; no warm tones, no earth tones.  Waste is
the visible mismatch between the two sheets, accumulating in real time.

A translucent ghost drone rides at the estimated pose beside the solid true
one.

## Camera

Primary framing is a high three-quarter view showing several rows at once, so
the coverage overlay and the row structure are both readable.  One cutaway low
pass through the canopy for the wind.  Nothing else.

## Where to put the reference

Drop the supplied image at `tier2/reference/field_reference.png` (gitignored by
default; add it with `git add -f` if it should ship with the repo).
