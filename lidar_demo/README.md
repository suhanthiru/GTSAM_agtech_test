# GTSAM in action: repairing a broken LiDAR map

A drone surveys a wheat field. Its navigation drifts, and its LiDAR is bolted on
a couple of degrees off where the drawing says. The resulting map is broken in
two recognisable ways. A factor graph then solves for the trajectory *and* the
mounting angle at once, and the map snaps into shape.

From **one** recorded set of returns the pipeline builds **two** maps:

| | trajectory and mount used | colour |
|---|---|---|
| blue | the aircraft's own belief, nominal mount | `#2E6FD6` |
| green | the graph's answer, recovered mount | `#1D9E75` |

Same returns. A different answer about where the drone was and where the sensor
was pointing. That difference is the whole film.

## What comes out

On `data/run01`, seed 0:

| | |
|---|---|
| mounting angle, true | roll 0.800, pitch 36.500, yaw 2.000 degrees |
| mounting angle, recovered | roll 0.804, pitch 36.466, yaw 2.013, +/- 0.03 |
| error | **0.036 degrees**, against a 0.2 degree gate |
| trajectory, blue vs truth | 1.33 m XY rms, 0.79 m Z, 1.35 degrees attitude |
| trajectory, green vs truth | 0.21 m XY rms, 0.09 m Z, 0.15 degrees attitude |
| bare earth, blue vs truth | 29 cm rms, with a 59 cm ripple at the flight lines |
| bare earth, green vs truth | 12 cm rms |
| the two flight directions disagree | 24 cm in blue, 2.3 cm in green |

## Running it

Only the factor graph needs WSL, because GTSAM has no Windows wheel. Everything
else runs on the Windows host: Open3D provides the ray caster and the
registration backend there, and pyvista the renderer. The two sides exchange a
run directory, so either can be swapped for the other on any step except the
solve.

```bash
scripts/wsl_setup.sh          # once: creates ~/venvs/agspray with gtsam
```

```bash
wsl -d Ubuntu-24.04 -e bash -lc "source ~/venvs/agspray/bin/activate && pip install open3d small_gicp"
```

```bash
python -m pip install open3d pyvista imageio imageio-ffmpeg pillow
```

Both registration backends are available on either side. `register.py` prefers
`small_gicp` and falls back to Open3D's generalised ICP, and a test pins the two
to the same transform convention rather than assuming it.

Then, in order:

```bash
python -m lidar_demo.checks.preview_scene --config configs/lidar_demo.yaml
```

```bash
python -m lidar_demo.checks.check_imu_drift --config configs/lidar_demo.yaml
```

```bash
scripts/wpy -m lidar_demo.sim.record --config configs/lidar_demo.yaml
```

```bash
scripts/wpy -m lidar_demo.checks.gate1_truth_project --run data/run01
```

```bash
scripts/wpy -m lidar_demo.est.solve --run data/run01 --threads 12
```

```bash
scripts/wpy -m lidar_demo.map.build --run data/run01
```

```bash
python -m lidar_demo.render.gates --run data/run01
```

```bash
python -m lidar_demo.render.sequence --run data/run01
```

Recording takes about a minute, the solve about six, the map forty seconds and
the film ninety seconds. `--scale 0.4 --fps 15` gives a preview render in ten
seconds.

## Layout

```
lidar_demo/
  frames.py      the frame conventions, and the only place they are defined
  config.py      dataclasses and yaml, mirroring agspray/config.py
  io.py          the run directory: manifest, sweeps, streams, PLY
  sim/           terrain, props, flight, LiDAR, IMU, GNSS, the recorder
  est/           deskew, the blue filter, registration, the graph, the solve
  map/           projection, rasters, the two clouds
  render/        the gate figures first, then the seven shots
  checks/        scene preview, IMU drift, gate 1
```

## The gates

Purely visual, and in order. Nothing proceeds past a failing one.

| | | |
|---|---|---|
| 1 | truth-projected cloud matches the terrain | 1.6 cm rms at 380 pts/m2 |
| 2 | blue drift lands in the 1-2 m band | 1.33 m rms |
| 3 | corrugation visible, one ridge per flight line | 59 cm peak to peak |
| 4 | green hugs the true profile | 12 cm rms, 2.5x closer than blue |
| 5 | mounting angle within 0.2 degrees | 0.036 degrees |
| 6 | the collapse reads smoothly | worst frame moves 5% of a point's travel |

`gate1_truth_project.py` runs twice, and the two runs have to disagree.
Projected through the mount the simulator built, the cloud sits on the terrain.
Projected through the mount the pipeline is told about, it lifts 27 cm and every
swath tilts, alternating sign on all seven adjacent pairs.

## Things that were not obvious

**Sweeps are stored in the sensor frame, uncompensated.** Store world points and
an answer about where the drone was is already baked in, and reprojecting the
same returns through two trajectories becomes impossible. Every azimuth column
is cast from the true pose at its own sub-sweep instant, so the recording
carries a real motion smear for the deskewing to remove.

**Registering adjacent keyframes is worthless here.** Their submaps are built
from the same sweeps, so the matcher hands back the seed trajectory and the
graph learns nothing while believing it has been told something. Pairs are
stepped by the submap window so the two clouds are independent.

**The footprint is 14 m ahead of the aircraft.** Two keyframes on opposite
passes that are close together are flying towards each other, and their
footprints hardly overlap. Overlap is detected between forward-projected
footprints, not between aircraft.

**Inter-pass pairs are registered only after the along-track solve.** Started
from a seed that is metres out over the GNSS-denied stretch, the matcher slides
along a field that is nearly self-similar, and that bias is indistinguishable
from a mounting-angle error. This one was worth about a degree of yaw.

**The prior on the mount stays at nominal across rounds.** Re-centring it on the
last estimate removes the only absolute anchor and it ratchets away.

**A crop has to have texture in it.** The rows run parallel to the flight lines,
so the furrows constrain a scan matcher across track and not at all along it. A
canopy modelled as a smooth sheet leaves that direction free and registration
slides metres while reporting an excellent fit.

**Pulses go through the canopy.** Without Beer-Lambert transmission there are no
ground returns under standing crop, so there is no bare-earth surface, no
terrain raster, and a cross-section with the ground visible only in the bare
patches.

## Two places the brief needed correcting

**On observability.** The brief says the mounting angle would be unobservable if
every pass flew the same heading. It would not be. The gauge that hides it --
rotate every pose, counter-rotate the mount -- leaves aircraft positions
untouched, so GNSS can never break it, but the IMU's gravity-referenced attitude
can. In the toy survey the tests build, reversing alternate passes roughly
halves the recovered uncertainty rather than deciding whether there is an answer
at all. What the reversal really buys is the picture: the same error tilts an
outbound swath one way and a return swath the other, so the map comes out
corrugated instead of smoothly wrong, and smoothly wrong is far harder to show
anyone. `tests/lidar_demo/test_factors.py` asserts all three claims.

**On the IMU.** The brief asks for a consumer MEMS profile multiplied five to ten
times *and* for 1-2 m of dead-reckoned drift. Those cannot both hold: at that
noise density a 30 second window leaks about eleven metres, and over the whole
survey any MEMS part leaks hundreds. The drift target was kept, because it is
what the viewer sees as ghosting, and the grade adjusted to industrial MEMS.
`check_imu_drift.py` prints both the measured drift and the closed-form budget.

## Isaac Sim

Isaac Sim 5.1 and Isaac Lab 2.3.2 are installed in their own environment at
`D:\isaacenv`, on Python 3.11, because Isaac Lab pins both. `scripts/isaacpy`
runs anything against it, the way `scripts/wpy` does for the WSL side.

```bash
scripts/isaacpy -m lidar_demo.checks.check_isaac --json D:/isaac/check.json
```

Two things had to be settled to get it running, and both are recorded in that
launcher rather than left for the next person to rediscover.

The licence prompt. Kit asks for the NVIDIA Omniverse agreement on first launch
and fails with EOF under any non-interactive shell. The user accepted it on
2026-09-07 and `OMNI_KIT_ACCEPT_EULA=YES` records that.

The graphics API. On this machine, an RTX 3080 Ti on driver 610.74, the Vulkan
backend segfaults inside `rtx.scenedb` about seven seconds in, while the
material library compiles its base MDL shaders. That happens on the first frame,
so it takes every run with it no matter which experience file is used, and no
renderer setting avoids it. On D3D12 the same build starts in nine seconds.
NVIDIA's own compatibility checker runs on D3D12 and reports this GPU and driver
as supported, which is the clue that led there. `LIDAR_DEMO_ISAAC_API=vulkan`
overrides it if a future driver fixes the crash.

## The Isaac backend

`lidar_demo/sim/isaac/` writes the scene into a USD stage and traces the sweeps
against it with warp on the GPU. Everything above the tracer is shared with the
offline path: same scene, same flight, same beam table, same rolling shutter,
same noise, same run directory. Swapping the backend changes who traces the rays
and nothing else, which is what makes the two runs comparable.

```bash
scripts/isaacpy -m lidar_demo.sim.isaac.record --name run_isaac --compare
```

```bash
scripts/isaacpy -m pytest tests/lidar_demo/test_isaac.py -q
```

Over a sample of sweeps cast from identical poses, the two tracers agree to
**0.000 cm** median range difference across 100% of shared beams, and gate 1 on
the Isaac recording reproduces the offline numbers: 1.59 cm against 1.55 cm on
the truth mount, the same 378 pts/m2 and 11.9 m half swath, seven of seven swath
tilts alternating, 17.3 cm of corrugation against 17.5 cm. A full run takes 38
seconds, faster than the Open3D path.

Three decisions in there are worth knowing about.

The backend casts against geometry read back **out** of the stage, not against
the numpy arrays the stage was written from. That way a mistake in the USD
export shows up as a bad cast instead of hiding behind correct source data.

The cast is a warp kernel rather than Isaac Lab's `raycast_mesh`. That wrapper
takes and returns torch tensors, and Isaac Sim 5.1 installs torch 2.7.0+cpu, a
build with no CUDA at all, so every ray would travel through host memory and
trace on the CPU of a machine whose GPU warp is perfectly happy to use. The
kernel is the same `wp.mesh_query_ray` call their wrapper makes, without the
round trip. `LIDAR_DEMO_ISAAC_RAYCAST=isaaclab` switches to theirs.

The pip `isaaclab` package is only a bootstrap stub that expects the source tree
beside it. The real library is the clone at `D:\isaac\IsaacLab`, and
`scripts/isaacpy` puts it first on the path so it shadows the stub.

## Phase two

What remains is the beauty pass. The stage now carries the terrain, the props, a
sun and a drone prim, so Isaac can render the opening shot over the same field
the LiDAR flew, and Blender can read the same file. `render/beauty.py` is still
the pyvista stand-in, and says so.
