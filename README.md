# agspray

Simulation study: does retroactive state correction reduce chemical waste in
agricultural spraying, given that spray deposition is irreversible?

Two tiers:

* `agspray/` - synthetic harness. Analytic trajectory, synthesized sensors,
  six estimators behind one interface, coverage rasters, metrics, sweeps.
  Every number in the writeup comes from here.
* `tier2/` - Isaac Lab replay of the same runs for video and stills. No
  statistics come from this tier.

Alongside them, `lidar_demo/` is a separate piece of work: a drone LiDAR survey
whose map is broken by navigation drift and by a LiDAR bolted on two degrees off
the drawing, and a GTSAM factor graph that recovers the trajectory and the
mounting angle together. It shares `agspray`'s trajectory, IMU and measurement
helpers and nothing else. See `lidar_demo/README.md`.

## Setup (tier 1, WSL / Linux)

GTSAM only ships Linux and macOS wheels, so the harness runs under WSL2 on a
Windows host.

```
scripts/wsl_setup.sh
source ~/venvs/agspray/bin/activate
pytest
python -m agspray.runner --config configs/default.yaml --methods all
```

## Layout

```
agspray/
  config.py            dataclasses + yaml loading, derived quantities
  trajectory.py        boustrophedon ground truth in SE(3)
  world.py             field, plants (landmarks), treeline, sun
  degradation.py       wind, canopy motion, occlusion, multipath, sun washout
  sensors.py           IMU / GNSS / camera synthesis
  measurement_models.py shared residuals + jacobians
  data_association.py  gated, descriptor-scored association
  estimators/          dead reckoning, EKF-SLAM, GTSAM graph family
  spray.py             coverage rasters and gating
  metrics.py           physical and estimation metrics
  runner.py            one trial
  sweeps.py            monte carlo grids, checkpointed
  tuning.py            per-method tuning on a held-out config
  analysis/            figures
  viz/                 matplotlib coverage animation
configs/               yaml configs
tests/
tier2/                 Isaac Lab scene, replay, render
lidar_demo/            LiDAR boresight demo (see its own README)
```
