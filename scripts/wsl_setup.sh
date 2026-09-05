#!/usr/bin/env bash
# One-time environment setup for the synthetic harness (Linux / WSL2).
set -euo pipefail
VENV="${VENV:-$HOME/venvs/agspray}"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -e ".[graph,dev]"
python - <<'PY'
import gtsam
print("gtsam", gtsam.__version__)
for n in ["IncrementalFixedLagSmoother", "ISAM2", "PreintegratedCombinedMeasurements",
          "GenericProjectionFactorCal3_S2", "GPSFactor", "LevenbergMarquardtOptimizer"]:
    assert hasattr(gtsam, n), n
print("ok")
PY
