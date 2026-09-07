"""Run directory reading and writing.

One recorded run is a directory of plain numpy files plus a manifest.  The
manifest is the contract: it carries the frame conventions and the *nominal*
extrinsic, and nothing else.  Truth lives in a sibling directory that the
estimation code never opens.

The single most important detail in the whole demo is in the sweep files.
Points are stored in the **sensor frame at the point's own timestamp**, exactly
as a spinning LiDAR delivers them: range times a fixed beam direction, with no
motion compensation and no world-frame projection.  Storing world points would
bake in an answer about where the drone was and where the sensor pointed, and
the entire piece rests on reprojecting one fixed set of returns through two
different trajectories and two different extrinsics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import frames

SCHEMA_VERSION = 1

CLASS_NAMES = {
    0: "none",
    1: "ground",
    2: "canopy_healthy",
    3: "canopy_stressed",
    4: "tree_trunk",
    5: "tree_crown",
    6: "building",
    7: "roof",
    8: "windmill",
    9: "post",
    10: "scarecrow",
}


# ---------------------------------------------------------------------------
# small containers
# ---------------------------------------------------------------------------


@dataclass
class Extrinsic:
    """``T_BS``: ``p_B = R @ p_S + t``."""

    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def from_rpy(rpy_deg, t) -> "Extrinsic":
        return Extrinsic(frames.rpy_zyx_to_R(*rpy_deg), np.asarray(t, dtype=float))

    @property
    def rpy_deg(self) -> np.ndarray:
        return frames.R_to_rpy_zyx(self.R)

    def inverse(self) -> "Extrinsic":
        R, t = frames.invert(self.R, self.t)
        return Extrinsic(R, t)

    def to_dict(self) -> dict[str, Any]:
        return {"rpy_deg": [float(v) for v in self.rpy_deg],
                "t_BS_m": [float(v) for v in self.t],
                "R_BS": [[float(v) for v in row] for row in self.R]}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Extrinsic":
        return Extrinsic(np.asarray(d["R_BS"], dtype=float),
                         np.asarray(d["t_BS_m"], dtype=float))


@dataclass
class Sweep:
    """One LiDAR revolution, as stored."""

    index: int
    t_start: float
    t_offset: np.ndarray     # (N,) f32, seconds after t_start
    xyz: np.ndarray          # (N, 3) f32, sensor frame at the point's own time
    range: np.ndarray        # (N,) f32
    intensity: np.ndarray    # (N,) f32
    ring: np.ndarray         # (N,) u8
    col: np.ndarray          # (N,) u16

    @property
    def t_abs(self) -> np.ndarray:
        return self.t_start + self.t_offset.astype(np.float64)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


@dataclass
class DenseTraj:
    """A trajectory on a dense clock, body to world."""

    t: np.ndarray            # (T,)
    p: np.ndarray            # (T, 3)
    R: np.ndarray            # (T, 3, 3)
    v: np.ndarray | None = None
    label: str = ""
    extrinsic: Extrinsic | None = None
    kf_t: np.ndarray | None = None
    kf_idx: np.ndarray | None = None

    def at(self, t_query: np.ndarray):
        """Interpolated ``(R, p)`` at arbitrary times."""
        return frames.interp_poses(self.t, self.p, self.R, t_query)

    def __len__(self) -> int:
        return int(self.t.size)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


class RunWriter:
    """Creates and fills one run directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        for sub in ("sweeps", "scene", "truth"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ---- sweeps ----

    def write_sweep(self, sweep: Sweep) -> Path:
        path = self.root / "sweeps" / f"sweep_{sweep.index:06d}.npz"
        np.savez(path,
                 t_start=np.float64(sweep.t_start),
                 t_offset=sweep.t_offset.astype(np.float32),
                 xyz=sweep.xyz.astype(np.float32),
                 range=sweep.range.astype(np.float32),
                 intensity=sweep.intensity.astype(np.float32),
                 ring=sweep.ring.astype(np.uint8),
                 col=sweep.col.astype(np.uint16))
        return path

    def write_sweep_truth(self, index: int, col_t: np.ndarray, col_R_WB: np.ndarray,
                          col_p_WB: np.ndarray, hit_class: np.ndarray) -> Path:
        """Per-column true poses.

        These make deskewing testable to machine precision instead of only
        testable against itself, and they never enter the reconstruction.
        """
        path = self.root / "truth" / f"sweep_truth_{index:06d}.npz"
        np.savez(path,
                 col_t=col_t.astype(np.float64),
                 col_R_WB=col_R_WB.astype(np.float32),
                 col_p_WB=col_p_WB.astype(np.float64),
                 hit_class=hit_class.astype(np.uint8))
        return path

    # ---- streams ----

    def write_imu(self, t, accel, gyro) -> Path:
        path = self.root / "imu.npz"
        np.savez(path, t=np.asarray(t, np.float64),
                 accel=np.asarray(accel, np.float64),
                 gyro=np.asarray(gyro, np.float64))
        return path

    def write_gnss(self, t, p_W, valid, sigma, imu_index) -> Path:
        path = self.root / "gnss.npz"
        np.savez(path, t=np.asarray(t, np.float64),
                 p_W=np.asarray(p_W, np.float64),
                 valid=np.asarray(valid, bool),
                 sigma=np.asarray(sigma, np.float64),
                 imu_index=np.asarray(imu_index, np.int64))
        return path

    def write_truth_trajectory(self, **arrays) -> Path:
        path = self.root / "truth" / "trajectory.npz"
        np.savez(path, **arrays)
        return path

    def write_truth_extrinsic(self, ext: Extrinsic) -> Path:
        path = self.root / "truth" / "extrinsic_true.json"
        path.write_text(json.dumps(ext.to_dict(), indent=2))
        return path

    def write_heightfield(self, **arrays) -> Path:
        path = self.root / "scene" / "terrain_heightfield.npz"
        np.savez(path, **arrays)
        return path

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        return path


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


class RunReader:
    """Reads a run directory.

    ``truth_*`` accessors exist for gate checks and rendering.  The estimation
    package must not call them, and the tests assert that it does not.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())

    # ---- manifest-derived ----

    @property
    def extrinsic_nominal(self) -> Extrinsic:
        return Extrinsic.from_dict(self.manifest["extrinsic_nominal"])

    @property
    def n_sweeps(self) -> int:
        return int(self.manifest["lidar"]["n_sweeps"])

    @property
    def sweep_period(self) -> float:
        return float(self.manifest["lidar"]["sweep_period_s"])

    # ---- streams ----

    def imu(self):
        d = np.load(self.root / "imu.npz")
        return d["t"], d["accel"], d["gyro"]

    def gnss(self):
        d = np.load(self.root / "gnss.npz")
        return d["t"], d["p_W"], d["valid"], d["sigma"], d["imu_index"]

    def sweep(self, index: int) -> Sweep:
        d = np.load(self.root / "sweeps" / f"sweep_{index:06d}.npz")
        return Sweep(index=index, t_start=float(d["t_start"]), t_offset=d["t_offset"],
                     xyz=d["xyz"], range=d["range"], intensity=d["intensity"],
                     ring=d["ring"], col=d["col"])

    def sweeps(self, start: int = 0, stop: int | None = None,
               step: int = 1) -> Iterator[Sweep]:
        stop = self.n_sweeps if stop is None else stop
        for k in range(start, stop, step):
            yield self.sweep(k)

    def sweep_start_times(self) -> np.ndarray:
        return np.arange(self.n_sweeps) * self.sweep_period

    # ---- truth (gates and render only) ----

    def truth_trajectory(self) -> DenseTraj:
        d = np.load(self.root / "truth" / "trajectory.npz")
        return DenseTraj(t=d["t"], p=d["p_WB"], R=d["R_WB"],
                         v=d["v_WB"] if "v_WB" in d else None, label="truth")

    def truth_arrays(self) -> dict[str, np.ndarray]:
        d = np.load(self.root / "truth" / "trajectory.npz")
        return {k: d[k] for k in d.files}

    def truth_extrinsic(self) -> Extrinsic:
        return Extrinsic.from_dict(
            json.loads((self.root / "truth" / "extrinsic_true.json").read_text()))

    def truth_sweep(self, index: int) -> dict[str, np.ndarray]:
        d = np.load(self.root / "truth" / f"sweep_truth_{index:06d}.npz")
        return {k: d[k] for k in d.files}

    def heightfield(self) -> dict[str, np.ndarray]:
        d = np.load(self.root / "scene" / "terrain_heightfield.npz")
        return {k: d[k] for k in d.files}

    def props(self) -> list[dict[str, Any]]:
        path = self.root / "scene" / "props.json"
        return json.loads(path.read_text()) if path.exists() else []


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------


def save_traj(path: str | Path, traj: DenseTraj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "t": np.asarray(traj.t, np.float64),
        "p": np.asarray(traj.p, np.float64),
        "R": np.asarray(traj.R, np.float64),
        "label": np.array(traj.label),
    }
    if traj.v is not None:
        arrays["v"] = np.asarray(traj.v, np.float64)
    if traj.extrinsic is not None:
        arrays["E_R"] = traj.extrinsic.R
        arrays["E_t"] = traj.extrinsic.t
        arrays["E_rpy_deg"] = traj.extrinsic.rpy_deg
    if traj.kf_t is not None:
        arrays["kf_t"] = np.asarray(traj.kf_t, np.float64)
    if traj.kf_idx is not None:
        arrays["kf_idx"] = np.asarray(traj.kf_idx, np.int64)
    np.savez(path, **arrays)
    return path


def load_traj(path: str | Path) -> DenseTraj:
    d = np.load(path, allow_pickle=False)
    ext = Extrinsic(d["E_R"], d["E_t"]) if "E_R" in d else None
    return DenseTraj(t=d["t"], p=d["p"], R=d["R"],
                     v=d["v"] if "v" in d else None,
                     label=str(d["label"]) if "label" in d else "",
                     extrinsic=ext,
                     kf_t=d["kf_t"] if "kf_t" in d else None,
                     kf_idx=d["kf_idx"] if "kf_idx" in d else None)


# ---------------------------------------------------------------------------
# PLY, written straight from numpy so nothing has to be installed
# ---------------------------------------------------------------------------


def write_ply_points(path: str | Path, xyz: np.ndarray,
                     rgb: np.ndarray | None = None) -> Path:
    """Binary little-endian PLY point cloud."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.ascontiguousarray(np.asarray(xyz, np.float32))
    n = xyz.shape[0]
    if rgb is None:
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        rec = np.empty(n, dtype=dt)
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        props = "property float x\nproperty float y\nproperty float z\n"
    else:
        rgb = np.ascontiguousarray(np.asarray(rgb, np.uint8))
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        rec = np.empty(n, dtype=dt)
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        props = ("property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n")
    header = (f"ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n{props}end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        rec.tofile(fh)
    return path


def write_ply_mesh(path: str | Path, V: np.ndarray, F: np.ndarray) -> Path:
    """Binary little-endian PLY triangle mesh."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    V = np.asarray(V, np.float32)
    F = np.asarray(F, np.int32)
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    vrec = np.empty(V.shape[0], dtype=vdt)
    vrec["x"], vrec["y"], vrec["z"] = V[:, 0], V[:, 1], V[:, 2]
    fdt = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    frec = np.empty(F.shape[0], dtype=fdt)
    frec["n"] = 3
    frec["a"], frec["b"], frec["c"] = F[:, 0], F[:, 1], F[:, 2]
    header = (f"ply\nformat binary_little_endian 1.0\n"
              f"element vertex {V.shape[0]}\n"
              f"property float x\nproperty float y\nproperty float z\n"
              f"element face {F.shape[0]}\n"
              f"property list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        vrec.tofile(fh)
        frec.tofile(fh)
    return path


def read_ply_mesh(path: str | Path):
    """Read back a mesh written by :func:`write_ply_mesh`."""
    with open(path, "rb") as fh:
        n_v = n_f = 0
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n_v = int(line.split()[-1])
            elif line.startswith("element face"):
                n_f = int(line.split()[-1])
            elif line == "end_header":
                break
        vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        vrec = np.fromfile(fh, dtype=vdt, count=n_v)
        fdt = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
        frec = np.fromfile(fh, dtype=fdt, count=n_f)
    V = np.stack([vrec["x"], vrec["y"], vrec["z"]], axis=1).astype(np.float64)
    F = np.stack([frec["a"], frec["b"], frec["c"]], axis=1).astype(np.int64)
    return V, F


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def build_manifest(cfg, n_sweeps: int, ring_elev_deg: np.ndarray,
                   backend: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the manifest.  Only the *nominal* extrinsic goes in here."""
    import datetime as _dt

    E_nom = Extrinsic.from_rpy(cfg.extrinsic.nominal_rpy_deg, cfg.extrinsic.nominal_t)
    man: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_name": cfg.run.name,
        "seed": int(cfg.run.seed),
        "backend": backend,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "frames": {
            "world": "local ENU, z up, origin at survey SW corner",
            "body": "FLU (x forward, y left, z up)",
            "sensor": "z = spin axis, x = azimuth 0, y left",
            "gravity_W": [0.0, 0.0, -9.80665],
            "flu_to_frd": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        },
        "extrinsic_nominal": {
            "convention": ("p_B = R_BS p_S + t_BS; R_BS = Rz(yaw) Ry(pitch) Rx(roll); "
                           "degrees; in FLU a positive pitch tilts the sensor forward "
                           "and down"),
            **E_nom.to_dict(),
        },
        "lidar": {
            "n_rings": int(cfg.lidar.n_rings),
            "n_cols": int(cfg.lidar.n_cols),
            "rate_hz": float(cfg.lidar.rate_hz),
            "sweep_period_s": 1.0 / float(cfg.lidar.rate_hz),
            "ring_elev_deg": [float(v) for v in ring_elev_deg],
            "col_az_deg": "360 * col / n_cols",
            "col_time": "t_start + sweep_period * col / n_cols",
            "min_range_m": float(cfg.lidar.min_range),
            "max_range_m": float(cfg.lidar.max_range),
            "range_sigma_m": float(cfg.lidar.range_sigma),
            "point_frame": ("sensor frame at the point's own timestamp; NOT motion "
                            "compensated and NOT projected into the world"),
            "n_sweeps": int(n_sweeps),
            "file_pattern": "sweeps/sweep_{:06d}.npz",
        },
        "imu": {
            "rate_hz": float(cfg.imu.rate_hz),
            "file": "imu.npz",
            "frame": "body FLU",
            "accel_is_specific_force": True,
            "accel_noise_density": float(cfg.imu.accel_noise_density),
            "gyro_noise_density": float(cfg.imu.gyro_noise_density),
            "accel_bias_rw": float(cfg.imu.accel_bias_rw),
            "gyro_bias_rw": float(cfg.imu.gyro_bias_rw),
        },
        "gnss": {
            "rate_hz": float(cfg.gnss.rate_hz),
            "file": "gnss.npz",
            "frame": "world ENU",
            "sigma_m": [float(cfg.gnss.sigma_horizontal),
                        float(cfg.gnss.sigma_horizontal),
                        float(cfg.gnss.sigma_vertical)],
            "antenna_lever_arm_B": [0.0, 0.0, 0.0],
            "dropout": (f"invalid within {cfg.gnss.dropout_radius} m of the treeline "
                        f"at y = {cfg.scene.treeline_y}, plus "
                        f"{cfg.gnss.random_dropout:.0%} at random"),
        },
        "scene": {
            "survey_xy": [[0.0, float(cfg.scene.survey_x)],
                          [0.0, float(cfg.scene.survey_y)]],
            "context_m": float(cfg.scene.context),
            "heightfield": "scene/terrain_heightfield.npz",
            "terrain_mesh": "scene/terrain.ply",
            "surface_mesh": "scene/surface.ply",
            "props_mesh": "scene/props.ply",
            "class_ids": {str(k): v for k, v in CLASS_NAMES.items()},
        },
        "truth": {
            "dir": "truth/",
            "note": "render and gate checks only; the estimation package never reads it",
        },
    }
    if extra:
        man.update(extra)
    return man
