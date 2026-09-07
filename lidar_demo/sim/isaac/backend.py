"""The Isaac ray-cast backend: warp kernels tracing against a USD stage.

Fulfils the same contract as the numpy and Open3D backends -- world-frame ray
origins and unit directions in, distances, surface normals and class ids out --
so the recorder above it does not change at all.  The beam table, the rolling
shutter, the noise model and the export are shared, which is the point: swapping
the backend changes who traces the rays and nothing else, and the two runs can
then be compared sweep for sweep.

The meshes come from Isaac Lab's ``convert_to_warp_mesh``, and they are built
from geometry read back *out* of the USD stage rather than from the numpy arrays
the stage was written from, so a mistake in the export shows up as a bad cast
instead of hiding behind correct source data.

The cast itself is a warp kernel here rather than Isaac Lab's ``raycast_mesh``,
and that is a deliberate choice worth explaining.  ``raycast_mesh`` takes and
returns torch tensors, and Isaac Sim 5.1 installs **torch 2.7.0+cpu** -- a build
with no CUDA at all -- so every ray would have to travel through host memory and
the trace would run on the CPU, on a machine whose GPU warp is perfectly happy
to use.  The kernel below is the same ``wp.mesh_query_ray`` call that Isaac Lab
makes, without the torch round trip, and it runs on the card.  Set
``LIDAR_DEMO_ISAAC_RAYCAST=isaaclab`` to use their wrapper instead and see the
difference.
"""

from __future__ import annotations

import os

import numpy as np

from ..backend import RayHits

_KERNEL = None


def _build_kernel():
    """Compile the ray-cast kernel once.

    Identical in substance to the one inside Isaac Lab's ``raycast_mesh``: a
    single ``wp.mesh_query_ray`` per thread, returning distance, face normal and
    face index, with a sentinel for a miss.
    """
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL

    import warp as wp

    @wp.kernel
    def raycast(mesh_id: wp.uint64,
                origins: wp.array(dtype=wp.vec3),
                dirs: wp.array(dtype=wp.vec3),
                far: float,
                t_out: wp.array(dtype=float),
                n_out: wp.array(dtype=wp.vec3),
                f_out: wp.array(dtype=wp.int32)):
        i = wp.tid()
        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        face = int(0)
        if wp.mesh_query_ray(mesh_id, origins[i], dirs[i], far, t, u, v, sign, n, face):
            t_out[i] = t
            n_out[i] = n
            f_out[i] = face
        else:
            t_out[i] = -1.0
            f_out[i] = -1

    _KERNEL = raycast
    return _KERNEL


class IsaacLabBackend:
    """Ray casting through warp, against geometry held in a USD stage."""

    name = "isaaclab"

    def __init__(self, device: str | None = None, stage=None,
                 usd_path: str | None = None, use_isaaclab_raycast: bool | None = None):
        self.device = device
        self.usd_path = usd_path
        self._stage = stage
        self._layout: dict[str, list[str]] = {}
        self._meshes: dict[str, object] = {}
        self._face_class: dict[str, np.ndarray] = {}
        self._n_faces: dict[str, int] = {}
        if use_isaaclab_raycast is None:
            use_isaaclab_raycast = (
                os.environ.get("LIDAR_DEMO_ISAAC_RAYCAST", "").lower() == "isaaclab")
        self.use_isaaclab_raycast = bool(use_isaaclab_raycast)

    # ---- construction ----

    def build(self, scene) -> None:
        """Write the scene into a USD stage and build a warp mesh per layer."""
        import warp as wp
        from isaaclab.utils.warp import convert_to_warp_mesh

        from . import stage as stage_mod

        wp.init()
        if self.device is None:
            # Warp's own view of the machine, not torch's: the torch that ships
            # with Isaac Sim is CPU-only and would send everything to the host.
            self.device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
        if self.use_isaaclab_raycast:
            import torch

            if not torch.cuda.is_available():
                self.device = "cpu"

        if self._stage is None:
            if self.usd_path:
                self._stage, self._layout = stage_mod.export(scene, self.usd_path)
            else:
                self._stage, self._layout = stage_mod.build_stage(scene)
        elif not self._layout:
            raise ValueError("a prebuilt stage must come with its layout")

        for layer, paths in self._layout.items():
            V, F, cls = stage_mod.read_layer(self._stage, paths)
            if F.shape[0] == 0:
                continue
            self._meshes[layer] = convert_to_warp_mesh(
                np.ascontiguousarray(V, np.float32),
                np.ascontiguousarray(F, np.int32), device=self.device)
            self._face_class[layer] = cls
            self._n_faces[layer] = int(F.shape[0])

        if "surface" not in self._meshes:
            raise RuntimeError("the stage has no surface geometry to cast against")
        _build_kernel()

    @property
    def stage(self):
        return self._stage

    @property
    def layout(self) -> dict[str, list[str]]:
        return dict(self._layout)

    def triangles(self, layer: str = "surface") -> int:
        return self._n_faces.get(layer, 0)

    # ---- casting ----

    def cast(self, origins_W: np.ndarray, dirs_W: np.ndarray,
             layer: str = "surface") -> RayHits:
        mesh = self._meshes.get(layer)
        origins_W = np.ascontiguousarray(origins_W, np.float32)
        dirs_W = np.ascontiguousarray(dirs_W, np.float32)
        n = int(origins_W.shape[0])
        if mesh is None or n == 0:
            return RayHits(t_hit=np.full(n, np.inf, np.float32),
                           normal=np.zeros((n, 3), np.float32),
                           class_id=np.zeros(n, np.uint8),
                           prim_id=np.full(n, -1, np.int64))

        if self.use_isaaclab_raycast:
            t, nrm, fid = self._cast_isaaclab(mesh, origins_W, dirs_W)
        else:
            t, nrm, fid = self._cast_warp(mesh, origins_W, dirs_W)

        # A miss comes back as a sentinel face; make the distance infinite, which
        # is what every backend promises its caller.
        miss = (fid < 0) | (fid >= self._n_faces[layer]) | ~np.isfinite(t) | (t < 0)
        t = np.where(miss, np.inf, t).astype(np.float32)
        cls = np.zeros(n, dtype=np.uint8)
        ok = ~miss
        if ok.any():
            cls[ok] = self._face_class[layer][fid[ok]]
        nrm = nrm.copy()
        nrm[miss] = 0.0
        fid = fid.copy()
        fid[miss] = -1
        return RayHits(t_hit=t, normal=nrm.astype(np.float32), class_id=cls,
                       prim_id=fid.astype(np.int64))

    def _cast_warp(self, mesh, origins_W, dirs_W):
        import warp as wp

        n = origins_W.shape[0]
        dev = self.device
        o = wp.array(origins_W, dtype=wp.vec3, device=dev)
        d = wp.array(dirs_W, dtype=wp.vec3, device=dev)
        t = wp.zeros(n, dtype=float, device=dev)
        nn = wp.zeros(n, dtype=wp.vec3, device=dev)
        ff = wp.zeros(n, dtype=wp.int32, device=dev)
        wp.launch(_build_kernel(), dim=n,
                  inputs=[mesh.id, o, d, 1.0e6, t, nn, ff], device=dev)
        wp.synchronize_device(dev)
        return (t.numpy().astype(np.float32), nn.numpy().astype(np.float32),
                ff.numpy().astype(np.int64))

    def _cast_isaaclab(self, mesh, origins_W, dirs_W):
        import torch
        from isaaclab.utils.warp import raycast_mesh

        dev = self.device
        starts = torch.as_tensor(origins_W, device=dev)
        dirs = torch.as_tensor(dirs_W, device=dev)
        _, dist, normal, face = raycast_mesh(
            starts, dirs, mesh, max_dist=1.0e6,
            return_distance=True, return_normal=True, return_face_id=True)
        return (dist.detach().cpu().numpy().astype(np.float32).reshape(-1),
                normal.detach().cpu().numpy().astype(np.float32).reshape(-1, 3),
                face.detach().cpu().numpy().astype(np.int64).reshape(-1))
