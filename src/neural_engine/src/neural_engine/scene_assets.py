#!/usr/bin/env python3
"""Centralized definition of inspection part assets."""

import os
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import rospkg
import trimesh
from tf.transformations import euler_matrix

PART_NAME = "ascent_mold"
PART_SCALE = np.array([1.0, 1.0, 1.0], dtype=np.float32)
PART_POSITION = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # desired centroid
PART_ORIENTATION = np.array([0.0, 0.0, 0.0], dtype=np.float32)


@lru_cache()
def _assets_root() -> str:
    pkg_root = rospkg.RosPack().get_path("inspection_cell")
    return os.path.join(pkg_root, "assets")


@lru_cache()
def _scaled_mesh(part_name: str) -> trimesh.Trimesh:
    mesh_path = os.path.join(_assets_root(), f"{part_name}.stl")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh for part '{part_name}' not found at {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(PART_SCALE)
    return mesh


def get_part_mesh_path(part_name: str = PART_NAME) -> str:
    """Return the absolute path to the STL mesh for the configured part."""
    return os.path.join(_assets_root(), f"{part_name}.stl")


@lru_cache()
def get_part_spec(part_name: str = PART_NAME) -> Dict[str, object]:
    """Return a reusable spec dictionary describing the current inspection part."""
    mesh = _scaled_mesh(part_name)
    centroid = mesh.centroid.astype(np.float32)
    mins, _ = mesh.bounds

    target_xy = PART_POSITION[:2]
    target_z = PART_POSITION[2]

    translation = np.array(
        [
            target_xy[0] - centroid[0],
            target_xy[1] - centroid[1],
            target_z - mins[2],
        ],
        dtype=np.float32,
    )

    return {
        "name": part_name,
        "mesh_path": get_part_mesh_path(part_name),
        "pose": {
            "position": translation.tolist(),
            "orientation": PART_ORIENTATION.tolist(),
        },
        "scale": PART_SCALE.tolist(),
    }


def load_part_mesh(
    apply_pose: bool = True,
) -> Tuple[trimesh.Trimesh, Dict[str, object]]:
    """Load the configured part mesh and optionally apply its canonical pose."""
    spec = get_part_spec()
    mesh = _scaled_mesh(spec["name"]).copy()
    if apply_pose:
        transform = euler_matrix(*spec["pose"]["orientation"])
        transform[:3, 3] = spec["pose"]["position"]
        mesh.apply_transform(transform)
    return mesh, spec
