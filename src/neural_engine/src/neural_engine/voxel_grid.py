#!/usr/bin/env python3
"""Lightweight voxel grid for accumulating pointcloud observations."""

import itertools
import numpy as np
import sensor_msgs.point_cloud2 as pc2


class VoxelGrid:
    """Maintains an axis-aligned occupancy grid over the ROI."""

    def __init__(self, bounds, grid_dims=(64, 64, 24)):
        """
        Args:
            bounds: dict with x_min/x_max/... describing ROI limits.
            grid_dims: tuple with number of voxels along (x, y, z).
        """
        self.bounds = bounds
        self.grid_dims = np.array(grid_dims, dtype=np.int32)
        self.extents = np.array(
            [
                bounds["x_max"] - bounds["x_min"],
                bounds["y_max"] - bounds["y_min"],
                bounds["z_max"] - bounds["z_min"],
            ],
            dtype=np.float32,
        )
        self.min_corner = np.array(
            [bounds["x_min"], bounds["y_min"], bounds["z_min"]], dtype=np.float32
        )
        self.voxel_size = self.extents / self.grid_dims.astype(np.float32)
        self.surface_mask = np.zeros(self.grid_dims[:2], dtype=bool)
        self.reset()
        self._compute_surface_mask()

    def reset(self):
        """Clear occupancy grid."""
        self.occupancy = np.zeros(self.grid_dims, dtype=bool)
        self.coverage = 0
        self._surface_covered = np.zeros_like(self.surface_mask, dtype=bool)

    def _points_to_indices(self, points):
        """Convert Nx3 points to voxel indices."""
        if points.size == 0:
            return None
        rel = (points - self.min_corner) / self.extents
        idx = np.floor(rel * self.grid_dims).astype(np.int32)
        mask = np.all((idx >= 0) & (idx < self.grid_dims), axis=1)
        if not np.any(mask):
            return None
        return idx[mask]

    def integrate_points(self, points):
        """Update grid with Nx3 numpy points in world frame."""
        if points is None or points.size == 0:
            return 0
        idx = self._points_to_indices(points)
        if idx is None:
            return 0
        before = self.coverage
        unique = np.unique(idx, axis=0)
        self.occupancy[unique[:, 0], unique[:, 1], unique[:, 2]] = True
        surface_hits = self.surface_mask[unique[:, 0], unique[:, 1]]
        self._surface_covered[unique[surface_hits, 0], unique[surface_hits, 1]] = True
        self.coverage = int(self.occupancy.sum())
        return self.coverage - before

    def is_surface_covered(self, threshold=0.98):
        total = max(int(self.surface_mask.sum()), 1)
        covered = int(self._surface_covered.sum())
        return covered / total >= threshold

    def _compute_surface_mask(self):
        x_dim, y_dim, z_dim = self.grid_dims
        mask = np.zeros((x_dim, y_dim), dtype=bool)
        mask[:, :] = True
        self.surface_mask = mask

    def integrate_pointcloud(self, cloud_msg):
        """Convert ROS PointCloud2 to numpy and integrate."""
        if cloud_msg is None:
            return 0
        iterator = pc2.read_points(
            cloud_msg, field_names=("x", "y", "z"), skip_nans=True
        )
        flat = np.fromiter(
            (coord for point in iterator for coord in point),
            dtype=np.float32,
        )
        if flat.size == 0:
            return 0
        pts_np = flat.reshape(-1, 3)
        return self.integrate_points(pts_np)

    def get_occupancy_grid(self):
        """Return a copy of the occupancy array."""
        return self.occupancy.copy()

    def get_coverage_ratio(self):
        """Coverage relative to total voxels."""
        total = np.prod(self.grid_dims)
        return float(self.coverage) / float(total)

    def get_occupied_voxel_centers(self):
        """Return centers of occupied voxels as Nx3 array."""
        idx = np.argwhere(self.occupancy)
        if idx.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        centers = self.min_corner + (idx + 0.5) * self.voxel_size
        return centers.astype(np.float32)

    def get_depth_embedding(self, flatten=True):
        """
        Generate a simple depth map by projecting occupancy along +Z.

        Returns:
            depth map normalized to [0, 1] as np.float32 array
            (flattened if requested).
        """
        occ = self.occupancy
        if occ.size == 0:
            return np.zeros(int(np.prod(self.grid_dims[:2])), dtype=np.float32)

        depth = np.zeros(self.grid_dims[:2], dtype=np.float32)
        mask = occ.any(axis=2)
        if mask.any():
            first_idx = np.argmax(occ, axis=2)
            normalized = first_idx.astype(np.float32) / max(self.grid_dims[2] - 1, 1)
            depth[mask] = normalized[mask]

        if flatten:
            return depth.flatten()
        return depth
