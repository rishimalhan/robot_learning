#!/usr/bin/env python3

import numpy as np
import ros_numpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from typing import List, Set


class GreedySetCover:
    """Reward-based set cover algorithm for viewpoint planning."""

    def __init__(self, min_reward_threshold: float = 0.1):
        """Initialize the reward-based set cover algorithm."""
        self.min_reward_threshold = min_reward_threshold
        self.tolerance = 1e-4

    def plan(
        self, viewpoints: List[PoseStamped], pointclouds: List[PointCloud2]
    ) -> List[PoseStamped]:
        """
        Plan optimal viewpoint sequence using dynamic reward-based greedy set cover.

        Args:
            viewpoints: List of candidate viewpoints
            pointclouds: List of point clouds corresponding to viewpoints

        Returns:
            List of selected viewpoints in optimal order
        """
        # Convert point clouds to numpy arrays
        point_arrays = self._convert_pointclouds_to_numpy(pointclouds)

        # Execute dynamic greedy set cover algorithm
        selected_indices = self._dynamic_greedy_set_cover(point_arrays)

        return [viewpoints[i] for i in selected_indices]

    def _convert_pointclouds_to_numpy(
        self, pointclouds: List[PointCloud2]
    ) -> List[np.ndarray]:
        """Convert ROS PointCloud2 messages to numpy arrays."""
        point_arrays = []

        for cloud in pointclouds:
            # Convert PointCloud2 to numpy array using ros_numpy
            point_array = ros_numpy.numpify(cloud)

            # Extract XYZ coordinates
            xyz_points = np.column_stack(
                (point_array["x"], point_array["y"], point_array["z"])
            )
            # Remove NaN values
            valid_mask = ~np.isnan(xyz_points).any(axis=1)
            xyz_points = xyz_points[valid_mask].astype(np.float32)
            point_arrays.append(xyz_points)

        return point_arrays

    def _dynamic_greedy_set_cover(self, point_arrays: List[np.ndarray]) -> List[int]:
        """
        Execute dynamic greedy set cover algorithm.

        At each step, recalculate rewards based on current coverage state.
        """
        if not point_arrays:
            return []

        n_viewpoints = len(point_arrays)
        selected_sequence = []
        remaining_indices = set(range(n_viewpoints))
        covered_points = set()  # Track covered points as (x, y, z) tuples

        # Start with viewpoint that has most points
        start_idx = self._find_best_starting_viewpoint(point_arrays, remaining_indices)
        selected_sequence.append(start_idx)
        remaining_indices.remove(start_idx)

        # Add all points from starting viewpoint to covered set
        covered_points.update(self._points_to_set(point_arrays[start_idx]))

        # Greedily add viewpoints based on dynamic rewards
        while remaining_indices:
            best_reward = -float("inf")
            best_next_idx = None

            # Calculate current reward for each remaining viewpoint
            for next_idx in remaining_indices:
                current_idx = selected_sequence[-1]  # Last selected viewpoint
                reward = self._calculate_dynamic_reward(
                    point_arrays[current_idx], point_arrays[next_idx], covered_points
                )

                if reward > best_reward and reward > self.min_reward_threshold:
                    best_reward = reward
                    best_next_idx = next_idx

            # If no viewpoint meets threshold, break
            if best_next_idx is None:
                break

            # Add best viewpoint to sequence
            selected_sequence.append(best_next_idx)
            remaining_indices.remove(best_next_idx)

            # Update covered points set
            new_points = self._points_to_set(point_arrays[best_next_idx])
            covered_points.update(new_points)

        return selected_sequence

    def _find_best_starting_viewpoint(
        self, point_arrays: List[np.ndarray], remaining_indices: Set[int]
    ) -> int:
        """Find the best starting viewpoint (one with most points)."""
        best_idx = None
        max_points = 0

        for idx in remaining_indices:
            if len(point_arrays[idx]) > max_points:
                max_points = len(point_arrays[idx])
                best_idx = idx

        return best_idx

    def _calculate_dynamic_reward(
        self, current_points: np.ndarray, next_points: np.ndarray, covered_points: Set
    ) -> float:
        """
        Calculate dynamic reward for transitioning from current to next viewpoint.

        Reward = (truly_new_points_in_next) / (points_in_current)
        where truly_new_points are those in next_points that aren't already covered.
        """
        if len(current_points) == 0:
            return 0.0

        if len(next_points) == 0:
            return -1.0  # Penalty for empty viewpoint

        # Convert next_points to set of tuples for comparison
        next_points_set = self._points_to_set(next_points)

        # Find points in next viewpoint that are truly new (not in covered_points)
        truly_new_points = next_points_set - covered_points

        # Calculate reward: new_points / points_in_current
        reward = len(truly_new_points) / len(current_points)

        return reward

    def _points_to_set(self, points: np.ndarray) -> Set:
        """Convert numpy array of points to set of tuples for fast lookup."""
        if len(points) == 0:
            return set()

        # Round to tolerance precision for consistent comparison
        precision = int(-np.log10(self.tolerance))
        rounded_points = np.round(points, precision)

        # Convert to set of tuples
        return set(map(tuple, rounded_points.tolist()))
