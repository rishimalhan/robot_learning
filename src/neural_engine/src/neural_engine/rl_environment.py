#!/usr/bin/env python3

from __future__ import annotations

# External

import rospy
import numpy as np
import trimesh
from dataclasses import dataclass, field
from typing import Optional, Tuple
from gymnasium import Env, spaces
from tf.transformations import (
    euler_matrix,
    quaternion_from_matrix,
    quaternion_from_euler,
    euler_from_matrix,
)
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

# Internal

from core.utils import get_param, resolve_package_path
from neural_engine.simulated_perception import SimulatedPerception
from neural_engine.voxel_grid import VoxelGrid
from neural_engine.utils import (
    wrap_angles,
    sample_pose_within_roi,
    pose_within_bounds,
    publish_voxel_grid,
    publish_pointcloud,
    publish_frame_marker,
)

XY_PADDING = 0.3
Z_PADDING = 0.1

PoseTuple = Tuple[np.ndarray, np.ndarray]


@dataclass
class CameraContext:
    """Lightweight container for camera pose/action data."""

    position: Optional[np.ndarray] = None
    orientation: Optional[np.ndarray] = None
    prev_action: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )


class InspectionEnv(Env):
    """Gymnasium environment that wraps the MuJoCo-based simulated perception."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        camera_angle: float = 30.0,
        publish_pointcloud: bool = False,
        visualize: bool = False,
        point_stride: int = 1,
    ) -> None:
        """
        Args:
            camera_angle: Maximum tilt (degrees) allowed from global -Z.
            publish_pointcloud: Whether simulated perception publishes ROS clouds.
            visualize: When True, emit RViz markers + debug clouds.
            point_stride: Downsampling factor for depth unprojection (>=1).
        """
        super().__init__()
        self._perception = SimulatedPerception(manual_camera_transform=np.eye(4))
        self._depth_min = self._perception.min_depth
        self._depth_max = self._perception.max_depth
        self._max_tilt = np.deg2rad(camera_angle)
        self._publish_pointcloud = publish_pointcloud
        self._visualize = visualize
        self._point_stride = max(1, int(point_stride))

        self._last_render = None
        self._max_sampling_attempts = 25
        self._part_rotation = np.eye(3)
        self._part_rotation_inv = np.eye(3)
        self._part_centroid = np.zeros(3)
        self._part_bounds = self._compute_part_bounds()
        self._roi_bounds = self._compute_roi_bounds()
        self._voxel_grid = VoxelGrid(self._part_bounds)
        self._step_penalty = 0.1
        self._ctx = CameraContext()
        depth_dim = int(np.prod(self._voxel_grid.grid_dims[:2]))
        state_dim = depth_dim + 6 + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self._marker_pub: Optional[rospy.Publisher] = None
        self._debug_cloud_pub: Optional[rospy.Publisher] = None
        if self._visualize:
            self._marker_pub = rospy.Publisher(
                "/inspection_env/markers", Marker, queue_size=10
            )
            self._debug_cloud_pub = rospy.Publisher(
                "/inspection_env/pointcloud", PointCloud2, queue_size=1
            )

    def _apply_pose(self, position: np.ndarray, euler: np.ndarray) -> None:
        """Apply the provided pose to MuJoCo and cache it in the context."""
        position = np.asarray(position, dtype=np.float32)
        euler = wrap_angles(np.asarray(euler, dtype=np.float32))
        quat_for_sim = quaternion_from_euler(*euler)
        self._perception.set_manual_camera_pose(
            position.tolist(), quat_for_sim.tolist()
        )
        self._ctx.position = position
        self._ctx.orientation = euler

    def _compute_part_bounds(self) -> dict:
        """Compute axis-aligned bounding box for the current part."""
        spec = get_param("/environment/part", None)
        if spec is None:
            raise RuntimeError("Part specification not available on parameter server.")

        mesh_path = resolve_package_path(spec["mesh_path"])
        mesh = trimesh.load(mesh_path, force="mesh")

        scale = spec.get("scale")
        mesh.apply_scale(scale)
        pose = spec.get("pose")
        position = pose.get("position")
        orientation = pose.get("orientation")
        transform = euler_matrix(*orientation)
        transform[:3, 3] = position
        mesh.apply_transform(transform)
        self._part_rotation = transform[:3, :3]
        self._part_rotation_inv = self._part_rotation.T
        self._part_centroid = np.array(mesh.centroid, dtype=np.float32)
        mins, maxs = mesh.bounds
        return {
            "x_min": mins[0],
            "x_max": maxs[0],
            "y_min": mins[1],
            "y_max": maxs[1],
            "z_min": mins[2],
            "z_max": maxs[2],
        }

    def _compute_roi_bounds(self) -> dict:
        """Pad part bounds to define the camera sampling ROI."""
        bounds = self._part_bounds
        x_min = bounds["x_min"] - XY_PADDING
        x_max = bounds["x_max"] + XY_PADDING
        y_min = bounds["y_min"] - XY_PADDING
        y_max = bounds["y_max"] + XY_PADDING
        z_top = bounds["z_max"]
        z_min = z_top + self._depth_min - Z_PADDING
        z_max = z_top + self._depth_max + Z_PADDING
        if z_max <= z_min:
            z_max = z_min + 0.1
        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset MuJoCo, resample a feasible camera pose, and clear voxel memory."""
        super().reset(seed=seed)
        self._voxel_grid.reset()
        sampled_pose: Optional[PoseTuple] = None
        for _ in range(self._max_sampling_attempts):
            position, orientation = sample_pose_within_roi(
                self._roi_bounds, self._max_tilt, shrink_scale=0.7
            )
            if pose_within_bounds(
                position=position,
                orientation=orientation,
                max_tilt=self._max_tilt,
                roi_bounds=self._roi_bounds,
            ):
                sampled_pose = (
                    np.asarray(position, dtype=np.float32),
                    np.asarray(orientation, dtype=np.float32),
                )
                self._apply_pose(*sampled_pose)
                break
        if sampled_pose is None:
            raise RuntimeError("Failed to sample valid pose during reset.")

        trigger_result = self._perception.trigger(
            publish=self._publish_pointcloud,
            downsample=self._point_stride,
        )
        cloud_msg = None
        if isinstance(trigger_result, tuple):
            points_np, _ = trigger_result
            self._voxel_grid.integrate_points(points_np)
        else:
            cloud_msg = trigger_result
            self._voxel_grid.integrate_pointcloud(cloud_msg)
        self._visualize_scene(cloud_msg=cloud_msg)
        self._ctx.prev_action = np.zeros_like(self._ctx.prev_action)
        state = self._build_state()
        surface_done = self._voxel_grid.is_surface_covered()
        info = {
            "pose_world": self._pose_world_vec(),
            "pose_local": self._camera_pose_local(),
            "action": "N/A",
            "coverage": self._voxel_grid.coverage,
            "delta_coverage": "N/A",
            "reward": "N/A",
            "terminated": False,
            "surface_covered": surface_done,
            "out_of_bounds": "N/A",
        }
        return state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Apply a 5-DOF delta in the local frame and return the new transition."""

        action = np.asarray(action, dtype=np.float32)
        action = np.append(action, 0.0) # yaw is fixed
        if action.shape != self._ctx.prev_action.shape:
            raise ValueError("Action must be 6-dimensional delta pose.")
        pose_local = self._camera_pose_local()
        if pose_local is None:
            raise RuntimeError("Camera pose is not set.")
        pos_local = pose_local[:3] + action[:3]
        ori_local = wrap_angles(pose_local[3:] + action[3:]) # roll, pitch
        rot_local_new = euler_matrix(*ori_local)[:3, :3]
        pos_world = self._part_rotation @ pos_local + self._part_centroid
        rot_world = self._part_rotation @ rot_local_new
        euler_world = wrap_angles(
            np.array(euler_from_matrix(rot_world), dtype=np.float32)
        )
        out_of_bounds = not pose_within_bounds(
            position=pos_world,
            orientation=euler_world,
            max_tilt=self._max_tilt,
            roi_bounds=self._roi_bounds,
        )
        if out_of_bounds:
            info = {
                "pose_world": self._pose_world_vec(),
                "pose_local": self._camera_pose_local(),
                "action": action,
                "coverage": self._voxel_grid.coverage,
                "delta_coverage": 0,
                "reward": -1.0,
                "terminated": True,
                "surface_covered": False,
                "out_of_bounds": True,
            }
            return self._build_state(), -1.0, False, True, info

        self._ctx.position = pos_world.astype(np.float32)
        self._ctx.orientation = euler_world
        self._perception.set_manual_camera_pose(
            self._ctx.position.tolist(),
            quaternion_from_euler(*self._ctx.orientation).tolist(),
        )
        self._ctx.prev_action = action.copy()

        trigger_result = self._perception.trigger(
            publish=self._publish_pointcloud,
            downsample=self._point_stride,
        )
        cloud_msg = None
        if isinstance(trigger_result, tuple):
            points_np, _ = trigger_result
            delta_cov = self._voxel_grid.integrate_points(points_np)
        else:
            cloud_msg = trigger_result
            delta_cov = self._voxel_grid.integrate_pointcloud(cloud_msg)
        self._visualize_scene(cloud_msg)
        state = self._build_state()
        total_cov = float(self._voxel_grid.coverage) or 1.0
        surface_done = self._voxel_grid.is_surface_covered()
        reward = (float(delta_cov) / total_cov)
        info = {
            "pose_world": self._pose_world_vec(),
            "pose_local": self._camera_pose_local(),
            "action": action,
            "coverage": self._voxel_grid.coverage,
            "delta_coverage": delta_cov,
            "reward": reward,
            "terminated": False,
            "surface_covered": surface_done,
            "out_of_bounds": "N/A",
        }
        return state, reward, surface_done, False, info

    def _visualize_scene(
        self,
        cloud_msg: Optional[PointCloud2],
    ) -> None:
        """Publish voxel grid, debug pointcloud, and reference frames when enabled."""
        if not self._visualize:
            return
        if self._debug_cloud_pub is not None and cloud_msg is not None:
            publish_pointcloud(self._debug_cloud_pub, cloud_msg)
        if self._marker_pub is None:
            return
        publish_voxel_grid(self._marker_pub, self._voxel_grid)
        publish_frame_marker(
            self._marker_pub,
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "world_axes",
            1,
            length=1.0,
        )
        part_tf = np.eye(4, dtype=np.float64)
        part_tf[:3, :3] = self._part_rotation.astype(np.float64, copy=False)
        part_quat = quaternion_from_matrix(part_tf)
        publish_frame_marker(
            self._marker_pub,
            self._part_centroid,
            part_quat,
            "part_axes",
            2,
            length=0.5,
        )
        curr_quat = quaternion_from_euler(*self._ctx.orientation)
        publish_frame_marker(
            self._marker_pub,
            self._ctx.position,
            curr_quat,
            "camera_active",
            3,
            length=0.1,
        )

    def _camera_pose_local(self) -> Optional[np.ndarray]:
        """Return current camera pose expressed in part-centric coordinates."""
        if self._ctx.position is None or self._ctx.orientation is None:
            return None
        pos_world = self._ctx.position
        rel_pos = pos_world - self._part_centroid
        pos_local = self._part_rotation_inv @ rel_pos

        rot_world = euler_matrix(*self._ctx.orientation)[:3, :3]
        rot_local = self._part_rotation_inv @ rot_world
        rot_local_h = np.eye(4)
        rot_local_h[:3, :3] = rot_local
        euler_local = wrap_angles(
            np.array(euler_from_matrix(rot_local_h), dtype=np.float32)
        )

        return np.concatenate([pos_local, euler_local]).astype(np.float32)

    def _pose_world_vec(self) -> Optional[np.ndarray]:
        if self._ctx.position is None or self._ctx.orientation is None:
            return None
        return np.concatenate([self._ctx.position, self._ctx.orientation]).astype(
            np.float32
        )

    def _build_state(self) -> np.ndarray:
        depth_embedding = self._voxel_grid.get_depth_embedding(flatten=True).astype(
            np.float32
        )
        pose_local = self._camera_pose_local()
        if pose_local is None:
            raise RuntimeError("Camera pose is not set.")
        state = np.concatenate(
            [depth_embedding, pose_local, self._ctx.prev_action]
        ).astype(np.float32)
        return state

    def render(self) -> np.ndarray:
        """Return the latest RGB image from the simulated sensor."""
        rgb, depth = self._perception.render_rgb_depth()
        self._last_render = (rgb, depth)
        return rgb

    def close(self) -> None:
        self._perception.cleanup()
        super().close()


if __name__ == "__main__":
    import time

    rospy.init_node("inspection_env_sanity", disable_signals=False)

    num_evals = 100
    env = InspectionEnv(publish_pointcloud=False, visualize=False, point_stride=4)
    start_time = time.time()
    obs, info = env.reset()
    print(f"Reset complete. Info: {info}\n")
    end_time = time.time()
    print(f"Time taken to reset: {end_time - start_time} seconds")
    episodes = 5
    start_time = time.time()
    counts = 0
    for _ in range(episodes):
        for i in range(num_evals):
            counts += 1
            mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # x, y, z, roll, pitch
            sigma = np.array([0.05, 0.05, 0.05, 0.1, 0.1])
            action = np.random.normal(mean, sigma)
            obs, reward, done, terminated, info = env.step(action)
            print(f"Step -> {i+1}, info={info}\n")
            if done or terminated:
                obs, info = env.reset()
                print(f"Reset complete. Info: {info}\n")
                break
    end_time = time.time()
    print(
        f"Time taken to count {counts}: {end_time - start_time} seconds. Avg: {(end_time - start_time) / counts} seconds."
    )

    env.close()
