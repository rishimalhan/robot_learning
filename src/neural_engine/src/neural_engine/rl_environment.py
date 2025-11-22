#!/usr/bin/env python3

# External

import rospy
import numpy as np
import trimesh
from dataclasses import dataclass, field
from gymnasium import Env, spaces
from tf.transformations import (
    euler_matrix,
    quaternion_matrix,
    quaternion_from_matrix,
    euler_from_quaternion,
    quaternion_from_euler,
    euler_from_matrix,
)
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

# Internal

from core.utils import sample_roi_poses, get_roi_info, get_param, resolve_package_path
from neural_engine.simulated_perception import SimulatedPerception
from neural_engine.voxel_grid import VoxelGrid


@dataclass
class CameraContext:
    """Lightweight container for camera pose/action data."""

    position: np.ndarray | None = None
    orientation: np.ndarray | None = None
    prev_action: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )


class InspectionEnv(Env):
    """Gymnasium environment that wraps the MuJoCo-based simulated perception."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        vert_angle: float = 30.0,
        horz_angle: float = 45.0,
        publish_pointcloud: bool = False,
        visualize_grid: bool = False,
    ):
        super().__init__()
        self._perception = SimulatedPerception(manual_camera_transform=np.eye(4))
        self._vert_angle = vert_angle
        self._horz_angle = horz_angle
        self._publish_pointcloud = publish_pointcloud
        self._visualize_grid = visualize_grid
        self._roi_info = get_roi_info()
        if self._roi_info is None:
            raise RuntimeError(
                "robot_roi must exist in the planning scene before using InspectionEnv."
            )

        self._last_render = None
        self._part_rotation = np.eye(3)
        self._part_centroid = np.zeros(3)
        self._part_bounds = self._compute_part_bounds()
        self._voxel_grid = VoxelGrid(self._part_bounds)
        self._step_penalty = 0.1
        self._ctx = CameraContext()

        depth_dim = int(np.prod(self._voxel_grid.grid_dims[:2]))
        state_dim = depth_dim + 6 + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self._grid_marker_pub = None
        if self._visualize_grid:
            self._grid_marker_pub = rospy.Publisher("/voxel_grid", Marker, queue_size=1)

    def _sample_camera_pose(self):
        """Sample a camera pose using the ROI constraints."""
        pose = sample_roi_poses(1, self._vert_angle, self._horz_angle)[0]
        return pose

    def _apply_pose(self, pose):
        """Apply the sampled pose to the MuJoCo sensor via manual transform."""
        position = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float32
        )
        quat = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float32,
        )
        euler = self._wrap_angles(
            np.array(euler_from_quaternion(quat), dtype=np.float32)
        )
        quat_for_sim = quaternion_from_euler(*euler)
        self._perception.set_manual_camera_pose(
            position.tolist(), quat_for_sim.tolist()
        )
        self._ctx.position = position
        self._ctx.orientation = euler

    def _compute_part_bounds(self):
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._voxel_grid.reset()
        pose = self._sample_camera_pose()
        self._apply_pose(pose)
        pointcloud = self._perception.trigger(publish=self._publish_pointcloud)
        self._voxel_grid.integrate_pointcloud(pointcloud)
        if self._visualize_grid:
            self._publish_voxel_grid()
        self._ctx.prev_action = np.zeros_like(self._ctx.prev_action)
        state = self._build_state()
        info = {
            "pose_world": self._pose_world_vec(),
            "coverage": self._voxel_grid.coverage,
            "camera_pose_local": self._camera_pose_local(),
        }
        return state, info

    def step(self, action):
        """Apply action (6D delta pose) in local frame, step environment."""

        action = np.asarray(action, dtype=np.float32)
        if action.shape != self._ctx.prev_action.shape:
            raise ValueError("Action must be 6-dimensional delta pose.")
        # Compute new local pose
        pose_local = self._camera_pose_local()
        if pose_local is None:
            raise RuntimeError("Camera pose is not set.")
        pos_local = pose_local[:3] + action[:3]
        ori_local = self._wrap_angles(pose_local[3:] + action[3:])
        rot_local_new = euler_matrix(*ori_local)[:3, :3]
        # Convert to world frame
        pos_world = self._part_rotation @ pos_local + self._part_centroid
        rot_world = self._part_rotation @ rot_local_new
        euler_world = self._wrap_angles(
            np.array(euler_from_matrix(rot_world), dtype=np.float32)
        )
        # Bounds check in world frame
        out_of_bounds = not self._is_within_bounds(pos_world)
        surface_done = False
        if out_of_bounds:
            info = {
                "pose_world": self._pose_world_vec(),
                "coverage": self._voxel_grid.coverage,
                "delta_coverage": 0.0,
                "camera_pose_local": pose_local,
                "violated_bounds": True,
            }
            return self._build_state(), -1.0, False, True, info
        # Apply pose
        self._ctx.position = pos_world.astype(np.float32)
        self._ctx.orientation = euler_world
        self._perception.set_manual_camera_pose(
            self._ctx.position.tolist(),
            quaternion_from_euler(*self._ctx.orientation).tolist(),
        )
        self._ctx.prev_action = action

        # Capture observation
        pointcloud = self._perception.trigger(publish=self._publish_pointcloud)
        delta_cov = self._voxel_grid.integrate_pointcloud(pointcloud)
        if self._visualize_grid:
            self._publish_voxel_grid()
        state = self._build_state()
        total_cov = float(self._voxel_grid.coverage) or 1.0
        surface_done = self._voxel_grid.is_surface_covered()
        reward = (float(delta_cov) / total_cov) - self._step_penalty * (1.0 - surface_done)
        info = {
            "pose_world": self._pose_world_vec(),
            "camera_pose_local": self._camera_pose_local(),
            "action": action,
            "coverage": self._voxel_grid.coverage,
            "delta_coverage": delta_cov,
            "reward": reward,
            "terminated": False,
            "surface_covered": surface_done,
        }
        return state, reward, surface_done, False, info

    def _publish_voxel_grid(self):
        """Publish occupied voxels as Marker for visualization."""
        if not self._visualize_grid or self._grid_marker_pub is None:
            return
        centers = self._voxel_grid.get_occupied_voxel_centers()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "voxel_grid"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        scale = float(np.min(self._voxel_grid.voxel_size))
        if scale <= 0:
            scale = 0.01
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 0.2
        marker.color.a = 0.4
        marker.points = [
            Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in centers
        ]
        self._grid_marker_pub.publish(marker)

    def _camera_pose_local(self):
        """Return current camera pose expressed in part-centric coordinates."""
        if self._ctx.position is None or self._ctx.orientation is None:
            return None
        pos_world = self._ctx.position
        rel_pos = pos_world - self._part_centroid
        pos_local = self._part_rotation.T @ rel_pos

        rot_world = euler_matrix(*self._ctx.orientation)[:3, :3]
        rot_local = self._part_rotation.T @ rot_world
        rot_local_h = np.eye(4)
        rot_local_h[:3, :3] = rot_local
        euler_local = self._wrap_angles(
            np.array(euler_from_matrix(rot_local_h), dtype=np.float32)
        )

        return np.concatenate([pos_local, euler_local]).astype(np.float32)

    def _pose_world_vec(self):
        if self._ctx.position is None or self._ctx.orientation is None:
            return None
        return np.concatenate([self._ctx.position, self._ctx.orientation]).astype(
            np.float32
        )

    def _build_state(self):
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

    @staticmethod
    def _wrap_angles(angles):
        """Wrap angles to [-pi, pi]."""
        return ((angles + np.pi) % (2 * np.pi)) - np.pi

    def _is_within_bounds(self, pos_world):
        bounds = self._part_bounds
        return (
            bounds["x_min"] <= pos_world[0] <= bounds["x_max"]
            and bounds["y_min"] <= pos_world[1] <= bounds["y_max"]
            and bounds["z_min"] <= pos_world[2] <= bounds["z_max"]
        )

    def render(self):
        """Return the latest RGB image from the simulated sensor."""
        rgb, depth = self._perception.render_rgb_depth()
        self._last_render = (rgb, depth)
        return rgb

    def close(self):
        self._perception.cleanup()
        super().close()


if __name__ == "__main__":
    import time

    rospy.init_node("inspection_env_sanity", disable_signals=False)
    num_evals = 10
    env = InspectionEnv(publish_pointcloud=False, visualize_grid=False)
    start_time = time.time()
    obs, info = env.reset()
    print(f"Reset complete. Pointcloud present={obs is not None}")
    end_time = time.time()
    print(f"Time taken to reset: {end_time - start_time} seconds")

    start_time = time.time()
    for i in range(num_evals):
        mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        sigma = np.array([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
        action = np.random.normal(mean, sigma)
        obs, reward, done, terminated, info = env.step(action)
        print(f"Step -> {i}, info={info}\n")
        time.sleep(0.5)
    end_time = time.time()
    print(
        f"Time taken to step {num_evals} times: {end_time - start_time} seconds. Avg: {(end_time - start_time) / num_evals} seconds."
    )

    env.close()
