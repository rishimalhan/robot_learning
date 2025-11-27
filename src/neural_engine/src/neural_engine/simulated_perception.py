#!/usr/bin/env python3

# External

import os
import shutil
import tempfile
import mujoco
import numpy as np
import rospkg
import rospy
import tf
import yaml
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from tf.transformations import (
    quaternion_from_euler,
    quaternion_matrix,
    quaternion_from_matrix,
)

# Internal

from core.utils import convert_stl_to_obj
from neural_engine.scene_assets import get_part_spec


class SimulatedPerception:
    POINTCLOUD_FIELDS = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("r", 12, PointField.FLOAT32, 1),
        PointField("g", 16, PointField.FLOAT32, 1),
        PointField("b", 20, PointField.FLOAT32, 1),
    ]

    def __init__(self, camera_frame=None, manual_camera_transform=None):
        package_root = rospkg.RosPack().get_path("neural_engine")
        config_path = os.path.join(package_root, "config", "perception.yaml")
        with open(config_path, "r") as cfg:
            config = yaml.safe_load(cfg)

        self.xml_template_path = os.path.join(
            package_root, "config", "inspection_sensor_scene.xml"
        )

        sensor_cfg = config["sensor"]
        intr = sensor_cfg["intrinsics"]
        self.frame_rate = sensor_cfg["dynamics"]["frame_rate"]
        self.width = intr["width"]
        self.height = intr["height"]
        self.fx = intr["fx"]
        self.fy = intr["fy"]

        depth_cfg = sensor_cfg["constraints"]["depth"]
        self.min_depth = depth_cfg["min"]
        self.max_depth = depth_cfg["max"]
        self.max_radius = sensor_cfg["constraints"]["radius"]
        self.max_orientation = sensor_cfg["constraints"]["orientation"]

        floor_cfg = sensor_cfg.get("floor", {})
        self.floor_height = floor_cfg.get("height", 0.0)
        self.floor_margin = floor_cfg.get("margin", 0.02)
        self.temp_dir = tempfile.mkdtemp(prefix="inspection_sim_")
        self.model = None
        self.data = None
        self.camera_id = None
        self.mocap_id = None
        self._env_signature = None
        self._camera_transform_mujoco = None
        self._manual_camera_transform = manual_camera_transform
        self.camera_frame = camera_frame
        self.tf_listener = None
        if self._manual_camera_transform is None and self.camera_frame is None:
            self.camera_frame = sensor_cfg.get("frame")
        self.tf_listener = tf.TransformListener()

        self.pointcloud_pub = rospy.Publisher(
            "/camera/points", PointCloud2, queue_size=1
        )
        self._load_environment(wait=True)
        self.monitor_timer = rospy.Timer(rospy.Duration(1.0), self._monitor_environment)

    def _monitor_environment(self, _):
        if self._load_environment(wait=False):
            rospy.loginfo("Inspection scene updated from environment parameters.")

    def _load_environment(self, wait):  # wait arg retained for API compatibility
        spec = get_part_spec()
        mesh_path = spec["mesh_path"]
        pose = spec.get("pose")
        position = pose.get("position")
        orientation = pose.get("orientation")
        scale = spec.get("scale")

        signature = (mesh_path, tuple(position), tuple(orientation), tuple(scale))
        if signature == self._env_signature:
            return False

        obj_path = convert_stl_to_obj(mesh_path, self.temp_dir)
        quat = quaternion_from_euler(*orientation)
        xml_string = self._build_scene_xml(obj_path, position, quat, scale)
        self._reload_simulator(xml_string)
        self._env_signature = signature
        return True

    def _build_scene_xml(self, mesh_file, position, quat, scale):
        fovy_rad = 2.0 * np.arctan(self.height / (2.0 * self.fy))
        fovy_deg = np.degrees(fovy_rad)

        with open(self.xml_template_path, "r") as f:
            template = f.read()

        return template.format(
            width=self.width,
            height=self.height,
            fovy=fovy_deg,
            mesh_file=mesh_file,
            scale_x=scale[0],
            scale_y=scale[1],
            scale_z=scale[2],
            pos_x=position[0],
            pos_y=position[1],
            pos_z=position[2],
            quat_w=quat[3],
            quat_x=quat[0],
            quat_y=quat[1],
            quat_z=quat[2],
        )

    def _reload_simulator(self, xml_string):
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "inspection_sensor"
        )

        sensor_head_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "sensor_head"
        )
        self.mocap_id = self.model.body_mocapid[sensor_head_body_id]
        assert self.mocap_id >= 0, "sensor_head is not a mocap body!"

        self.renderer = mujoco.Renderer(
            self.model, height=self.height, width=self.width
        )

        self.scene = mujoco.MjvScene(self.model, maxgeom=2000)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = self.camera_id
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt)

        try:
            self.context = self.renderer._context
        except AttributeError:
            self.context = mujoco.MjrContext(
                self.model, mujoco.mjtFontScale.mjFONTSCALE_100
            )

        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        self.near = float(self.model.vis.map.znear)
        self.far = float(self.model.vis.map.zfar)

    def get_camera_transform(self):
        """Get camera transform from TF (latest available)."""
        if self._manual_camera_transform is not None:
            return self._manual_camera_transform
        try:
            self.tf_listener.waitForTransform(
                "world", self.camera_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            (trans, rot) = self.tf_listener.lookupTransform(
                "world", self.camera_frame, rospy.Time(0)
            )
            T = quaternion_matrix(rot)
            T[:3, 3] = trans
            rospy.loginfo(
                f"Obtained camera transform {(trans, rot)} from TF for reference frame: {self.camera_frame}"
            )
            return T
        except Exception:
            return None

    def _update_camera_pose(self):
        T = self.get_camera_transform()
        if T is None:
            return False

        flip_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        position = T[:3, 3]
        rotation_camera_to_world = T[:3, :3]
        rotation_camera_to_world_mujoco = rotation_camera_to_world @ flip_rotation

        T_camera_in_world = np.eye(4)
        T_camera_in_world[:3, :3] = rotation_camera_to_world_mujoco
        T_camera_in_world[:3, 3] = position
        self._camera_transform_mujoco = T_camera_in_world

        quat_ros = quaternion_from_matrix(T_camera_in_world)
        quaternion_mujoco = np.array(
            [quat_ros[3], quat_ros[0], quat_ros[1], quat_ros[2]]
        )

        self.data.mocap_pos[self.mocap_id] = position
        self.data.mocap_quat[self.mocap_id] = quaternion_mujoco
        mujoco.mj_forward(self.model, self.data)
        return True

    def render_rgb_depth(self):
        if not self._update_camera_pose():
            rospy.logwarn_throttle(
                5.0,
                f"Camera transform not available (world -> {self.camera_frame}), skipping.",
            )
            return None, None

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        mujoco.mjr_render(self.viewport, self.scene, self.context)

        rgb_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth_buffer = np.zeros((self.height, self.width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_buffer, depth_buffer, self.viewport, self.context)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)

        rgb = np.flipud(rgb_buffer)
        depth_buffer = np.flipud(depth_buffer)
        depth_buffer = np.clip(depth_buffer, 0.0, 0.999999)

        extent = self.model.stat.extent
        depth = (self.near * self.far * extent) / (
            self.far - depth_buffer * (self.far - self.near)
        )

        return rgb, depth

    def _unproject_depth(self, depth, rgb, stride=1):
        """Unproject depth image to 3D points in camera frame."""
        stride = max(int(stride), 1)
        u_coords = np.arange(0, self.width, stride)
        v_coords = np.arange(0, self.height, stride)
        u, v = np.meshgrid(u_coords, v_coords)
        u = u.reshape(-1)
        v = v.reshape(-1)
        z = depth[::stride, ::stride].reshape(-1)

        mask = (z > self.min_depth) & (z < self.max_depth) & (z > 0)
        if not np.any(mask):
            return None, None

        u = u[mask]
        v = v[mask]
        z = z[mask]

        cx_sim = self.width * 0.5
        cy_sim = self.height * 0.5
        x_ros = (u - cx_sim) * z / self.fx
        y_ros = (v - cy_sim) * z / self.fy
        points_ros = np.column_stack((x_ros, y_ros, z))
        colors = rgb[::stride, ::stride].reshape(-1, 3)[mask] / 255.0

        return points_ros, colors

    def _transform_to_world(self, points_cam, colors):
        """Transform points from MuJoCo camera frame to world frame."""
        if self._camera_transform_mujoco is None or len(points_cam) == 0:
            return None, None

        F = np.diag([1.0, -1.0, -1.0])
        points_mj = (F @ points_cam.T).T
        points_h = np.column_stack((points_mj, np.ones(len(points_mj))))
        points_world_h = (self._camera_transform_mujoco @ points_h.T).T
        points_world = points_world_h[:, :3]

        return points_world, colors

    def _filter_floor(self, points_world, colors):
        """Filter out points below floor threshold."""
        floor_threshold = self.floor_height + self.floor_margin
        above_floor = points_world[:, 2] > floor_threshold
        return points_world[above_floor], colors[above_floor]

    def _physical_constraints_filter(self, points_world, colors):
        """Filter points based on physical constraints in ROS camera frame (+Z forward)."""
        if len(points_world) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        T_ros = self.get_camera_transform()
        if T_ros is None:
            raise RuntimeError(
                "Camera transform not available while filtering points based on physical constraints."
            )

        camera_pos = T_ros[:3, 3]
        camera_R = T_ros[:3, :3]
        points_cam = (camera_R.T @ (points_world - camera_pos).T).T

        depth = points_cam[:, 2]
        depth_mask = (depth >= self.min_depth) & (depth <= self.max_depth)

        xy_dist = np.sqrt(points_cam[:, 0] ** 2 + points_cam[:, 1] ** 2)
        radius_mask = xy_dist <= self.max_radius

        z_axis = np.array([0, 0, 1])
        point_vectors = points_cam / (
            np.linalg.norm(points_cam, axis=1, keepdims=True) + 1e-10
        )
        cos_angles = np.dot(point_vectors, z_axis)
        angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))
        orientation_mask = angles <= self.max_orientation

        mask = depth_mask & radius_mask & orientation_mask
        return points_world[mask], colors[mask]

    def _create_pointcloud_message(self, points, colors):
        """Create ROS PointCloud2 message from points and colors."""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        if len(points) == 0:
            return pc2.create_cloud(header, self.POINTCLOUD_FIELDS, [])
        cloud_data = np.column_stack((points, colors)).astype(np.float32)
        return pc2.create_cloud(header, self.POINTCLOUD_FIELDS, cloud_data)

    def trigger(self, publish=False, downsample=1):
        """Trigger perception: generate pointcloud and optionally publish to ROS.

        Args:
            publish: If True, publish pointcloud to ROS topic. If False, only return it.
            downsample: Integer stride for depth/RGB unprojection (>=1).

        Returns:
            PointCloud2 message (default) or tuple (points, colors) when return_raw=True.
        """

        rgb, depth = self.render_rgb_depth()
        if rgb is None or depth is None:
            return None

        points_ros, colors = self._unproject_depth(depth, rgb, stride=downsample)
        if points_ros is None:
            cloud_msg = self._create_pointcloud_message([], [])
            if publish:
                self.pointcloud_pub.publish(cloud_msg)
            return cloud_msg

        points_world, colors = self._transform_to_world(points_ros, colors)
        if points_world is None:
            cloud_msg = self._create_pointcloud_message([], [])
            if publish:
                self.pointcloud_pub.publish(cloud_msg)
            return cloud_msg

        points_world, colors = self._filter_floor(points_world, colors)
        if len(points_world) == 0:
            cloud_msg = self._create_pointcloud_message([], [])
            if publish:
                self.pointcloud_pub.publish(cloud_msg)
            return cloud_msg

        filtered_points, filtered_colors = self._physical_constraints_filter(
            points_world, colors
        )

        cloud_msg = self._create_pointcloud_message(filtered_points, filtered_colors)
        if publish:
            self.pointcloud_pub.publish(cloud_msg)
            rospy.loginfo(
                f"Published pointcloud with {len(filtered_points)} points for reference frame: {self.camera_frame}"
            )
        return cloud_msg

    def set_manual_camera_pose(self, position, orientation_xyzw):
        """Set camera pose manually (bypass TF)."""
        T = quaternion_matrix(orientation_xyzw)
        T[:3, 3] = position
        self._manual_camera_transform = T

    def clear_manual_camera_pose(self):
        """Revert to TF-based camera pose updates."""
        self._manual_camera_transform = None

    def cleanup(self):
        try:
            self.monitor_timer.shutdown()
        except Exception:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def run_node():
    rospy.init_node("simulated_perception", disable_signals=False)
    node = SimulatedPerception()
    rate = rospy.Rate(node.frame_rate)
    try:
        while not rospy.is_shutdown():
            node.trigger(publish=True)
            rate.sleep()
    finally:
        node.cleanup()


if __name__ == "__main__":
    try:
        run_node()
    except rospy.ROSInterruptException:
        pass
