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
from tf.transformations import quaternion_from_euler, quaternion_matrix, quaternion_from_matrix

# Internal

from core.utils import get_param, resolve_package_path, convert_stl_to_obj


class SimulatedPerception:
    def __init__(self):
        package_root = rospkg.RosPack().get_path("neural_engine")
        config_path = os.path.join(package_root, "config", "perception.yaml")
        with open(config_path, "r") as cfg:
            config = yaml.safe_load(cfg)
            rospy.loginfo("Config loaded successfully.")

        sensor_cfg = config["sensor"]
        intr = sensor_cfg["intrinsics"]

        self.camera_frame = sensor_cfg.get("frame", "inspection_camera")
        self.frame_rate = sensor_cfg["dynamics"]["frame_rate"]
        self.width = intr["width"]
        self.height = intr["height"]
        self.fx = intr["fx"]
        self.fy = intr["fy"]
        self.cx = intr["cx"]
        self.cy = intr["cy"]
        floor_cfg = sensor_cfg.get("floor", {})
        self.floor_height = floor_cfg.get("height", 0.0)
        self.floor_margin = floor_cfg.get("margin", 0.02)

        depth_cfg = sensor_cfg["constraints"]["depth"]
        self.min_depth = depth_cfg["min"]
        self.max_depth = depth_cfg["max"]
        self.max_radius = sensor_cfg["constraints"]["radius"]

        self.tf_listener = tf.TransformListener()

        self.temp_dir = tempfile.mkdtemp(prefix="inspection_sim_")
        self.model = None
        self.data = None
        self.camera_id = None
        self.mocap_id = None
        self._env_signature = None
        self._camera_transform_ros = None  # ROS camera transform (for point cloud transformation)
        self._camera_transform_mujoco = None  # MuJoCo camera transform (for camera pose)

        self.pointcloud_pub = rospy.Publisher("/camera/points", PointCloud2, queue_size=1)
        rospy.loginfo("Pointcloud publisher initialized.")

        self._load_environment(wait=True)
        self.monitor_timer = rospy.Timer(rospy.Duration(1.0), self._monitor_environment)

    def _monitor_environment(self, _):
        changed = self._load_environment(wait=False)
        if changed:
            rospy.loginfo("Inspection scene updated from environment parameters.")

    def _load_environment(self, wait):
        timeout = rospy.Time.now() + rospy.Duration(10.0) if wait else None
        spec = get_param("/environment/part", None)
        while spec is None and wait and rospy.Time.now() < timeout:
            rospy.sleep(0.5)
            spec = get_param("/environment/part", None)
        if spec is None:
            if wait:
                raise RuntimeError("Inspection part not available on parameter server.")
            return False

        mesh_path = resolve_package_path(spec["mesh_path"])
        pose = spec.get("pose")
        position = pose.get("position")
        orientation = pose.get("orientation")
        scale = spec.get("scale")

        signature = (mesh_path, position, orientation, scale)
        if signature == self._env_signature:
            return False

        rospy.loginfo(f"Environment part specified: {spec}")
        obj_path = convert_stl_to_obj(mesh_path, self.temp_dir)
        quat = quaternion_from_euler(*orientation)
        xml_string = self._build_scene_xml(obj_path, position, quat, scale)
        self._reload_simulator(xml_string)
        self._env_signature = signature
        return True

    def _build_scene_xml(self, mesh_file, position, quat, scale):
        # Compute fovy from fy to match intrinsics
        # fovy = 2 * arctan(H / (2 * fy)) in radians, then convert to degrees
        # MuJoCo XML expects fovy in DEGREES
        fovy_rad = 2.0 * np.arctan(self.height / (2.0 * self.fy))
        fovy_deg = np.degrees(fovy_rad)
        rospy.loginfo(f"Computed fovy from fy: {fovy_deg:.2f} deg = {fovy_rad:.6f} rad (from fy={self.fy:.2f}, height={self.height})")
        
        return f"""
<mujoco model="inspection_sensor_scene">
    <option timestep="0.01" gravity="0 0 -9.81"/>
    <visual>
        <global offwidth="{self.width}" offheight="{self.height}"/>
        <map znear="0.01" zfar="5.0"/>
    </visual>
    <asset>
        <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.3 0.3 0.3" rgb2="0.4 0.4 0.4"/>
        <material name="floor_mat" texture="floor_tex" texrepeat="5 5" texuniform="true" reflectance="0.2"/>
        <material name="fixture_mat" rgba="0.2 0.2 0.2 1"/>
        <material name="workpiece_mat" rgba="0.8 0.2 0.2 1"/>
        <mesh name="workpiece_mesh" file="{mesh_file}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
    </asset>
    <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" material="floor_mat"/>
        <body name="sensor_head" mocap="true">
            <!-- Temporarily disabled for self-occlusion testing -->
            <!-- <geom name="housing" type="box" size="0.1 0.15 0.06" material="fixture_mat"/> -->
            <!-- <geom name="lens" type="cylinder" size="0.04 0.02" pos="0 0 -0.06" rgba="0 0 0 1" quat="0.707 0.707 0 0"/> -->
            <camera name="inspection_sensor" pos="0 0 0" quat="1 0 0 0" fovy="{fovy_deg}" resolution="{self.width} {self.height}"/>
        </body>
        <body name="workpiece" pos="{position[0]} {position[1]} {position[2]}" quat="{quat[3]} {quat[0]} {quat[1]} {quat[2]}">
            <geom name="workpiece_geom" type="mesh" mesh="workpiece_mesh" material="workpiece_mat"/>
        </body>
        <light name="key_light" pos="0 0 2.5" dir="0 0 -1" diffuse="1 1 1"/>
    </worldbody>
</mujoco>
"""

    def _reload_simulator(self, xml_string):
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "inspection_sensor")
        
        # Get mocap body ID for sensor_head
        sensor_head_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sensor_head")
        self.mocap_id = self.model.body_mocapid[sensor_head_body_id]
        rospy.loginfo(f"sensor_head_body_id={sensor_head_body_id}, mocap_id={self.mocap_id}")
        assert self.mocap_id >= 0, "sensor_head is not a mocap body!"
        
        # Calibrate intrinsics first (before any rendering)
        self._calibrate_intrinsics()
        
        # Use Renderer class which handles OpenGL context setup better on macOS
        rospy.loginfo("Initializing Renderer (handles OpenGL context setup)...")
        try:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            rospy.loginfo("Renderer created successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to create Renderer: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            raise
        
        # Create shared context and scene for depth reading (use renderer's context if accessible)
        rospy.loginfo("Setting up depth reading context...")
        try:
            self.scene = mujoco.MjvScene(self.model, maxgeom=2000)
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.cam)
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = self.camera_id
            self.opt = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.opt)
            # Try to use renderer's internal context, fallback to creating our own
            try:
                self.context = self.renderer._context
                rospy.loginfo("Using renderer's internal context for depth.")
            except AttributeError:
                self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
                rospy.loginfo("Created separate context for depth.")
            self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)
            rospy.loginfo("Depth reading context ready.")
        except Exception as e:
            rospy.logerr(f"Failed to setup depth reading context: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            raise
        
        # Read znear/zfar from XML visual map (matches GL frustum)
        self.near = float(self.model.vis.map.znear)
        self.far = float(self.model.vis.map.zfar)
        rospy.loginfo(f"Depth frustum: znear={self.near}, zfar={self.far}")

    def _calibrate_intrinsics(self):
        """Verify MuJoCo camera intrinsics match YAML configuration."""
        # Get fovy from model (MuJoCo stores fovy in DEGREES in cam_fovy)
        fovy_deg = float(self.model.cam_fovy[self.camera_id])
        fovy_rad = np.radians(fovy_deg)
        
        rospy.loginfo(f"MuJoCo model fovy: {fovy_deg:.3f} deg = {fovy_rad:.6f} rad")
        
        # Compute MuJoCo's effective intrinsics from fovy
        # MuJoCo uses: fovy = 2 * arctan(H / (2 * fy))
        # So: fy = H / (2 * tan(fovy/2))
        half_fovy_rad = fovy_rad / 2.0
        tan_half_fovy = np.tan(half_fovy_rad)
        
        if tan_half_fovy <= 0 or not np.isfinite(tan_half_fovy) or abs(tan_half_fovy) < 1e-10:
            rospy.logwarn(f"Invalid fovy: deg={fovy_deg:.3f}, rad={fovy_rad:.6f}, tan_half={tan_half_fovy:.6f}")
            return

        fy_mujoco = self.height / (2.0 * tan_half_fovy)
        # MuJoCo enforces: fx = fy * (W/H) for aspect ratio
        fx_mujoco = fy_mujoco * (self.width / self.height)
        cx_mujoco = self.width * 0.5
        cy_mujoco = self.height * 0.5
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("Camera Intrinsics Calibration:")
        rospy.loginfo(f"  MuJoCo fovy: {fovy_deg:.2f} deg")
        rospy.loginfo(f"  YAML fx: {self.fx:.2f}  |  MuJoCo fx: {fx_mujoco:.2f}  |  Diff: {abs(self.fx - fx_mujoco):.2f}")
        rospy.loginfo(f"  YAML fy: {self.fy:.2f}  |  MuJoCo fy: {fy_mujoco:.2f}  |  Diff: {abs(self.fy - fy_mujoco):.2f}")
        rospy.loginfo(f"  YAML cx: {self.cx:.2f}  |  MuJoCo cx: {cx_mujoco:.2f}  |  Diff: {abs(self.cx - cx_mujoco):.2f}")
        rospy.loginfo(f"  YAML cy: {self.cy:.2f}  |  MuJoCo cy: {cy_mujoco:.2f}  |  Diff: {abs(self.cy - cy_mujoco):.2f}")
        
        # Check if they match (within tolerance)
        tolerance = 1.0  # pixels
        fx_match = abs(self.fx - fx_mujoco) < tolerance
        fy_match = abs(self.fy - fy_mujoco) < tolerance
        cx_match = abs(self.cx - cx_mujoco) < tolerance
        cy_match = abs(self.cy - cy_mujoco) < tolerance
        
        if fx_match and fy_match and cx_match and cy_match:
            rospy.loginfo("  ✓ Intrinsics match! (within tolerance)")
        else:
            rospy.logwarn("  ✗ Intrinsics mismatch detected!")
            if not fx_match:
                rospy.logwarn(f"    fx mismatch: {abs(self.fx - fx_mujoco):.2f} pixels")
            if not fy_match:
                rospy.logwarn(f"    fy mismatch: {abs(self.fy - fy_mujoco):.2f} pixels")
            if not cx_match:
                rospy.logwarn(f"    cx mismatch: {abs(self.cx - cx_mujoco):.2f} pixels (using centered cx in unprojection)")
            if not cy_match:
                rospy.logwarn(f"    cy mismatch: {abs(self.cy - cy_mujoco):.2f} pixels (using centered cy in unprojection)")
        rospy.loginfo("=" * 60)

    def get_camera_transform(self):
        """Get the camera transform from TF. Returns None if transform is not available."""
        try:
            self.tf_listener.waitForTransform(
                "world", self.camera_frame, rospy.Time(0), rospy.Duration(0.1)
            )
            (trans, rot) = self.tf_listener.lookupTransform(
                "world", self.camera_frame, rospy.Time(0)
            )
            T = quaternion_matrix(rot)
            T[:3, 3] = trans
            rospy.loginfo_throttle(10.0, f"Camera transform: {(trans, rot)}")
            return T
        except Exception:
            return None

    def _update_camera_pose_from_tf(self):
        """Update MuJoCo camera pose from TF transform. Returns True if successful, False otherwise."""
        T = self.get_camera_transform()
        if T is None:
            return False
        
        # Store ROS camera transform for point cloud transformation
        self._camera_transform_ros = T
        
        # TF lookupTransform("world", "camera_frame") returns transform FROM camera_frame TO world
        # T * p_camera = p_world
        # T[:3, 3] is camera position in world frame (correct)
        # T[:3, :3] is rotation from camera to world (camera orientation in world frame)
        
        # MuJoCo Camera Explanation:
        # - MuJoCo cameras are OpenGL rendering viewpoints, not physical sensor models
        # - They use OpenGL's coordinate convention: camera looks along -Z axis
        # - +X = right in image, +Y = up in image, -Z = viewing direction (into scene)
        # - This is standard OpenGL/computer graphics convention (right-handed, Z points away from viewer)
        #
        # ROS Camera Frame Convention:
        # - ROS uses optical frame convention: +Z is forward (what camera sees)
        # - +X = right, +Y = down (image coordinates)
        # - This matches real camera conventions (Z points toward scene)
        #
        # Conversion: Rotate 180 degrees around X axis to flip Y and Z
        # This converts from ROS (+Z forward) to MuJoCo (-Z forward) while maintaining right-handedness
        flip_rotation = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        
        position = T[:3, 3]  # Camera position in world
        rotation_camera_to_world = T[:3, :3]  # Camera orientation in world frame (ROS convention)
        
        # Apply rotation to convert from ROS (+Z forward) to MuJoCo (-Z forward)
        rotation_camera_to_world_mujoco = rotation_camera_to_world @ flip_rotation
        
        # Build transform matrix for quaternion conversion (MuJoCo camera pose)
        T_camera_in_world = np.eye(4)
        T_camera_in_world[:3, :3] = rotation_camera_to_world_mujoco
        T_camera_in_world[:3, 3] = position
        
        # Store MuJoCo camera transform for reference
        self._camera_transform_mujoco = T_camera_in_world
        
        quat_ros = quaternion_from_matrix(T_camera_in_world)  # Returns (x, y, z, w) in ROS format
        
        # Convert to MuJoCo format (w, x, y, z)
        quaternion_mujoco = np.array([quat_ros[3], quat_ros[0], quat_ros[1], quat_ros[2]])
        
        rospy.loginfo_throttle(5.0, 
            f"TF quat (xyzw): [{quat_ros[0]:.4f}, {quat_ros[1]:.4f}, {quat_ros[2]:.4f}, {quat_ros[3]:.4f}], "
            f"MuJoCo quat (wxyz): [{quaternion_mujoco[0]:.4f}, {quaternion_mujoco[1]:.4f}, {quaternion_mujoco[2]:.4f}, {quaternion_mujoco[3]:.4f}]")
        
        # Update mocap body pose (world coordinates) - this is the correct way for runtime pose updates
        self.data.mocap_pos[self.mocap_id] = position
        self.data.mocap_quat[self.mocap_id] = quaternion_mujoco
        mujoco.mj_forward(self.model, self.data)
        
        # Verify MuJoCo is actually using the pose we set
        cam_pos_mj = self.data.cam_xpos[self.camera_id].copy()
        cam_R_mj = self.data.cam_xmat[self.camera_id].reshape(3, 3).copy()
        cam_fwd_mj = -cam_R_mj[:, 2]  # MuJoCo camera forward direction (-Z)
        
        # Expected forward from TF (MuJoCo convention)
        fwd_expected = rotation_camera_to_world_mujoco @ np.array([0, 0, -1])
        
        rospy.loginfo_throttle(2.0,
            f"[MuJoCo truth] cam_pos={cam_pos_mj}, cam_fwd={cam_fwd_mj}")
        rospy.loginfo_throttle(2.0,
            f"[TF] cam_pos={position}, fwd_expected={fwd_expected}")
        
        # Check if camera is facing the workpiece
        workpiece_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "workpiece")
        if workpiece_body_id >= 0:
            wp_pos = self.data.xpos[workpiece_body_id]
            to_wp = wp_pos - cam_pos_mj
            to_wp_norm = np.linalg.norm(to_wp)
            if to_wp_norm > 1e-6:
                to_wp = to_wp / to_wp_norm
                dot = cam_fwd_mj @ to_wp
                rospy.loginfo_throttle(2.0,
                    f"cam_fwd·to_workpiece={dot:.3f} (want > 0.5), "
                    f"distance={to_wp_norm:.3f}m")
        
        return True

    def cleanup(self):
        try:
            self.monitor_timer.shutdown()
        except Exception:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def render_rgb_depth(self):
        if not self._update_camera_pose_from_tf():
            rospy.logwarn_throttle(5.0, f"Camera transform not available (world -> {self.camera_frame}), skipping perception trigger.")
            return None, None
        
        # Update scene once (following mujoco_RGBD library approach)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        
        # Render to offscreen buffer
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        
        # Read BOTH RGB and depth in a single call (ensures same framebuffer)
        # Following mujoco_RGBD library: mjr_readPixels(color_buffer, depth_buffer, viewport, context)
        rgb_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth_buffer = np.zeros((self.height, self.width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_buffer, depth_buffer, self.viewport, self.context)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        
        # Flip to match ROS image convention (origin at top-left)
        rgb = np.flipud(rgb_buffer)
        depth_buffer = np.flipud(depth_buffer)
        depth_buffer = np.clip(depth_buffer, 0.0, 0.999999)
        
        # Convert normalized depth buffer to metric depth (meters)
        # Following mujoco_RGBD: depth = z_near * z_far * extent / (z_far - raw_depth * (z_far - z_near))
        extent = self.model.stat.extent
        depth = (self.near * self.far * extent) / (self.far - depth_buffer * (self.far - self.near))
        
        # Validation prints (as suggested)
        rospy.loginfo_throttle(2.0,
            f"cam_fovy(deg)={self.model.cam_fovy[self.camera_id]:.3f}, "
            f"vis znear,zfar={self.model.vis.map.znear:.3f},{self.model.vis.map.zfar:.3f} "
            f"extent={extent:.3f}, "
            f"depth(min,max)={np.min(depth):.3f},{np.max(depth):.3f}")
        
        # Raycast sanity check
        self._sanity_raycast()
        
        return rgb, depth
    
    def _sanity_raycast(self):
        """Raycast from camera to verify depth matches geometry."""
        try:
            cam_pos = self.data.cam_xpos[self.camera_id]
            cam_R = self.data.cam_xmat[self.camera_id].reshape(3, 3)
            cam_fwd = -cam_R[:, 2]  # MuJoCo camera forward direction (-Z)
            
            geomid = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(self.model, self.data, cam_pos, cam_fwd, None, 1, -1, geomid)
            
            rospy.loginfo_throttle(2.0,
                f"mj_ray hit geom {geomid[0]} at dist {dist:.3f} m")
        except (AttributeError, TypeError) as e:
            # mj_ray might not be available in all MuJoCo versions
            rospy.logdebug_throttle(5.0, f"mj_ray not available: {e}")

    def depth_to_pointcloud(self, depth, rgb):
        # Flatten pixel grid
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        u = u.reshape(-1)
        v = v.reshape(-1)
        z = depth.reshape(-1)

        # Basic validity + (optional) range limits
        mask = np.isfinite(z) & (z > self.min_depth) & (z < self.max_depth)
        if not np.any(mask):
            return np.array([], dtype=np.float32)

        u = u[mask]; v = v[mask]; z = z[mask]

        # Unproject using centered principal point (MuJoCo assumes centered cx/cy)
        cx_sim = self.width * 0.5
        cy_sim = self.height * 0.5
        x_ros = (u - cx_sim) * z / self.fx
        y_ros = (v - cy_sim) * z / self.fy
        z_ros = z
        points_ros = np.column_stack((x_ros, y_ros, z_ros))

        # Convert ROS optical -> MuJoCo camera basis
        # ROS = F * MJ  => MJ = F * ROS  (F = diag([1,-1,-1]))
        F = np.diag([1.0, -1.0, -1.0])
        points_mj = (F @ points_ros.T).T   # now +Y up, -Z forward

        colors = rgb.reshape(-1, 3)[mask] / 255.0

        if self._camera_transform_mujoco is None:
            rospy.logwarn_throttle(5.0, "MuJoCo camera transform not available.")
            return np.array([], dtype=np.float32)

        # Filter out invalid points before transformation
        valid_points = np.isfinite(points_mj).all(axis=1) & (np.abs(points_mj).max(axis=1) < 1e6)
        if not np.any(valid_points):
            return np.array([], dtype=np.float32)
        
        points_mj = points_mj[valid_points]
        colors = colors[valid_points]

        # Transform using MuJoCo camera->world (world_from_cam_mj)
        points_h = np.column_stack((points_mj, np.ones(len(points_mj))))
        points_world_h = (self._camera_transform_mujoco @ points_h.T).T
        points_world = points_world_h[:, :3]

        valid = np.isfinite(points_world).all(axis=1)
        points_world = points_world[valid]
        colors = colors[valid]

        if len(points_world) == 0:
            return np.array([], dtype=np.float32)

        return np.column_stack((points_world, colors)).astype(np.float32)

    def publish_pointcloud(self):
        if self.model is None:
            return
        rgb, depth = self.render_rgb_depth()
        if rgb is None or depth is None:
            return
        points = self.depth_to_pointcloud(depth, rgb)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("r", 12, PointField.FLOAT32, 1),
            PointField("g", 16, PointField.FLOAT32, 1),
            PointField("b", 20, PointField.FLOAT32, 1),
        ]
        if points.size == 0:
            cloud_msg = pc2.create_cloud(header, fields, [])
            rospy.logwarn_throttle(5.0, "MuJoCo sensor produced zero valid points.")
        else:
            cloud_msg = pc2.create_cloud(header, fields, points)
            rospy.loginfo_throttle(2.0, f"Published {points.shape[0]} points from MuJoCo sensor.")
            self.pointcloud_pub.publish(cloud_msg)


def run_node():
    print("Running simulated perception node...")
    rospy.init_node("simulated_perception", disable_signals=False)
    node = SimulatedPerception()
    rate = rospy.Rate(node.frame_rate)
    try:
        while not rospy.is_shutdown():
            node.publish_pointcloud()
            rate.sleep()
    finally:
        node.cleanup()


if __name__ == "__main__":
    try:
        run_node()
    except rospy.ROSInterruptException:
        pass
