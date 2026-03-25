#!/usr/bin/env python3
"""
MuJoCo RGB-D camera sensor node (headless).
Publishes RGB, depth, and pointcloud topics to ROS.
"""

import mujoco
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from resource_retriever import get_filename
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 1024
FOV = 60.0  # degrees

# Intrinsic parameters
FX = FY = CAMERA_WIDTH / (2.0 * np.tan(np.radians(FOV / 2.0)))
CX = CAMERA_WIDTH / 2.0
CY = CAMERA_HEIGHT / 2.0

DEFAULT_SCENE_URI = "package://neural_engine/config/mujoco_camera.xml"


class MuJoCoCameraNode:
    """MuJoCo simulation node that publishes ROS sensor messages."""

    def __init__(self):
        # Initialize ROS node
        rospy.init_node("mujoco_camera_node", anonymous=True)

        # Create publishers
        self.rgb_pub = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/camera/depth/image_raw", Image, queue_size=1)
        self.pointcloud_pub = rospy.Publisher(
            "/camera/points", PointCloud2, queue_size=1
        )
        self.camera_info_pub = rospy.Publisher(
            "/camera/rgb/camera_info", CameraInfo, queue_size=1
        )
        self.depth_info_pub = rospy.Publisher(
            "/camera/depth/camera_info", CameraInfo, queue_size=1
        )

        # No cv_bridge needed - we'll create messages directly

        # Load MuJoCo model from XML file
        scene_uri = rospy.get_param("~scene_xml", DEFAULT_SCENE_URI)
        rospy.loginfo(f"Loading MuJoCo scene: {scene_uri}")
        xml_path = get_filename(scene_uri, use_protocol=False)
        with open(xml_path, "r") as f:
            xml_string = f.read()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        # Get camera ID
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "inspection_sensor"
        )
        self.camera_pose_pub = rospy.Publisher(
            "/mujoco/camera_pose", PoseStamped, queue_size=1, latch=True
        )

        # Camera frame name
        self.camera_frame = "camera_optical_frame"

        # Create renderer (reused for efficiency)
        rospy.loginfo("Creating renderer...")
        self.renderer = mujoco.Renderer(
            self.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH
        )
        rospy.loginfo("Renderer created")

        # Warm up renderer by doing an initial render (this initializes OpenGL context)
        rospy.loginfo("Warming up renderer...")
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        try:
            _ = self.renderer.render()
            rospy.loginfo("Renderer ready")
        except Exception as e:
            rospy.logwarn(f"Renderer warmup had issue (may be OK): {e}")

        # Set up offscreen rendering resources for depth capture
        self.scene = mujoco.MjvScene(self.model, maxgeom=2000)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = self.camera_id
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_100
        )
        self.viewport = mujoco.MjrRect(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
        # Use visualization near/far settings
        self.near = float(
            self.model.vis.map.znear if self.model.vis.map.znear > 0 else 0.01
        )
        self.far = float(
            self.model.vis.map.zfar if self.model.vis.map.zfar > 0 else 50.0
        )

        self._cam_pos_view = self.model.cam_pos.reshape(self.model.ncam, 3)
        self._cam_quat_view = self.model.cam_quat.reshape(self.model.ncam, 4)
        cam_pos_start = self._cam_pos_view[self.camera_id].copy()
        self.camera_orientation = self._cam_quat_view[self.camera_id].copy()
        bounds_param = rospy.get_param(
            "~camera_bounds",
            {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [2.6, 3.2]},
        )
        self.camera_bounds = {
            axis: np.array(bounds_param.get(axis, default))
            for axis, default in zip(
                ["x", "y", "z"], [[-0.5, 0.5], [-0.5, 0.5], [2.6, 3.2]]
            )
        }
        self.camera_random_period = rospy.Duration(
            rospy.get_param("~camera_random_period", 0.01)
        )
        self.next_camera_update = rospy.Time.now()
        self._apply_camera_pose(cam_pos_start, publish=True)

        # Publishing rate
        self.rate = rospy.Rate(30)  # 30 Hz

        rospy.loginfo("MuJoCo camera node initialized")
        rospy.loginfo(f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        rospy.loginfo(
            f"Camera intrinsics: fx={FX:.2f}, fy={FY:.2f}, cx={CX:.2f}, cy={CY:.2f}"
        )

    def get_camera_info(self):
        """Create CameraInfo message with intrinsic parameters."""
        info = CameraInfo()
        info.header.frame_id = self.camera_frame
        info.width = CAMERA_WIDTH
        info.height = CAMERA_HEIGHT
        info.distortion_model = "plumb_bob"

        # Intrinsic matrix (3x3)
        info.K = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]

        # Distortion coefficients (assuming minimal distortion for industrial camera)
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Rectification matrix (identity)
        info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Projection matrix (3x4)
        info.P = [FX, 0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]

        return info

    def numpy_to_image_msg(self, img_array, encoding="rgb8"):
        """
        Convert numpy array to ROS Image message.

        Args:
            img_array: numpy array of shape (H, W) or (H, W, C)
            encoding: ROS image encoding (e.g., "rgb8", "16UC1", "mono8")

        Returns:
            sensor_msgs.msg.Image
        """
        img_msg = Image()
        img_msg.height = img_array.shape[0]
        img_msg.width = img_array.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = False

        # Set step (bytes per row)
        if encoding == "rgb8":
            img_msg.step = img_array.shape[1] * 3
            # Convert RGB to bytes (H, W, 3) -> flat array
            img_msg.data = img_array.astype(np.uint8).tobytes()
        elif encoding == "16UC1":
            img_msg.step = img_array.shape[1] * 2  # uint16 = 2 bytes
            # Ensure uint16 and little-endian
            img_array_uint16 = img_array.astype(np.uint16)
            img_msg.data = img_array_uint16.tobytes()
        elif encoding == "mono8":
            img_msg.step = img_array.shape[1]
            img_msg.data = img_array.astype(np.uint8).tobytes()
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")

        return img_msg

    def _sample_camera_position(self):
        return np.array(
            [
                np.random.uniform(*self.camera_bounds["x"]),
                np.random.uniform(*self.camera_bounds["y"]),
                np.random.uniform(*self.camera_bounds["z"]),
            ]
        )

    def _apply_camera_pose(self, position, publish=False):
        self._cam_pos_view[self.camera_id] = position
        mujoco.mj_forward(self.model, self.data)
        if publish:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position.x = position[0]
            pose_msg.pose.position.y = position[1]
            pose_msg.pose.position.z = position[2]
            pose_msg.pose.orientation.x = self.camera_orientation[0]
            pose_msg.pose.orientation.y = self.camera_orientation[1]
            pose_msg.pose.orientation.z = self.camera_orientation[2]
            pose_msg.pose.orientation.w = self.camera_orientation[3]
            self.camera_pose_pub.publish(pose_msg)

    def _maybe_update_camera_pose(self):
        if rospy.Time.now() >= self.next_camera_update:
            position = self._sample_camera_position()
            self._apply_camera_pose(position, publish=True)
            self.next_camera_update = rospy.Time.now() + self.camera_random_period

    def depth_to_pointcloud(self, depth_image, rgb_image=None):
        """Convert depth image to point cloud using vectorized operations."""
        height, width = depth_image.shape

        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Filter valid depth values
        valid_mask = (depth_image > 0) & (depth_image < 10.0)

        if not np.any(valid_mask):
            return np.array([]).reshape(0, 3 if rgb_image is None else 6)

        # Get valid pixels
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth_image[valid_mask]

        # Convert to 3D points using camera intrinsics
        x_valid = (u_valid - CX) * z_valid / FX
        y_valid = (v_valid - CY) * z_valid / FY

        if rgb_image is not None:
            # Get RGB values for valid pixels
            r_valid = rgb_image[valid_mask, 0] / 255.0
            g_valid = rgb_image[valid_mask, 1] / 255.0
            b_valid = rgb_image[valid_mask, 2] / 255.0
            points = np.column_stack(
                [x_valid, y_valid, z_valid, r_valid, g_valid, b_valid]
            )
        else:
            points = np.column_stack([x_valid, y_valid, z_valid])

        return points

    def publish_images(self):
        """Capture and publish RGB and depth images."""
        # Randomize camera pose periodically
        self._maybe_update_camera_pose()

        # Forward simulation step
        mujoco.mj_forward(self.model, self.data)

        # Render RGB using high-level renderer
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb = self.renderer.render()

        # Render depth using low-level OpenGL buffers
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

        depth_buffer = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.float32)
        mujoco.mjr_readPixels(None, depth_buffer, self.viewport, self.context)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)

        # MuJoCo depth buffer has origin at bottom-left; flip vertically to match image
        depth_buffer = np.flipud(depth_buffer)

        # Convert normalized depth buffer to metric depth (meters along camera view)
        # Depth buffer values outside [0, 1] are invalid
        depth_buffer = np.clip(depth_buffer, 0.0, 0.999999)
        depth_z = (self.near * self.far) / (
            self.far - (self.far - self.near) * depth_buffer
        )

        # Create header
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.camera_frame

        # Publish RGB image
        try:
            rgb_msg = self.numpy_to_image_msg(rgb, encoding="rgb8")
            rgb_msg.header = header
            self.rgb_pub.publish(rgb_msg)
            # Log first frame for debugging
            if not hasattr(self, "_first_frame_logged"):
                rospy.loginfo(
                    f"RGB image published: shape={rgb.shape}, dtype={rgb.dtype}, min={rgb.min()}, max={rgb.max()}"
                )
                self._first_frame_logged = True
        except Exception as e:
            rospy.logerr(f"Error publishing RGB: {e}")

        # Publish depth image (16UC1 format in mm)
        try:
            # Convert to uint16 in millimeters
            depth_mm = (depth_z * 1000).astype(np.uint16)
            depth_msg = self.numpy_to_image_msg(depth_mm, encoding="16UC1")
            depth_msg.header = header
            self.depth_pub.publish(depth_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing depth: {e}")

        # Publish camera info
        info = self.get_camera_info()
        info.header = header
        self.camera_info_pub.publish(info)
        self.depth_info_pub.publish(info)

        # Generate and publish point cloud
        try:
            points = self.depth_to_pointcloud(depth_z, rgb)
            if len(points) > 0:
                # Create point cloud message
                header_pc = Header()
                header_pc.stamp = header.stamp
                header_pc.frame_id = self.camera_frame

                # Define fields
                fields = [
                    PointField("x", 0, PointField.FLOAT32, 1),
                    PointField("y", 4, PointField.FLOAT32, 1),
                    PointField("z", 8, PointField.FLOAT32, 1),
                ]

                if points.shape[1] == 6:  # Has RGB
                    fields.extend(
                        [
                            PointField("r", 12, PointField.FLOAT32, 1),
                            PointField("g", 16, PointField.FLOAT32, 1),
                            PointField("b", 20, PointField.FLOAT32, 1),
                        ]
                    )

                # Create point cloud
                pc_msg = pc2.create_cloud(header_pc, fields, points)
                self.pointcloud_pub.publish(pc_msg)

                # Log first point cloud for debugging
                if not hasattr(self, "_first_pc_logged"):
                    rospy.loginfo(
                        f"Point cloud published: {len(points)} points, shape={points.shape}"
                    )
                    rospy.loginfo(
                        f"  Point cloud range: x=[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
                        f"y=[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
                        f"z=[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]"
                    )
                    self._first_pc_logged = True
        except Exception as e:
            rospy.logerr(f"Error publishing point cloud: {e}")

    def run(self):
        """Main simulation loop (headless)."""
        rospy.loginfo("Starting MuJoCo camera sensor node...")
        frame_count = 0

        while not rospy.is_shutdown():
            mujoco.mj_step(self.model, self.data)
            self.publish_images()

            frame_count += 1
            if frame_count % 30 == 0:
                rospy.loginfo(f"Published {frame_count} frames")

            self.rate.sleep()

        rospy.loginfo("Shutting down MuJoCo camera node")


def main():
    try:
        node = MuJoCoCameraNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
