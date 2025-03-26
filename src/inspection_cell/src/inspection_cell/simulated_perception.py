#!/usr/bin/env python3

import rospy
import yaml
import numpy as np
import os
import rospkg
import tf2_ros
import tf.transformations
from geometry_msgs.msg import (
    TransformStamped,
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    Vector3,
)
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, ColorRGBA
import cv2
from cv_bridge import CvBridge
import random
import math


class SimulatedPerception:
    """
    Simulated perception system that detects objects within a camera's field of view and a perception ROI.

    This class simulates a camera mounted on the robot's tool frame, and detects objects
    within both the camera's field of view and a perception ROI defined in the environment.
    """

    def __init__(self):
        """Initialize the simulated perception system."""
        rospy.loginfo("Initializing SimulatedPerception...")

        # Initialize ROS node if it hasn't been initialized already
        if not rospy.core.is_initialized():
            rospy.init_node("simulated_perception", anonymous=True)

        # Create RosPack instance to get paths
        self.rospack = rospkg.RosPack()

        # Load camera configuration
        self._load_camera_config()

        # Initialize TF listener for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Create publishers for visualization and simulated sensor data
        self._create_publishers()

        # Initialize OpenCV bridge for image conversions
        self.cv_bridge = CvBridge()

        # Get planning scene interface to access objects
        self._get_planning_scene_interface()

        # Initialize the perception ROI information
        self._get_perception_roi_info()

        # Timer for continuous perception updates
        self.perception_rate = rospy.Rate(self.camera_config["sensing"]["frame_rate"])
        self.perception_timer = rospy.Timer(
            rospy.Duration(1.0 / self.camera_config["sensing"]["frame_rate"]),
            self._perception_callback,
        )

        rospy.loginfo("SimulatedPerception initialized")

    def _load_camera_config(self):
        """Load camera configuration from YAML file."""
        config_path = os.path.join(
            self.rospack.get_path("inspection_cell"), "config", "perception_camera.yaml"
        )
        rospy.loginfo(f"Loading camera configuration from: {config_path}")

        try:
            with open(config_path, "r") as f:
                self.camera_config = yaml.safe_load(f)["camera"]
            rospy.loginfo("Camera configuration loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load camera configuration: {str(e)}")
            # Set default configuration
            self.camera_config = {
                "frame": {
                    "reference_frame": "tool0",
                    "transform": {
                        "position": [0.0, 0.0, 0.05],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                },
                "field_of_view": {
                    "type": "frustum",
                    "horizontal_fov": 60.0,
                    "vertical_fov": 45.0,
                    "min_range": 0.1,
                    "max_range": 2.0,
                },
                "sensing": {
                    "publish_rgb": True,
                    "publish_depth": True,
                    "publish_point_cloud": True,
                    "noise_level": 0.005,
                    "frame_rate": 30,
                },
                "visualization": {
                    "show_camera_frame": True,
                    "show_frustum": True,
                    "frustum_color": [0.0, 0.7, 1.0, 0.3],
                },
            }
            rospy.logwarn("Using default camera configuration")

    def _create_publishers(self):
        """Create publishers for visualization and simulated sensor data."""
        # Publishers for simulated sensor data
        if self.camera_config["sensing"]["publish_rgb"]:
            self.rgb_pub = rospy.Publisher(
                "/camera/color/image_raw", Image, queue_size=1
            )

        if self.camera_config["sensing"]["publish_depth"]:
            self.depth_pub = rospy.Publisher(
                "/camera/depth/image_raw", Image, queue_size=1
            )

        if self.camera_config["sensing"]["publish_point_cloud"]:
            self.pointcloud_pub = rospy.Publisher(
                "/camera/depth/points", PointCloud2, queue_size=1
            )

        # Camera info publisher
        self.camera_info_pub = rospy.Publisher(
            "/camera/camera_info", CameraInfo, queue_size=1
        )

        # Visualization publishers
        self.camera_frame_pub = rospy.Publisher(
            "/simulated_perception/camera_frame", MarkerArray, queue_size=1
        )
        self.frustum_pub = rospy.Publisher(
            "/simulated_perception/frustum", Marker, queue_size=1
        )
        self.detected_objects_pub = rospy.Publisher(
            "/simulated_perception/detected_objects", MarkerArray, queue_size=1
        )

    def _get_planning_scene_interface(self):
        """Initialize planning scene interface to access objects."""
        try:
            from moveit_commander import PlanningSceneInterface

            self.scene = PlanningSceneInterface()
            rospy.loginfo("Connected to MoveIt planning scene")
        except Exception as e:
            rospy.logerr(f"Failed to connect to MoveIt planning scene: {str(e)}")
            self.scene = None

    def _get_perception_roi_info(self):
        """Get information about the perception ROI from the planning scene."""
        if not self.scene:
            rospy.logerr(
                "Planning scene interface not available, cannot get perception ROI info"
            )
            return

        # Wait for the scene to be fully loaded
        rospy.sleep(1.0)

        # Get all collision objects from the scene
        scene_objects = self.scene.get_objects()

        # Check if perception_roi exists in the scene
        if "perception_roi" not in scene_objects:
            rospy.logerr("perception_roi not found in planning scene objects")
            self.perception_roi = None
            return

        # Extract perception ROI information
        roi_object = scene_objects["perception_roi"]
        self.perception_roi = roi_object
        rospy.loginfo("Retrieved perception ROI from planning scene")

    def _perception_callback(self, event=None):
        """Periodic callback for perception updates."""
        # Update camera pose
        if not self._update_camera_pose():
            return

        # Visualize camera and its field of view
        self._visualize_camera()

        # Get objects in the scene
        detected_objects = self._detect_objects()

        # Visualize detected objects
        self._visualize_detected_objects(detected_objects)

        # Generate and publish simulated sensor data
        self._publish_simulated_data(detected_objects)

    def _update_camera_pose(self):
        """Update the current camera pose based on the robot's position."""
        try:
            # Get the transform from base_link to the reference frame (e.g., tool0)
            ref_frame = self.camera_config["frame"]["reference_frame"]
            transform = self.tf_buffer.lookup_transform(
                "base_link", ref_frame, rospy.Time(0), rospy.Duration(1.0)
            )

            # Apply the camera offset from the reference frame
            camera_offset = self.camera_config["frame"]["transform"]["position"]
            camera_orientation = self.camera_config["frame"]["transform"]["orientation"]

            # Store camera pose
            self.camera_pose = transform

            # Store camera offset and orientation for later use
            self.camera_offset = camera_offset
            self.camera_orientation = camera_orientation

            return True
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup failed: {str(e)}")
            return False

    def _visualize_camera(self):
        """Visualize the camera frame and its field of view."""
        if self.camera_config["visualization"]["show_camera_frame"]:
            self._visualize_camera_frame()

        if self.camera_config["visualization"]["show_frustum"]:
            self._visualize_camera_frustum()

    def _visualize_camera_frame(self):
        """Visualize the camera coordinate frame."""
        marker_array = MarkerArray()

        # Create markers for the coordinate axes
        for i, (axis, color) in enumerate(
            zip(["x", "y", "z"], [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
        ):
            # Create axis marker
            marker = Marker()
            marker.header.frame_id = self.camera_config["frame"]["reference_frame"]
            marker.header.stamp = rospy.Time.now()
            marker.ns = "camera_frame"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # Set marker scale (arrow dimensions)
            marker.scale.x = 0.05  # Shaft diameter
            marker.scale.y = 0.08  # Head diameter
            marker.scale.z = 0.05  # Head length

            # Set marker color
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            # Set marker position (origin of the camera frame)
            marker.pose.position.x = self.camera_offset[0]
            marker.pose.position.y = self.camera_offset[1]
            marker.pose.position.z = self.camera_offset[2]

            # Set marker orientation
            if axis == "x":
                # X-axis: point along x
                rpy = [0, 0, 0]
                if self.camera_orientation[0] != 0:
                    rpy[0] += self.camera_orientation[0]
            elif axis == "y":
                # Y-axis: point along y
                rpy = [0, 0, math.pi / 2]
                if self.camera_orientation[1] != 0:
                    rpy[1] += self.camera_orientation[1]
            else:  # z
                # Z-axis: point along z
                rpy = [0, math.pi / 2, 0]
                if self.camera_orientation[2] != 0:
                    rpy[2] += self.camera_orientation[2]

            q = tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            # Add to marker array
            marker_array.markers.append(marker)

        # Publish the marker array
        self.camera_frame_pub.publish(marker_array)

    def _visualize_camera_frustum(self):
        """Visualize the camera field of view as a frustum."""
        # Create a marker for the camera frustum
        marker = Marker()
        marker.header.frame_id = self.camera_config["frame"]["reference_frame"]
        marker.header.stamp = rospy.Time.now()
        marker.ns = "camera_frustum"
        marker.id = 0

        # Use a pyramid mesh for the frustum
        if self.camera_config["field_of_view"]["type"] == "frustum":
            marker.type = Marker.LINE_LIST
        else:  # cone
            marker.type = Marker.LINE_LIST  # We'll still use LINE_LIST for both types

        marker.action = Marker.ADD

        # Set marker scale
        marker.scale.x = 0.01  # Line width

        # Set marker color from config
        color = self.camera_config["visualization"]["frustum_color"]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3] if len(color) > 3 else 0.5

        # Calculate frustum vertices
        h_fov = math.radians(self.camera_config["field_of_view"]["horizontal_fov"] / 2)
        v_fov = math.radians(self.camera_config["field_of_view"]["vertical_fov"] / 2)
        near = self.camera_config["field_of_view"]["min_range"]
        far = self.camera_config["field_of_view"]["max_range"]

        # Calculate near and far planes dimensions
        near_height = 2 * near * math.tan(v_fov)
        near_width = 2 * near * math.tan(h_fov)
        far_height = 2 * far * math.tan(v_fov)
        far_width = 2 * far * math.tan(h_fov)

        # Define near plane corners (x forward, y left, z up)
        near_top_left = Point(near, near_width / 2, near_height / 2)
        near_top_right = Point(near, -near_width / 2, near_height / 2)
        near_bottom_left = Point(near, near_width / 2, -near_height / 2)
        near_bottom_right = Point(near, -near_width / 2, -near_height / 2)

        # Define far plane corners
        far_top_left = Point(far, far_width / 2, far_height / 2)
        far_top_right = Point(far, -far_width / 2, far_height / 2)
        far_bottom_left = Point(far, far_width / 2, -far_height / 2)
        far_bottom_right = Point(far, -far_width / 2, -far_height / 2)

        # Define the camera position (origin of frustum)
        origin = Point(0, 0, 0)

        # Add lines for near plane
        points = []

        # Near plane
        points.extend([near_top_left, near_top_right])
        points.extend([near_top_right, near_bottom_right])
        points.extend([near_bottom_right, near_bottom_left])
        points.extend([near_bottom_left, near_top_left])

        # Far plane
        points.extend([far_top_left, far_top_right])
        points.extend([far_top_right, far_bottom_right])
        points.extend([far_bottom_right, far_bottom_left])
        points.extend([far_bottom_left, far_top_left])

        # Connecting lines
        points.extend([near_top_left, far_top_left])
        points.extend([near_top_right, far_top_right])
        points.extend([near_bottom_left, far_bottom_left])
        points.extend([near_bottom_right, far_bottom_right])

        # Add points to marker
        marker.points = points

        # Apply camera offset
        marker.pose.position.x = self.camera_offset[0]
        marker.pose.position.y = self.camera_offset[1]
        marker.pose.position.z = self.camera_offset[2]

        # Apply camera orientation
        if any(self.camera_orientation):
            q = tf.transformations.quaternion_from_euler(
                self.camera_orientation[0],
                self.camera_orientation[1],
                self.camera_orientation[2],
            )
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
        else:
            # Default orientation (camera looking forward)
            marker.pose.orientation.w = 1.0

        # Publish the frustum marker
        self.frustum_pub.publish(marker)

    def _detect_objects(self):
        """Detect objects that are within both the camera's field of view and the perception ROI."""
        if not self.scene or not self.perception_roi:
            return []

        # Get all collision objects from the scene
        scene_objects = self.scene.get_objects()

        # Filter objects that are within the perception ROI and camera FOV
        detected_objects = []

        for obj_name, obj in scene_objects.items():
            # Skip the ROI objects themselves
            if obj_name in ["perception_roi", "robot_roi"]:
                continue

            # Check if object is in perception ROI and camera FOV
            if self._is_object_in_roi(obj) and self._is_object_in_camera_fov(obj):
                detected_objects.append((obj_name, obj))

        return detected_objects

    def _is_object_in_roi(self, obj):
        """Check if an object is within the perception ROI."""
        if not self.perception_roi or not obj.primitives:
            return False

        # Get perception ROI dimensions and position
        roi_dims = list(self.perception_roi.primitives[0].dimensions)
        roi_pos = self.perception_roi.primitive_poses[0].position

        # Get object position and dimensions
        obj_pos = obj.primitive_poses[0].position

        # For simplicity, we'll use a bounding box check
        # This is a simplified approximation - for more accurate results,
        # proper collision checking would be needed

        # Calculate half-dimensions of the ROI
        roi_half_x = roi_dims[0] / 2
        roi_half_y = roi_dims[1] / 2
        roi_half_z = roi_dims[2] / 2

        # Check if object center is within the ROI box
        in_x_range = abs(obj_pos.x - roi_pos.x) <= roi_half_x
        in_y_range = abs(obj_pos.y - roi_pos.y) <= roi_half_y
        in_z_range = abs(obj_pos.z - roi_pos.z) <= roi_half_z

        return in_x_range and in_y_range and in_z_range

    def _is_object_in_camera_fov(self, obj):
        """Check if an object is within the camera's field of view."""
        if not hasattr(self, "camera_pose") or not obj.primitives:
            return False

        # Get camera position in world frame
        cam_pos_world = Point(
            self.camera_pose.transform.translation.x,
            self.camera_pose.transform.translation.y,
            self.camera_pose.transform.translation.z,
        )

        # Apply the camera offset in camera's local frame
        # (simplified - we should really apply a full transform here)

        # Get object position
        obj_pos = obj.primitive_poses[0].position

        # Calculate vector from camera to object
        dx = obj_pos.x - cam_pos_world.x
        dy = obj_pos.y - cam_pos_world.y
        dz = obj_pos.z - cam_pos_world.z

        # Calculate distance to object
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Check distance range
        min_range = self.camera_config["field_of_view"]["min_range"]
        max_range = self.camera_config["field_of_view"]["max_range"]
        if distance < min_range or distance > max_range:
            return False

        # For a proper implementation, we would transform the object position
        # into the camera frame and check if it's within the frustum angle
        # For simplicity, we'll use a rough approximation here

        # Convert horizontal and vertical FOV to radians
        h_fov_rad = math.radians(self.camera_config["field_of_view"]["horizontal_fov"])
        v_fov_rad = math.radians(self.camera_config["field_of_view"]["vertical_fov"])

        # Get camera orientation quaternion
        cam_quat = [
            self.camera_pose.transform.rotation.x,
            self.camera_pose.transform.rotation.y,
            self.camera_pose.transform.rotation.z,
            self.camera_pose.transform.rotation.w,
        ]

        # Convert camera quaternion to rotation matrix
        cam_rot_matrix = tf.transformations.quaternion_matrix(cam_quat)

        # Camera's forward vector in world frame
        # In camera local frame, forward is along X
        cam_forward = np.array(
            [cam_rot_matrix[0, 0], cam_rot_matrix[1, 0], cam_rot_matrix[2, 0]]
        )

        # Normalize the forward vector
        cam_forward = cam_forward / np.linalg.norm(cam_forward)

        # Vector from camera to object
        cam_to_obj = np.array([dx, dy, dz])

        # Normalize
        if np.linalg.norm(cam_to_obj) > 0:
            cam_to_obj = cam_to_obj / np.linalg.norm(cam_to_obj)
        else:
            return False  # Object is at the same position as camera

        # Compute the dot product (cosine of angle between vectors)
        cos_angle = np.dot(cam_forward, cam_to_obj)

        # Convert to angle
        angle = math.acos(min(max(cos_angle, -1.0), 1.0))

        # Simplified check - we're approximating the frustum as a cone
        # The actual check would compute the horizontal and vertical angles separately
        max_angle = max(h_fov_rad, v_fov_rad) / 2

        return angle <= max_angle

    def _visualize_detected_objects(self, detected_objects):
        """Visualize the detected objects."""
        marker_array = MarkerArray()

        # Create markers for each detected object
        for i, (obj_name, obj) in enumerate(detected_objects):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "detected_objects"
            marker.id = i
            marker.type = Marker.CUBE  # Can be adapted based on object type
            marker.action = Marker.ADD

            # Set marker pose to object pose
            marker.pose = obj.primitive_poses[0] if obj.primitive_poses else Pose()

            # Set marker scale based on object dimensions
            if obj.primitives and len(obj.primitives[0].dimensions) >= 3:
                marker.scale.x = obj.primitives[0].dimensions[0]
                marker.scale.y = obj.primitives[0].dimensions[1]
                marker.scale.z = obj.primitives[0].dimensions[2]
            else:
                marker.scale = Vector3(0.1, 0.1, 0.1)  # Default size

            # Set marker color (green for detected objects)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5

            marker_array.markers.append(marker)

        # Publish the marker array
        self.detected_objects_pub.publish(marker_array)

    def _publish_simulated_data(self, detected_objects):
        """Generate and publish simulated sensor data for the detected objects."""
        if not detected_objects:
            return

        # Generate and publish data based on config
        if self.camera_config["sensing"]["publish_rgb"]:
            self._publish_rgb_image(detected_objects)

        if self.camera_config["sensing"]["publish_depth"]:
            self._publish_depth_image(detected_objects)

        if self.camera_config["sensing"]["publish_point_cloud"]:
            self._publish_point_cloud(detected_objects)

        # Always publish camera info
        self._publish_camera_info()

    def _publish_rgb_image(self, detected_objects):
        """Publish a simulated RGB image."""
        # Create a blank image
        width, height = 640, 480  # Standard VGA resolution
        img = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray background

        # Add detected objects to the image (simplified)
        for obj_name, obj in detected_objects:
            # In a real implementation, project 3D objects onto 2D image
            # Here we just draw random colored rectangles for demonstration
            x1, y1 = random.randint(0, width // 2), random.randint(0, height // 2)
            x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
            x2, y2 = min(x2, width - 1), min(y2, height - 1)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.putText(
                img,
                obj_name,
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        # Convert to ROS image and publish
        img_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="rgb8")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = self.camera_config["frame"]["reference_frame"]
        self.rgb_pub.publish(img_msg)

    def _publish_depth_image(self, detected_objects):
        """Publish a simulated depth image."""
        # Create a blank depth image (far distance by default)
        width, height = 640, 480
        max_range = self.camera_config["field_of_view"]["max_range"]
        depth_img = np.ones((height, width), dtype=np.float32) * max_range

        # Add detected objects to the depth image (simplified)
        for obj_name, obj in detected_objects:
            # In a real implementation, project 3D objects onto 2D depth image
            # Here we just draw random depth regions for demonstration
            x1, y1 = random.randint(0, width // 2), random.randint(0, height // 2)
            x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
            x2, y2 = min(x2, width - 1), min(y2, height - 1)

            # Random depth value between min and max range
            min_range = self.camera_config["field_of_view"]["min_range"]
            depth_value = random.uniform(min_range, max_range * 0.8)

            # Add some noise
            noise_level = self.camera_config["sensing"]["noise_level"]
            noise = np.random.normal(0, noise_level, (y2 - y1, x2 - x1))

            # Set depth values with noise
            depth_img[y1:y2, x1:x2] = depth_value + noise

        # Convert to ROS image and publish
        img_msg = self.cv_bridge.cv2_to_imgmsg(depth_img, encoding="32FC1")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = self.camera_config["frame"]["reference_frame"]
        self.depth_pub.publish(img_msg)

    def _publish_point_cloud(self, detected_objects):
        """Publish a simulated point cloud for the detected objects."""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.camera_config["frame"]["reference_frame"]

        # Create point cloud data (simplified)
        points = []
        for obj_name, obj in detected_objects:
            # Generate random points for each object
            num_points = 100  # Number of points per object

            # Get object position and dimensions
            position = (
                obj.primitive_poses[0].position if obj.primitive_poses else Point()
            )
            dimensions = (
                list(obj.primitives[0].dimensions)
                if obj.primitives
                else [0.1, 0.1, 0.1]
            )

            # Generate random points within the object's bounds
            for _ in range(num_points):
                # Random point within object dimensions
                x_offset = random.uniform(-dimensions[0] / 2, dimensions[0] / 2)
                y_offset = random.uniform(-dimensions[1] / 2, dimensions[1] / 2)
                z_offset = random.uniform(-dimensions[2] / 2, dimensions[2] / 2)

                # Point coordinates relative to object center
                x = position.x + x_offset
                y = position.y + y_offset
                z = position.z + z_offset

                # Add some noise
                noise_level = self.camera_config["sensing"]["noise_level"]
                x += random.gauss(0, noise_level)
                y += random.gauss(0, noise_level)
                z += random.gauss(0, noise_level)

                # Random color (RGB)
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)

                # Pack RGB into a single integer
                rgb = (r << 16) | (g << 8) | b

                # Add point with coordinates and color
                points.append([x, y, z, rgb])

        # Create point cloud message
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        # Create and publish point cloud
        if points:
            pc_msg = pc2.create_cloud(header, fields, points)
            self.pointcloud_pub.publish(pc_msg)

    def _publish_camera_info(self):
        """Publish camera intrinsic parameters."""
        # Create camera info message with standard parameters
        cam_info = CameraInfo()
        cam_info.header.stamp = rospy.Time.now()
        cam_info.header.frame_id = self.camera_config["frame"]["reference_frame"]

        # Set image dimensions
        cam_info.width = 640
        cam_info.height = 480

        # Set camera matrix (K) - focal lengths and principal point
        fx = fy = 525.0  # Standard focal length for 640x480 images
        cx = 320.0  # Principal point x (usually width/2)
        cy = 240.0  # Principal point y (usually height/2)
        cam_info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

        # Set projection matrix (P)
        cam_info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]

        # Set rectification matrix (R) - identity for non-stereo cameras
        cam_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        # Set distortion parameters - assume no distortion
        cam_info.D = [0, 0, 0, 0, 0]

        # Set distortion model
        cam_info.distortion_model = "plumb_bob"

        # Publish the camera info
        self.camera_info_pub.publish(cam_info)

    def get_detected_objects(self):
        """Get the currently detected objects."""
        return self._detect_objects()

    def shutdown(self):
        """Shutdown the perception system."""
        if hasattr(self, "perception_timer"):
            self.perception_timer.shutdown()
        rospy.loginfo("SimulatedPerception shutdown")


if __name__ == "__main__":
    # Test the simulated perception system
    perception = SimulatedPerception()
    rospy.loginfo("SimulatedPerception is running. Press Ctrl+C to stop.")
    rospy.spin()
