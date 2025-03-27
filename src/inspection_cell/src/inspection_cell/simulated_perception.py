#!/usr/bin/env python3

import rospy
import yaml
import numpy as np
import os
import rospkg
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped, Point, Pose, Vector3, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, ColorRGBA
import cv2
from cv_bridge import CvBridge
import trimesh
import math


# Add this custom frustum creation function since trimesh.creation doesn't have frustum
def create_frustum(fovy, aspect, near, far):
    """
    Create a frustum mesh.

    Parameters:
    -----------
    fovy : float
        Vertical field of view in radians
    aspect : float
        Aspect ratio width/height
    near : float
        Near plane distance
    far : float
        Far plane distance

    Returns:
    --------
    frustum : trimesh.Trimesh
        A mesh representing the frustum
    """
    # Calculate dimensions at near and far planes
    near_height = 2 * near * math.tan(fovy / 2)
    near_width = near_height * aspect
    far_height = 2 * far * math.tan(fovy / 2)
    far_width = far_height * aspect

    # Define vertices
    vertices = np.array(
        [
            # Near plane (z = near)
            [-near_width / 2, -near_height / 2, near],  # 0: near bottom left
            [near_width / 2, -near_height / 2, near],  # 1: near bottom right
            [near_width / 2, near_height / 2, near],  # 2: near top right
            [-near_width / 2, near_height / 2, near],  # 3: near top left
            # Far plane (z = far)
            [-far_width / 2, -far_height / 2, far],  # 4: far bottom left
            [far_width / 2, -far_height / 2, far],  # 5: far bottom right
            [far_width / 2, far_height / 2, far],  # 6: far top right
            [-far_width / 2, far_height / 2, far],  # 7: far top left
        ]
    )

    # Define faces using triangles
    faces = np.array(
        [
            # Near plane
            [0, 1, 2],
            [0, 2, 3],
            # Far plane
            [4, 6, 5],
            [4, 7, 6],
            # Connect near to far, side planes
            [0, 3, 7],
            [0, 7, 4],  # Left
            [1, 5, 6],
            [1, 6, 2],  # Right
            [3, 2, 6],
            [3, 6, 7],  # Top
            [0, 4, 5],
            [0, 5, 1],  # Bottom
        ]
    )

    return trimesh.Trimesh(vertices=vertices, faces=faces)


class SimulatedPerception:
    """
    Camera RGB-D detection system focused on detecting a part object.

    This class creates a virtual camera attached to the robot's tool frame
    and publishes RGB-D images, point clouds, and camera info.
    """

    def __init__(self):
        """Initialize the RGB-D detection system."""
        rospy.loginfo("Initializing SimulatedPerception...")

        # Initialize ROS node if it hasn't been initialized already
        if not rospy.core.is_initialized():
            rospy.init_node("simulated_perception", anonymous=True)

        # Get RosPack for file paths
        self.rospack = rospkg.RosPack()

        # Load camera configuration
        self._load_camera_config()

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Create publishers for visualization and camera outputs
        self._create_publishers()

        # Initialize bridge for image conversion
        self.cv_bridge = CvBridge()

        # Initialize part mesh and visibility state
        self.part_mesh = None
        self.is_visible = False
        self.visibility_percentage = 0.0

        # Flag to track whether boolean operations should be attempted
        self.use_boolean_operations = True

        # Get the part from the planning scene
        self._get_planning_scene_interface()
        self._load_part_mesh()

        # Set up update timer
        update_rate = 10.0  # Hz
        self.update_timer = rospy.Timer(
            rospy.Duration(1.0 / update_rate), self._update_callback
        )

        rospy.loginfo("FrustumDetection initialized")

    def _load_camera_config(self):
        """Load camera configuration from YAML file."""
        config_path = os.path.join(
            self.rospack.get_path("inspection_cell"), "config", "perception_camera.yaml"
        )

        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)["camera"]
            rospy.loginfo("Camera configuration loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load camera config: {str(e)}")
            # Set default configuration
            self.config = {
                "frame": {
                    "reference_frame": "tool0",
                    "transform": {
                        "position": [0.0, 0.0, 0.1],  # 10cm forward from tool0
                        "orientation": [0.0, 0.0, 0.0],  # Z aligned with tool0
                    },
                },
                "field_of_view": {
                    "horizontal_fov": 60.0,
                    "vertical_fov": 45.0,
                    "min_range": 0.1,
                    "max_range": 2.0,
                },
                "visualization": {
                    "show_camera_frame": True,
                    "show_frustum": True,
                    "frustum_color": [0.0, 0.7, 1.0, 0.3],
                    "show_intersection": True,
                },
            }

    def _create_publishers(self):
        """Create publishers for visualization and camera outputs."""
        # Standard camera output topics
        self.rgb_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/camera/depth/image_raw", Image, queue_size=1)
        self.pointcloud_pub = rospy.Publisher(
            "/camera/depth/points", PointCloud2, queue_size=1
        )
        self.camera_info_pub = rospy.Publisher(
            "/camera/camera_info", CameraInfo, queue_size=1
        )

        # Visualization topics
        self.camera_frame_pub = rospy.Publisher(
            "/frustum_detection/camera_frame", Marker, queue_size=1
        )
        self.frustum_pub = rospy.Publisher(
            "/frustum_detection/frustum", Marker, queue_size=1
        )
        self.intersection_pub = rospy.Publisher(
            "/frustum_detection/intersection", Marker, queue_size=1
        )
        self.part_pub = rospy.Publisher("/frustum_detection/part", Marker, queue_size=1)

    def _get_planning_scene_interface(self):
        """Get interface to the planning scene."""
        try:
            from moveit_commander import PlanningSceneInterface

            self.scene = PlanningSceneInterface()
            rospy.loginfo("Connected to MoveIt planning scene")
        except Exception as e:
            rospy.logerr(f"Failed to connect to planning scene: {str(e)}")
            self.scene = None

    def _load_part_mesh(self):
        """Load the part mesh from the planning scene."""
        if not self.scene:
            rospy.logerr("No planning scene interface available")
            return

        # Wait for planning scene to be populated
        rospy.sleep(1.0)

        # Get objects from the scene
        scene_objects = self.scene.get_objects()

        # Look for the part object
        if "part" not in scene_objects:
            rospy.logwarn("Part object not found in planning scene")
            return

        # Get the mesh file for the part
        # For this example, we'll create a simple box mesh
        # In practice, you would load the actual mesh from the STL file
        extents = (0.1, 0.1, 0.1)  # 10cm box
        self.part_mesh = trimesh.creation.box(extents)

        # Position the mesh based on the planning scene object
        part_pose = (
            scene_objects["part"].primitive_poses[0]
            if scene_objects["part"].primitive_poses
            else Pose()
        )
        translation = [part_pose.position.x, part_pose.position.y, part_pose.position.z]

        # Apply the translation to the mesh
        self.part_mesh.apply_translation(translation)

        rospy.loginfo(f"Part mesh loaded and positioned at {translation}")

        # Visualize the part
        self._visualize_part()

    def _update_callback(self, event=None):
        """Timer callback to update camera pose and check visibility."""
        # Update camera pose
        if not self._update_camera_pose():
            return

        # Visualize camera and frustum
        self._visualize_camera_frame()
        self._visualize_frustum()

        # Check if part is visible in the frustum
        self._check_part_visibility()

        # Generate and publish camera outputs
        self._publish_camera_data()

    def _update_camera_pose(self):
        """Update the camera pose based on the current tool position."""
        try:
            # Get the transform from base_link to the reference frame (tool0)
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                self.config["frame"]["reference_frame"],
                rospy.Time(0),
                rospy.Duration(1.0),
            )

            # Store transform for later use
            self.camera_transform = transform

            # Extract position and orientation
            self.camera_pos = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )

            self.camera_rot = np.array(
                [
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ]
            )

            return True
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup failed: {str(e)}")
            return False

    def _visualize_camera_frame(self):
        """Visualize the camera coordinate frame."""
        marker = Marker()
        marker.header.frame_id = self.config["frame"]["reference_frame"]
        marker.header.stamp = rospy.Time.now()
        marker.ns = "camera_frame"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Arrow pointing in the Z direction (camera's viewing direction)
        marker.scale.x = 0.01  # Shaft diameter
        marker.scale.y = 0.02  # Head diameter
        marker.scale.z = 0.03  # Head length

        # Blue color for Z-axis
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Position at camera offset
        camera_offset = self.config["frame"]["transform"]["position"]
        marker.pose.position.x = camera_offset[0]
        marker.pose.position.y = camera_offset[1]
        marker.pose.position.z = camera_offset[2]

        # Orient based on camera orientation
        camera_orientation = self.config["frame"]["transform"]["orientation"]
        q = tf.transformations.quaternion_from_euler(
            camera_orientation[0], camera_orientation[1], camera_orientation[2]
        )
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        # Publish the marker
        self.camera_frame_pub.publish(marker)

    def _visualize_frustum(self):
        """Visualize the camera frustum."""
        marker = Marker()
        marker.header.frame_id = self.config["frame"]["reference_frame"]
        marker.header.stamp = rospy.Time.now()
        marker.ns = "camera_frustum"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Line width
        marker.scale.x = 0.005

        # Set color from config
        color = self.config["visualization"]["frustum_color"]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3] if len(color) > 3 else 0.5

        # Calculate frustum vertices
        h_fov = math.radians(self.config["field_of_view"]["horizontal_fov"] / 2)
        v_fov = math.radians(self.config["field_of_view"]["vertical_fov"] / 2)
        near = self.config["field_of_view"]["min_range"]
        far = self.config["field_of_view"]["max_range"]

        # Calculate dimensions at near and far planes
        near_height = 2 * near * math.tan(v_fov)
        near_width = 2 * near * math.tan(h_fov)
        far_height = 2 * far * math.tan(v_fov)
        far_width = 2 * far * math.tan(h_fov)

        # Define camera position (origin of frustum)
        camera_offset = self.config["frame"]["transform"]["position"]

        # Near plane corners (z forward, y up, x right in camera frame)
        near_top_right = Point(near_width / 2, near_height / 2, near)
        near_top_left = Point(-near_width / 2, near_height / 2, near)
        near_bottom_right = Point(near_width / 2, -near_height / 2, near)
        near_bottom_left = Point(-near_width / 2, -near_height / 2, near)

        # Far plane corners
        far_top_right = Point(far_width / 2, far_height / 2, far)
        far_top_left = Point(-far_width / 2, far_height / 2, far)
        far_bottom_right = Point(far_width / 2, -far_height / 2, far)
        far_bottom_left = Point(-far_width / 2, -far_height / 2, far)

        # Camera origin
        origin = Point(0, 0, 0)

        # Define lines for the frustum
        lines = []

        # Near plane
        lines.extend([near_top_left, near_top_right])
        lines.extend([near_top_right, near_bottom_right])
        lines.extend([near_bottom_right, near_bottom_left])
        lines.extend([near_bottom_left, near_top_left])

        # Far plane
        lines.extend([far_top_left, far_top_right])
        lines.extend([far_top_right, far_bottom_right])
        lines.extend([far_bottom_right, far_bottom_left])
        lines.extend([far_bottom_left, far_top_left])

        # Connecting lines
        lines.extend([near_top_left, far_top_left])
        lines.extend([near_top_right, far_top_right])
        lines.extend([near_bottom_left, far_bottom_left])
        lines.extend([near_bottom_right, far_bottom_right])

        # Set the points
        marker.points = lines

        # Apply camera position offset
        marker.pose.position.x = camera_offset[0]
        marker.pose.position.y = camera_offset[1]
        marker.pose.position.z = camera_offset[2]

        # Apply camera orientation
        camera_orientation = self.config["frame"]["transform"]["orientation"]
        q = tf.transformations.quaternion_from_euler(
            camera_orientation[0], camera_orientation[1], camera_orientation[2]
        )
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        # Publish the frustum
        self.frustum_pub.publish(marker)

    def _check_part_visibility(self):
        """Check if the part is visible within the camera frustum."""
        if not self.part_mesh:
            rospy.logwarn("Part mesh not available for visibility check")
            self.is_visible = False
            self.visibility_percentage = 0.0
            return

        # Convert camera transform to a 4x4 matrix
        camera_matrix = np.eye(4)

        # Add translation from base_link to camera
        camera_matrix[:3, 3] = self.camera_pos

        # Add rotation from base_link to camera
        rot_matrix = tf.transformations.quaternion_matrix(self.camera_rot)
        camera_matrix[:3, :3] = rot_matrix[:3, :3]

        # Apply the camera offset and orientation
        offset = self.config["frame"]["transform"]["position"]
        orientation = self.config["frame"]["transform"]["orientation"]

        # Convert offset and orientation to a 4x4 matrix
        offset_matrix = np.eye(4)
        offset_matrix[:3, 3] = offset

        orientation_matrix = tf.transformations.euler_matrix(
            orientation[0], orientation[1], orientation[2]
        )

        # Camera pose in world frame is the composition of these transforms
        camera_pose = np.dot(camera_matrix, np.dot(offset_matrix, orientation_matrix))

        # Get frustum parameters
        h_fov = math.radians(self.config["field_of_view"]["horizontal_fov"] / 2)
        v_fov = math.radians(self.config["field_of_view"]["vertical_fov"] / 2)
        near = self.config["field_of_view"]["min_range"]
        far = self.config["field_of_view"]["max_range"]

        # Create frustum using our custom function instead of trimesh.creation.frustum
        frustum = create_frustum(
            fovy=v_fov * 2, aspect=math.tan(h_fov) / math.tan(v_fov), near=near, far=far
        )

        # Transform frustum to camera pose
        frustum.apply_transform(camera_pose)

        # Try boolean operations only if they haven't failed before
        if self.use_boolean_operations:
            try:
                # Check if both meshes are watertight (required for boolean operations)
                if frustum.is_watertight and self.part_mesh.is_watertight:
                    intersection = trimesh.boolean.intersection(
                        [frustum, self.part_mesh]
                    )
                    if intersection is not None and intersection.volume > 0:
                        self.is_visible = True
                        self.visibility_percentage = (
                            100.0 * intersection.volume / self.part_mesh.volume
                        )
                        # Visualize the intersection if enabled
                        if self.config["visualization"]["show_intersection"]:
                            self._visualize_intersection(intersection)
                    else:
                        self.is_visible = False
                        self.visibility_percentage = 0.0
                else:
                    # Meshes aren't watertight, don't try boolean operations anymore
                    self.use_boolean_operations = False
                    rospy.logwarn(
                        "One or both meshes are not watertight volumes. Using bounding box check instead."
                    )
                    self._check_visibility_with_bbox(frustum)
            except ValueError as e:
                # Handle "Not all meshes are volumes!" error
                rospy.logwarn(
                    f"Boolean operation failed: {e}. Switching to bounding box check for future updates."
                )
                self.use_boolean_operations = False
                self._check_visibility_with_bbox(frustum)
            except Exception as e:
                rospy.logerr(f"Error in visibility check: {e}")
                self.use_boolean_operations = False
                self.is_visible = False
                self.visibility_percentage = 0.0
        else:
            # Skip boolean operations since they failed before
            self._check_visibility_with_bbox(frustum)

        rospy.logdebug(
            f"Part visibility: {self.is_visible}, percentage: {self.visibility_percentage:.1f}%"
        )

    def _check_visibility_with_bbox(self, frustum):
        """Check visibility using bounding box overlap as a fallback method.

        Args:
            frustum: The camera frustum mesh
        """
        # Get the bounding boxes
        frustum_bbox = frustum.bounding_box
        part_bbox = self.part_mesh.bounding_box

        # Check if bounding boxes overlap using manual check
        # Get bounds as min/max corners
        frustum_min, frustum_max = frustum_bbox.bounds
        part_min, part_max = part_bbox.bounds

        # Check if the boxes overlap in all three dimensions
        overlap = (
            frustum_min[0] <= part_max[0]
            and frustum_max[0] >= part_min[0]
            and frustum_min[1] <= part_max[1]
            and frustum_max[1] >= part_min[1]
            and frustum_min[2] <= part_max[2]
            and frustum_max[2] >= part_min[2]
        )

        if overlap:
            # Approximate visibility based on distance from camera to part center
            part_center = self.part_mesh.centroid
            camera_pos = np.array(
                [self.camera_pos[0], self.camera_pos[1], self.camera_pos[2]]
            )

            # Calculate distance
            distance = np.linalg.norm(camera_pos - part_center)
            max_distance = self.config["field_of_view"]["max_range"]
            min_distance = self.config["field_of_view"]["min_range"]

            # Normalize distance to 0-1 range and invert (closer = higher visibility)
            normalized_distance = 1.0 - min(
                1.0, max(0.0, (distance - min_distance) / (max_distance - min_distance))
            )

            # Approximate visibility percentage
            self.is_visible = True
            self.visibility_percentage = normalized_distance * 100.0

            # Create a simple intersection visualization
            if self.config["visualization"]["show_intersection"]:
                self._visualize_approximate_intersection(frustum_bbox, part_bbox)
        else:
            self.is_visible = False
            self.visibility_percentage = 0.0

    def _visualize_approximate_intersection(self, frustum_bbox, part_bbox):
        """Visualize an approximate intersection between bounding boxes.

        Args:
            frustum_bbox: Bounding box of the frustum
            part_bbox: Bounding box of the part
        """
        # Create marker for visualization
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "part_intersection"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Calculate the overlapping region of the bounding boxes
        # This is a simplification - in reality, we'd need to calculate the actual intersection
        # of the two boxes, but for visualization purposes this approximation is sufficient
        center = (frustum_bbox.centroid + part_bbox.centroid) / 2.0
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]

        # Set orientation to identity
        marker.pose.orientation.w = 1.0

        # Set scale to approximate intersection size (this is a simple approximation)
        avg_scale = (np.array(frustum_bbox.extents) + np.array(part_bbox.extents)) / 4.0
        marker.scale.x = max(avg_scale[0], 0.01)
        marker.scale.y = max(avg_scale[1], 0.01)
        marker.scale.z = max(avg_scale[2], 0.01)

        # Set color (yellow for approximate intersection)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        # Publish the marker
        self.intersection_pub.publish(marker)

    def _visualize_intersection(self, intersection_mesh):
        """Visualize the intersection between the frustum and part."""
        if not intersection_mesh:
            return

        # Create marker for visualization
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "part_intersection"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD

        # For a real mesh, we would save the intersection mesh to a file and load it here
        # For this example, we'll just use a cube marker
        marker.type = Marker.CUBE

        # Set position to center of intersection
        center = intersection_mesh.centroid
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]

        # Set orientation to identity
        marker.pose.orientation.w = 1.0

        # Set scale based on bounding box
        bounds = intersection_mesh.bounds
        scale = bounds[1] - bounds[0]
        marker.scale.x = max(scale[0], 0.01)
        marker.scale.y = max(scale[1], 0.01)
        marker.scale.z = max(scale[2], 0.01)

        # Set color (green for intersection)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.7

        # Publish the marker
        self.intersection_pub.publish(marker)

    def _visualize_part(self):
        """Visualize the part mesh."""
        if not self.part_mesh:
            return

        # Create marker for visualization
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "part"
        marker.id = 0
        marker.type = Marker.CUBE  # Using a cube for simplicity
        marker.action = Marker.ADD

        # Set position to center of part
        center = self.part_mesh.centroid
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]

        # Set orientation to identity
        marker.pose.orientation.w = 1.0

        # Set scale based on bounding box
        bounds = self.part_mesh.bounds
        scale = bounds[1] - bounds[0]
        marker.scale.x = max(scale[0], 0.01)
        marker.scale.y = max(scale[1], 0.01)
        marker.scale.z = max(scale[2], 0.01)

        # Set color (blue for part)
        marker.color.r = 0.2
        marker.color.g = 0.2
        marker.color.b = 1.0
        marker.color.a = 0.5

        # Publish the marker
        self.part_pub.publish(marker)

    def _publish_camera_data(self):
        """Publish simulated camera data streams."""
        # Only publish if the part is visible
        if not self.is_visible:
            return

        # Get camera resolution
        width, height = 640, 480

        # 1. RGB Image
        rgb_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Fill with a gradient
        for y in range(height):
            for x in range(width):
                rgb_img[y, x] = [int(255 * x / width), int(255 * y / height), 128]

        # Add a marker in the center showing the part
        cv2.circle(
            rgb_img,
            center=(width // 2, height // 2),
            radius=int(50 * self.visibility_percentage / 100.0),
            color=(0, 255, 0),
            thickness=-1,
        )

        # Convert to ROS Image and publish
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
        rgb_msg.header.stamp = rospy.Time.now()
        rgb_msg.header.frame_id = self.config["frame"]["reference_frame"]
        self.rgb_pub.publish(rgb_msg)

        # 2. Depth Image
        depth_img = np.ones((height, width), dtype=np.float32) * 5.0  # 5m background

        # Add a circular depth region in the center
        center_x, center_y = width // 2, height // 2
        radius = int(100 * self.visibility_percentage / 100.0)

        for y in range(height):
            for x in range(width):
                # Calculate distance from center
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist < radius:
                    # Depth value decreases towards center
                    depth_img[y, x] = 1.0 - 0.5 * (1 - dist / radius)

        # Convert to ROS Image and publish
        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_img, encoding="32FC1")
        depth_msg.header.stamp = rospy.Time.now()
        depth_msg.header.frame_id = self.config["frame"]["reference_frame"]
        self.depth_pub.publish(depth_msg)

        # 3. Camera Info
        camera_info = CameraInfo()
        camera_info.header.stamp = rospy.Time.now()
        camera_info.header.frame_id = self.config["frame"]["reference_frame"]
        camera_info.width = width
        camera_info.height = height

        # Set intrinsic parameters
        fx = fy = width / (
            2
            * math.tan(math.radians(self.config["field_of_view"]["horizontal_fov"]) / 2)
        )
        cx = width / 2
        cy = height / 2

        camera_info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        camera_info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        camera_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        camera_info.D = [0, 0, 0, 0, 0]  # No distortion
        camera_info.distortion_model = "plumb_bob"

        self.camera_info_pub.publish(camera_info)

        # 4. Point Cloud
        # Create a simple point cloud from the depth image
        points = []

        # Sample a subset of points for efficiency
        step = 10
        for y in range(0, height, step):
            for x in range(0, width, step):
                # Skip points at max depth
                if depth_img[y, x] >= 5.0:
                    continue

                # Calculate 3D point
                depth = depth_img[y, x]

                # Use camera model to project to 3D
                z = depth
                x3d = (x - cx) * z / fx
                y3d = (y - cy) * z / fy

                # Get RGB color
                r, g, b = rgb_img[y, x]

                # Pack RGB into an integer
                rgb_packed = (r << 16) | (g << 8) | b

                # Add the point
                points.append([x3d, y3d, z, rgb_packed])

        # Create point cloud message
        if points:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.config["frame"]["reference_frame"]

            fields = [
                pc2.PointField(
                    name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1
                ),
                pc2.PointField(
                    name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1
                ),
                pc2.PointField(
                    name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1
                ),
                pc2.PointField(
                    name="rgb", offset=12, datatype=pc2.PointField.UINT32, count=1
                ),
            ]

            pc_msg = pc2.create_cloud(header, fields, points)
            self.pointcloud_pub.publish(pc_msg)

    def is_part_visible(self):
        """Return whether the part is visible in the camera frustum."""
        return self.is_visible

    def get_visibility_percentage(self):
        """Return the percentage of the part's volume visible in the frustum."""
        return self.visibility_percentage

    def shutdown(self):
        """Shutdown the detection system."""
        if hasattr(self, "update_timer"):
            self.update_timer.shutdown()
        rospy.loginfo("SimulatedPerception shutdown")


if __name__ == "__main__":
    # Test the frustum detection
    detector = SimulatedPerception()
    rospy.loginfo("SimulatedPerception is running. Press Ctrl+C to stop.")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        detector.shutdown()
