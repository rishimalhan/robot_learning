#! /usr/bin/env python3

import torch
import numpy as np
import rospy
from dataclasses import dataclass
from moveit_commander import PlanningSceneInterface
import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import time
from tf.transformations import euler_matrix, euler_from_quaternion
from typing import Any


def get_param(param_name: str, default: Any = None) -> Any:
    """Get a parameter from the ROS parameter server."""
    return rospy.get_param(param_name, default)


@dataclass
class CameraParameters:
    """Camera parameters for point cloud generation."""

    frame_id: str

    # Geometric constraints
    max_normal_angle: (
        float  # Maximum angle between triangle normal and camera direction (degrees)
    )
    min_vertical_distance: float  # Minimum vertical distance (meters)
    max_vertical_distance: float  # Maximum vertical distance (meters)
    max_horizontal_distance: float  # Maximum horizontal distance (meters)

    # Sensing parameters
    noise_horizontal: float  # Standard deviation for horizontal Gaussian noise (meters)
    noise_vertical: float  # Standard deviation for vertical Gaussian noise (meters)
    noise_distance_factor: float  # Factor for scaling noise with distance


class SimulatedPerception:
    """Generate point clouds directly from mesh vertices using efficient GPU operations."""

    def __init__(self):
        """Initialize the point cloud generator."""
        rospy.loginfo("Initializing SimulatedPerception...")

        # Select optimal device for computation
        self.device = self._select_optimal_device()
        rospy.loginfo(f"Using device: {self.device}")

        # Get configuration parameters
        self.camera_params = self._load_camera_parameters()

        # Initialize mesh storage
        self.mesh_vertices = None
        self.mesh_faces = None

        # Cached mesh data (these don't change between frames)
        self.face_vertices = None
        self.face_centers = None
        self.face_normals = None
        self.face_normals_flipped = False

        # Set up TF listener for camera transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Set up point cloud publisher
        self.pointcloud_pub = rospy.Publisher(
            "camera/points", PointCloud2, queue_size=1
        )

        # Load mesh from planning scene
        self._load_mesh_from_scene()

        # Pre-compute mesh data that doesn't change
        self._precompute_mesh_data()

        rospy.loginfo("SimulatedPerception initialized successfully")

    def _select_optimal_device(self):
        """Select the optimal device for computation."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_camera_parameters(self):
        """Load camera parameters from ROS parameter server."""
        camera_frame = get_param("/perception/camera/frame", "tool0")

        # Load constraint parameters
        constraints = get_param("/perception/camera/constraints", {})
        max_normal_angle = constraints.get("max_normal_angle", 30.0)

        vertical_distance = constraints.get("vertical_distance", {})
        min_vertical_distance = vertical_distance.get("min", 0.05)
        max_vertical_distance = vertical_distance.get("max", 0.5)

        horizontal_distance = constraints.get("horizontal_distance", {})
        max_horizontal_distance = horizontal_distance.get("max", 0.3)

        # Load sensing parameters
        sensing = get_param("/perception/camera/sensing", {})
        noise_config = sensing.get("noise", {})

        # Get noise parameters with defaults
        noise_horizontal = noise_config.get("horizontal", 0.0)
        noise_vertical = noise_config.get("vertical", 0.0)
        noise_distance_factor = noise_config.get("distance_factor", 0.0)

        # Clamp noise levels to valid range
        noise_horizontal = max(0.0, min(1.0, noise_horizontal))
        noise_vertical = max(0.0, min(1.0, noise_vertical))
        noise_distance_factor = max(0.0, min(1.0, noise_distance_factor))

        # Create camera parameters
        params = CameraParameters(
            frame_id=camera_frame,
            max_normal_angle=max_normal_angle,
            min_vertical_distance=min_vertical_distance,
            max_vertical_distance=max_vertical_distance,
            max_horizontal_distance=max_horizontal_distance,
            noise_horizontal=noise_horizontal,
            noise_vertical=noise_vertical,
            noise_distance_factor=noise_distance_factor,
        )

        rospy.loginfo(f"Camera parameters loaded: {params}")
        return params

    def _load_mesh_from_scene(self):
        """Load mesh data from the planning scene."""
        rospy.loginfo("Loading mesh from planning scene...")

        # Get object name from parameter server
        mesh_config = get_param("/perception/mesh", {})
        object_name = mesh_config.get("scene_object_name", "part")
        rospy.loginfo(f"Looking for object: {object_name}")

        # Connect to planning scene
        scene = PlanningSceneInterface()
        rospy.sleep(1.0)

        # Get object from scene
        scene_objects = scene.get_objects()
        if not scene_objects:
            rospy.logerr("No objects found in planning scene")
            raise ValueError("No objects found in planning scene")

        rospy.loginfo(f"Found objects in scene: {list(scene_objects.keys())}")

        # Get the target object
        obj = scene_objects.get(object_name)
        if not obj:
            rospy.logerr(f"Required object '{object_name}' not found")
            raise ValueError(
                f"Required object '{object_name}' not found. Available: {', '.join(scene_objects.keys())}"
            )

        # Validate mesh data exists
        if not getattr(obj, "meshes", None):
            rospy.logerr(f"Object '{object_name}' has no mesh data")
            raise ValueError(f"Object '{object_name}' has no mesh data")

        # Extract mesh data
        mesh_data = obj.meshes[0]

        # Extract vertices and faces
        vertices = [[float(p.x), float(p.y), float(p.z)] for p in mesh_data.vertices]
        faces = [[int(idx) for idx in t.vertex_indices] for t in mesh_data.triangles]

        # Check if faces exist
        if not faces:
            rospy.logerr(f"Object '{object_name}' has no triangle data")
            raise ValueError(f"Object '{object_name}' has no triangle data")

        rospy.loginfo(f"Mesh has {len(vertices)} vertices and {len(faces)} faces")

        if not vertices:
            rospy.logerr(f"Object '{object_name}' has empty vertex data")
            raise ValueError(f"Object '{object_name}' has empty vertex data")

        # Convert to tensors on selected device
        vertices_tensor = torch.tensor(
            vertices, dtype=torch.float32, device=self.device
        )
        faces_tensor = torch.tensor(faces, dtype=torch.int64, device=self.device)

        # Apply mesh pose if available
        if hasattr(obj, "pose"):
            pose = obj.pose
            translation = [pose.position.x, pose.position.y, pose.position.z]

            # Convert quaternion to rotation matrix
            quat = [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
            rot_matrix = np.array(
                euler_matrix(*euler_from_quaternion(quat), "sxyz")[:3, :3],
                dtype=np.float32,
            )

            # Apply transformation to vertices
            if not np.allclose(rot_matrix, np.eye(3)):
                rot_tensor = torch.tensor(
                    rot_matrix, dtype=torch.float32, device=self.device
                )
                vertices_tensor = torch.matmul(
                    vertices_tensor, rot_tensor.transpose(0, 1)
                )

            # Apply translation to vertices
            translation_tensor = torch.tensor(
                translation, dtype=torch.float32, device=self.device
            )
            vertices_tensor = vertices_tensor + translation_tensor

            rospy.loginfo(
                f"Applied mesh transform: position {translation}, rotation {quat}"
            )

        # Store mesh data
        self.mesh_vertices = vertices_tensor
        self.mesh_faces = faces_tensor

        # Use all faces regardless of mesh size, no performance optimization
        rospy.loginfo(f"Using all {len(faces)} faces for geometric filtering")

        rospy.loginfo("Mesh loaded successfully")

    def _precompute_mesh_data(self):
        """Pre-compute mesh data that doesn't change between frames."""
        if self.mesh_vertices is None or self.mesh_faces is None:
            rospy.logwarn("Mesh data not available, cannot pre-compute mesh data")
            return

        with torch.no_grad():
            # Get vertices of each triangle in world space
            self.face_vertices = self.mesh_vertices[self.mesh_faces]

            # Calculate face centers
            self.face_centers = torch.mean(self.face_vertices, dim=1)

            # Calculate triangle normals in world space
            v0, v1, v2 = (
                self.face_vertices[:, 0],
                self.face_vertices[:, 1],
                self.face_vertices[:, 2],
            )
            normals = torch.cross(v1 - v0, v2 - v0)

            # Normalize normals
            normal_lengths = torch.norm(normals, dim=1, keepdim=True)
            # Avoid division by zero
            normal_lengths[normal_lengths < 1e-10] = 1.0
            normals = normals / normal_lengths

            # Ensure normals point outward from the mesh
            # Calculate the centroid of the mesh
            mesh_centroid = torch.mean(self.mesh_vertices, dim=0)

            # Calculate vector from centroid to face center
            centroid_to_face = self.face_centers - mesh_centroid

            # Normalize these vectors
            centroid_dist = torch.norm(centroid_to_face, dim=1, keepdim=True)
            centroid_dist[centroid_dist < 1e-10] = 1.0
            centroid_to_face = centroid_to_face / centroid_dist

            # Dot product to check if normal and centroid-to-face have similar direction
            normal_alignment = torch.sum(normals * centroid_to_face, dim=1)

            # Flip normals that point inward (negative dot product with centroid-to-face)
            flip_mask = normal_alignment < 0
            normals[flip_mask] = -normals[flip_mask]

            # Store the pre-computed normals
            self.face_normals = normals

            # Log how many normals were flipped
            flip_count = torch.sum(flip_mask).item()
            rospy.loginfo(
                f"Pre-computed {len(normals)} face normals, flipped {flip_count} to point outward"
            )
            self.face_normals_flipped = True

    def get_camera_transform(self):
        """Get the camera transform from TF."""
        try:
            # Look up the transform from world to camera frame
            transform = self.tf_buffer.lookup_transform(
                "world", self.camera_params.frame_id, rospy.Time(0), rospy.Duration(0.1)
            )

            # Extract translation and rotation
            trans = transform.transform.translation
            rot = transform.transform.rotation

            # Create transform matrix
            transform_matrix = np.eye(4, dtype=np.float32)
            transform_matrix[:3, :3] = np.array(
                euler_matrix(*euler_from_quaternion([rot.x, rot.y, rot.z, rot.w]))[
                    :3, :3
                ],
                dtype=np.float32,
            )
            transform_matrix[:3, 3] = [trans.x, trans.y, trans.z]

            return transform_matrix

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"Could not get camera transform: {e}")

            # Use a default transform (in front of the scene)
            transform_matrix = np.eye(4, dtype=np.float32)
            transform_matrix[:3, 3] = [1.0, 0.0, 1.0]
            return transform_matrix

    def generate_pointcloud(self):
        """Generate point cloud directly from mesh vertices through camera projection."""
        if self.mesh_vertices is None or self.mesh_faces is None:
            rospy.logwarn_throttle(
                5.0, "Mesh data not available, cannot generate point cloud"
            )
            return None

        # Check if we've pre-computed mesh data
        if self.face_normals is None:
            rospy.logwarn_throttle(5.0, "Mesh data not pre-computed, doing it now")
            self._precompute_mesh_data()

        # Record start time for performance monitoring
        start_time = time.time()

        # Get camera transform
        camera_transform = self.get_camera_transform()
        camera_transform_tensor = torch.tensor(
            camera_transform, dtype=torch.float32, device=self.device
        )

        # Camera Z axis (forward direction) in world coordinates
        camera_forward = camera_transform[:3, 2]  # Third column is Z axis

        # Camera forward vector as tensor
        camera_forward_tensor = torch.tensor(
            camera_forward, dtype=torch.float32, device=self.device
        )

        # Camera X and Y axes for noise application
        camera_x = camera_transform[:3, 0]  # First column is X axis
        camera_y = camera_transform[:3, 1]  # Second column is Y axis
        camera_z = camera_transform[:3, 2]  # Third column is Z axis

        # Convert to tensors once
        camera_x_tensor = torch.tensor(
            camera_x, dtype=torch.float32, device=self.device
        )
        camera_y_tensor = torch.tensor(
            camera_y, dtype=torch.float32, device=self.device
        )
        camera_z_tensor = torch.tensor(
            camera_z, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            # Now that normals are correctly oriented, calculate angle with camera
            normal_dot_view = torch.abs(
                torch.matmul(self.face_normals, camera_forward_tensor)
            )
            # Convert to angle in degrees
            normal_angles = (
                torch.acos(torch.clamp(normal_dot_view, -1.0, 1.0)) * 180.0 / np.pi
            )

            # Filter by maximum normal angle
            max_angle_rad = self.camera_params.max_normal_angle
            visible_faces = normal_angles <= max_angle_rad

            # Get inverse of camera transform (camera_T_world)
            camera_transform_inv = torch.inverse(camera_transform_tensor)

            # For each visible face, process its vertices
            visible_face_indices = torch.nonzero(visible_faces, as_tuple=True)[0]

            if len(visible_face_indices) == 0:
                rospy.logwarn_throttle(
                    5.0,
                    "No faces visible from current camera view based on normal angle",
                )
                return np.array([], dtype=np.float32).reshape(0, 3)

            # Get vertices of visible faces
            visible_face_vertices = self.face_vertices[visible_face_indices]

            # Flatten to get all vertices from visible faces (may include duplicates)
            vertices = visible_face_vertices.reshape(-1, 3)

            # Transform vertices to camera space - more efficient batch operation
            ones = torch.ones(
                (vertices.shape[0], 1),
                dtype=torch.float32,
                device=self.device,
            )
            vertices_homogeneous = torch.cat([vertices, ones], dim=1)

            # More efficient matrix multiplication
            vertices_camera = torch.matmul(vertices_homogeneous, camera_transform_inv.T)

            # Get camera space coordinates
            x, y, z = (
                vertices_camera[:, 0],
                vertices_camera[:, 1],
                vertices_camera[:, 2],
            )

            # Calculate vertical distance (along Z axis)
            vertical_distance = z

            # Calculate horizontal distance (perpendicular to Z axis)
            horizontal_distance = torch.sqrt(x * x + y * y)

            # Apply constraints
            in_vertical_range = (
                vertical_distance >= self.camera_params.min_vertical_distance
            ) & (vertical_distance <= self.camera_params.max_vertical_distance)

            in_horizontal_range = (
                horizontal_distance <= self.camera_params.max_horizontal_distance
            )

            # Combine constraints
            valid_vertices = in_vertical_range & in_horizontal_range

            if torch.sum(valid_vertices) == 0:
                rospy.logwarn_throttle(
                    5.0,
                    "No points visible from current camera view based on distance constraints",
                )
                return np.array([], dtype=np.float32).reshape(0, 3)

            # Get filtered world space points
            valid_world_points = vertices[valid_vertices]

            # Also get camera space coordinates of valid points
            valid_camera_points = vertices_camera[valid_vertices]

            # Log filtered results with throttling
            rospy.logdebug_throttle(
                1.0,
                f"Filtered points: {len(valid_world_points)} of {len(vertices)} | "
                f"Faces: {len(visible_face_indices)} of {len(self.mesh_faces)}",
            )

            # Add Gaussian noise to the point cloud if noise parameters > 0
            if (
                self.camera_params.noise_horizontal > 0
                or self.camera_params.noise_vertical > 0
                or self.camera_params.noise_distance_factor > 0
            ):

                # Get point distances for distance-dependent noise scaling
                distances = torch.norm(valid_camera_points[:, :3], dim=1)
                distance_scale = (
                    1.0 + distances * self.camera_params.noise_distance_factor
                )

                # Create noise tensors
                noise = torch.zeros_like(valid_world_points)

                # Generate horizontal noise (in X and Y) - vectorized
                if self.camera_params.noise_horizontal > 0:
                    # Generate random noise along camera X and Y directions
                    x_noise = (
                        torch.randn(valid_world_points.size(0), device=self.device)
                        * self.camera_params.noise_horizontal
                    )
                    y_noise = (
                        torch.randn(valid_world_points.size(0), device=self.device)
                        * self.camera_params.noise_horizontal
                    )

                    # Scale by distance factor
                    x_noise = x_noise * distance_scale
                    y_noise = y_noise * distance_scale

                    # Vectorized noise application
                    noise += torch.einsum("i,j->ij", x_noise, camera_x_tensor)
                    noise += torch.einsum("i,j->ij", y_noise, camera_y_tensor)

                # Generate vertical noise (along camera Z) - vectorized
                if self.camera_params.noise_vertical > 0:
                    # Generate random noise along camera Z direction
                    z_noise = (
                        torch.randn(valid_world_points.size(0), device=self.device)
                        * self.camera_params.noise_vertical
                    )

                    # Scale by distance factor
                    z_noise = z_noise * distance_scale

                    # Vectorized noise application
                    noise += torch.einsum("i,j->ij", z_noise, camera_z_tensor)

                # Apply combined noise
                valid_world_points = valid_world_points + noise

                # Log noise info with throttling
                rospy.logdebug_throttle(
                    1.0,
                    f"Added noise: h={self.camera_params.noise_horizontal:.4f}, "
                    f"v={self.camera_params.noise_vertical:.4f}, "
                    f"scale={self.camera_params.noise_distance_factor:.2f}",
                )

            # Convert to numpy only at the end
            pointcloud = valid_world_points.cpu().numpy()

            # Log performance metrics with throttling
            elapsed = time.time() - start_time
            rospy.logdebug_throttle(
                1.0, f"Point cloud generation took {elapsed:.3f} seconds"
            )

            return pointcloud

    def publish_pointcloud(self):
        """Generate and publish the point cloud."""
        try:
            # Generate point cloud
            pointcloud = self.generate_pointcloud()

            if pointcloud is None or len(pointcloud) == 0:
                rospy.logwarn_throttle(5.0, "No points to publish")
                return

            # Create point cloud message
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            # Create message with world frame
            cloud_msg = pc2.create_cloud(
                std_msgs.msg.Header(frame_id="world", stamp=rospy.Time.now()),
                fields,
                pointcloud,
            )

            # Publish message
            self.pointcloud_pub.publish(cloud_msg)

            # Log with throttling
            rospy.logdebug_throttle(
                1.0, f"Published point cloud with {len(pointcloud)} points"
            )

        except Exception as e:
            rospy.logerr(f"Error publishing point cloud: {e}")

    def cleanup(self):
        """Clean up resources."""
        rospy.loginfo("Cleaning up resources...")
        # Clear GPU memory if using CUDA or MPS
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def run_node():
    """Run the point cloud generator node."""
    # Initialize ROS node
    rospy.init_node("simulated_perception", disable_signals=False)

    # Get rate parameter
    rate_hz = get_param("/perception/camera/sensing/frame_rate", 10)  # Default to 10Hz
    rospy.loginfo(f"Publishing at {rate_hz} Hz")

    generator = None
    try:
        # Create generator
        generator = SimulatedPerception()

        # Set update rate
        rate = rospy.Rate(rate_hz)

        rospy.loginfo("Simulated perception is running")

        # Main loop - use ROS's built-in shutdown handling
        while not rospy.is_shutdown():
            try:
                # Record frame start time
                frame_start = time.time()

                # Generate and publish point cloud
                generator.publish_pointcloud()

                # Calculate actual processing time
                processing_time = time.time() - frame_start

                # Log if processing takes too long
                if processing_time > 1.0 / rate_hz:
                    rospy.logwarn_throttle(
                        5.0,
                        f"Frame processing time ({processing_time:.3f}s) exceeds target frame time ({1.0/rate_hz:.3f}s)",
                    )

                # Sleep to maintain rate
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("Interrupted, shutting down")
                break
            except Exception as e:
                rospy.logerr(f"Error in main loop: {e}")
                time.sleep(0.1)  # Short sleep on error
    except Exception as e:
        rospy.logerr(f"Error initializing point cloud generator: {e}")
    finally:
        if generator:
            generator.cleanup()
        rospy.loginfo("Node shutting down")


if __name__ == "__main__":
    # Run the node with standard ROS exception handling
    try:
        run_node()
    except rospy.ROSInterruptException:
        pass
