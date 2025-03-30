#!/usr/bin/env python3

import math
import rospy
import numpy as np
import random
from inspection_cell.load_system import EnvironmentLoader
from inspection_cell.planner import Planner
from inspection_cell.executor import Executor
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from collections import deque
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive, Mesh
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Quaternion
import traceback
import open3d as o3d
import uuid


class PointCloudProcessor:
    def __init__(self):
        # Initialize collections to store point clouds
        self.pointcloud_buffer = deque(
            maxlen=5
        )  # Store last 5 pointclouds for recent data
        self.latest_pointcloud = None
        self.augmented_pointcloud = None
        self.persistent_points = None  # Store all points persistently

        # Parameters for point cloud processing
        self.voxel_size = 0.005  # 5mm voxel size for deduplication (was 0.01)
        self.outlier_threshold = 0.1  # 10cm threshold for statistical outlier removal

        # Subscribe to the camera pointcloud topic
        self.pointcloud_sub = rospy.Subscriber(
            "camera/points", PointCloud2, self.pointcloud_callback, queue_size=1
        )

        # Set up publisher for the augmented pointcloud
        self.reconstruction_pub = rospy.Publisher(
            "reconstruction", PointCloud2, queue_size=1
        )

        # Set up publisher for the reconstructed mesh
        self.mesh_pub = rospy.Publisher("reconstructed_mesh", MarkerArray, queue_size=1)

        # Mesh reconstruction parameters
        self.alpha = 0.05  # Alpha value for Alpha shape reconstruction
        self.downsample_voxel_size = (
            0.01  # Downsample to 1cm voxels before mesh creation
        )
        self.min_mesh_points = 100  # Minimum points needed for mesh reconstruction
        self.mesh_color = [0.0, 0.8, 0.3, 0.6]  # RGBA (semi-transparent green)
        self.mesh_publish_interval = (
            10  # Only publish mesh every X point clouds to avoid flooding
        )
        self.point_count = 0  # Counter for mesh publishing interval

        # Suppress Open3D console output to avoid flooding
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

        rospy.loginfo("PointCloud processor initialized")

    def pointcloud_callback(self, cloud_msg):
        """Process incoming pointcloud data"""
        rospy.logdebug(
            "Received pointcloud with %d points", cloud_msg.width * cloud_msg.height
        )

        # Store the message
        self.latest_pointcloud = cloud_msg

        # Convert PointCloud2 to a numpy array of points
        points = pc2.read_points(cloud_msg, skip_nans=True)
        points_array = np.array(list(points))

        if len(points_array) == 0:
            rospy.logwarn("Received empty pointcloud")
            return

        # Apply basic noise filtering to remove outliers
        if points_array.shape[0] > 10:  # Only filter if enough points
            points_array = self._filter_outliers(points_array)

        # Add to buffer
        self.pointcloud_buffer.append(points_array)

        # Add to persistent points store
        if self.persistent_points is None:
            self.persistent_points = points_array
        else:
            # When combining, ensure we don't create artificial seams
            self.persistent_points = np.vstack([self.persistent_points, points_array])

            # Apply deduplication more frequently but with higher precision
            if (
                len(self.persistent_points) > 5000
            ):  # Lower threshold to process more often
                self.persistent_points = self._remove_duplicate_points(
                    self.persistent_points, distance_threshold=self.voxel_size
                )

        # Create augmented pointcloud
        self.create_augmented_pointcloud()

    def _filter_outliers(self, points):
        """Filter out statistical outliers from point cloud"""
        try:
            # Calculate distance to nearest neighbors for each point
            # Simple approach: use distance from each point to the mean position
            xyz = points[:, :3]
            mean_point = np.mean(xyz, axis=0)
            distances = np.sqrt(np.sum((xyz - mean_point) ** 2, axis=1))

            # Calculate standard deviation and identify outliers
            std_dev = np.std(distances)
            inlier_indices = distances < (self.outlier_threshold + 2 * std_dev)

            filtered_points = points[inlier_indices]
            removed = len(points) - len(filtered_points)
            if removed > 0:
                rospy.logdebug(f"Removed {removed} outlier points")

            return filtered_points
        except Exception as e:
            rospy.logwarn(f"Error filtering outliers: {e}")
            return points

    def _remove_duplicate_points(self, points, distance_threshold=0.005):
        """Remove duplicate points that are very close to each other using voxel grid"""
        rospy.logdebug(f"Deduplicating points: {len(points)} input points")

        try:
            # Create grid with cells of size distance_threshold
            min_vals = np.min(points[:, :3], axis=0)
            max_vals = np.max(points[:, :3], axis=0)

            # Add small epsilon to avoid division by zero
            ranges = np.maximum(max_vals - min_vals, 1e-6)

            # Calculate voxel indices for each point
            voxel_indices = np.floor(
                (points[:, :3] - min_vals) / distance_threshold
            ).astype(int)

            # Use a more efficient method for large point clouds
            # Create a unique voxel ID for each point
            h1, h2, h3 = voxel_indices.max(axis=0) + 1
            voxel_ids = (
                voxel_indices[:, 0]
                + voxel_indices[:, 1] * h1
                + voxel_indices[:, 2] * h1 * h2
            )

            # Find unique voxels and their centroids
            unique_voxel_ids, inverse_indices = np.unique(
                voxel_ids, return_inverse=True
            )

            # Keep only one point per voxel (first occurrence)
            # This is faster than the dictionary approach for large clouds
            unique_indices = np.zeros(len(unique_voxel_ids), dtype=int)
            for i, idx in enumerate(inverse_indices):
                if unique_indices[idx] == 0:
                    unique_indices[idx] = i

            # Remove zeros
            unique_indices = unique_indices[unique_indices > 0]

            # Get the unique points
            unique_points = points[unique_indices]

            rospy.logdebug(f"After deduplication: {len(unique_points)} points")
            return unique_points
        except Exception as e:
            rospy.logwarn(f"Error in deduplication: {e}, keeping original points")
            return points

    def create_augmented_pointcloud(self):
        """Create an augmented pointcloud from all persistent points and generate mesh"""
        if self.persistent_points is None or len(self.persistent_points) == 0:
            rospy.logwarn(
                "No persistent points available to create augmented pointcloud"
            )
            return

        # Create pointcloud fields (XYZ)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Create header with world frame
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"  # Explicitly set to world frame

        # Create PointCloud2 message from persistent points
        self.augmented_pointcloud = pc2.create_cloud(
            header, fields, self.persistent_points[:, :3]  # Use only XYZ coordinates
        )

        # Publish the augmented pointcloud
        self.reconstruction_pub.publish(self.augmented_pointcloud)

        # Create and publish mesh from the point cloud
        self.create_and_publish_mesh()

        rospy.logdebug(
            "Published persistent augmented pointcloud with %d points",
            len(self.persistent_points),
        )

    def create_and_publish_mesh(self):
        """Create a mesh from the point cloud and publish it as markers"""
        if (
            self.persistent_points is None
            or len(self.persistent_points) < self.min_mesh_points
        ):
            return

        # Only regenerate mesh periodically to avoid overwhelming RViz and console
        self.point_count += 1
        if self.point_count % self.mesh_publish_interval != 0:
            return

        try:
            # Convert numpy array to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.persistent_points[:, :3])

            # Statistical outlier removal to clean up the point cloud
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            # Downsample for efficiency and better mesh quality
            pcd = pcd.voxel_down_sample(voxel_size=self.downsample_voxel_size)

            # Ensure we still have enough points after filtering
            if len(pcd.points) < self.min_mesh_points:
                rospy.logwarn(f"Too few points after preprocessing: {len(pcd.points)}")
                return

            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

            # Use Alpha shape reconstruction for the visible surfaces
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, self.alpha
            )

            # Clean up the mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            # Create marker array
            marker_array = MarkerArray()

            # Create mesh marker
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "reconstructed_mesh"
            marker.id = 0
            marker.type = Marker.TRIANGLE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color.r = self.mesh_color[0]
            marker.color.g = self.mesh_color[1]
            marker.color.b = self.mesh_color[2]
            marker.color.a = self.mesh_color[3]

            # Add triangles to marker
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            # Skip if no triangles were created
            if len(triangles) == 0:
                rospy.logwarn("No triangles generated in the mesh")
                return

            # Add all triangles to the marker
            for triangle in triangles:
                for vertex_idx in triangle:
                    point = Point()
                    point.x = vertices[vertex_idx, 0]
                    point.y = vertices[vertex_idx, 1]
                    point.z = vertices[vertex_idx, 2]
                    marker.points.append(point)

            # Add marker to array
            marker_array.markers.append(marker)

            # Publish marker array
            self.mesh_pub.publish(marker_array)
            rospy.loginfo(
                f"Published reconstructed mesh with {len(triangles)} triangles"
            )

        except RuntimeError as e:
            # Handle Open3D specific errors
            error_msg = str(e)
            if "operator()" in error_msg or "Failed to close loop" in error_msg:
                rospy.logwarn(
                    "Open3D mesh construction error - using different alpha value next time"
                )
                # Adjust alpha value slightly to avoid the same error next time
                self.alpha = max(0.01, self.alpha * 0.9)
            else:
                rospy.logwarn(f"Error creating mesh: {e}")
        except Exception as e:
            rospy.logwarn(f"Error creating mesh: {e}")


def get_robot_roi_bounds(env):
    """Get the X,Y bounds of the robot_roi from the planning scene, including any translation"""
    rospy.loginfo("Retrieving robot_roi information from planning scene...")

    # Wait a bit for the scene to be fully loaded
    rospy.sleep(1.0)

    # Get all collision objects from the scene
    scene_objects = env.scene.get_objects()

    # Check if robot_roi exists in the scene
    if "robot_roi" not in scene_objects:
        rospy.logerr("robot_roi not found in planning scene objects")
        return None

    roi_object = scene_objects["robot_roi"]

    # Extract information from the collision object
    if not roi_object.primitives or roi_object.primitives[0].type != SolidPrimitive.BOX:
        rospy.logerr("robot_roi is not a box primitive")
        return None

    # Get dimensions from primitive
    dimensions = list(roi_object.primitives[0].dimensions)

    # Get position and orientation from pose
    position = roi_object.pose.position

    # Calculate bounds (half-width from center position)
    half_x, half_y, half_z = dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2
    x, y, z = position.x, position.y, position.z

    # Calculate bounds in world coordinates
    x_min, x_max = x - half_x, x + half_x
    y_min, y_max = y - half_y, y + half_y

    # Create and return bounds
    roi_bounds = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }

    rospy.loginfo(
        f"ROI bounds (in world frame): X: [{x_min:.3f}, {x_max:.3f}], Y: [{y_min:.3f}, {y_max:.3f}], Z: {z:.3f}"
    )
    return roi_bounds


def main():
    # Initialize ROS node
    rospy.init_node("test_robot_motion_node", anonymous=True)
    env = EnvironmentLoader()
    executor = Executor()

    # Get the robot_roi bounds
    roi_bounds = get_robot_roi_bounds(env)
    if not roi_bounds:
        rospy.logerr("Required robot_roi object not found in the scene. Exiting.")
        return

    # Initialize the pointcloud processor
    PointCloudProcessor()

    try:
        # Step 1: Move to home position
        rospy.loginfo("Step 1: Moving to home position...")
        success, plan, planning_time, error_code = env.planner.plan_to_named_target(
            "home"
        )
        if success and plan:
            executor.execute_plan(plan)
        else:
            rospy.logerr("Failed to plan to home position")
            return

        # Get current pose - we'll use this to calculate the z-offset
        current_pose = env.planner.move_group.get_current_pose("tool0").pose
        orientation = current_pose.orientation

        # Calculate Z position with 0.25m lower than current
        z_position = current_pose.position.z - 0.4

        # Step 2: Set up zigzag pattern waypoints
        rospy.loginfo("Step 2: Setting up zigzag pattern waypoints...")

        # Define grid size
        grid_points_x = 10  # Number of points in X direction
        grid_points_y = 10  # Number of points in Y direction

        # Calculate step sizes based on ROI bounds
        x_min = roi_bounds["x_min"] + 0.2
        x_max = roi_bounds["x_max"] - 0.2
        y_min = roi_bounds["y_min"] + 0.3
        y_max = roi_bounds["y_max"] - 0.3

        step_x = (x_max - x_min) / max(1, grid_points_x - 1)
        step_y = (y_max - y_min) / max(1, grid_points_y - 1)

        # Generate zigzag waypoints
        waypoints = []

        for y_idx in range(grid_points_y):
            # Determine if moving left-to-right or right-to-left
            is_left_to_right = y_idx % 2 == 0

            # Define x points based on direction
            if is_left_to_right:
                x_indices = range(grid_points_x)
            else:
                x_indices = range(grid_points_x - 1, -1, -1)

            # Add waypoints for this row
            for x_idx in x_indices:
                # Calculate actual position
                x_pos = x_min + x_idx * step_x
                y_pos = y_min + y_idx * step_y

                # Create pose with current orientation
                pose = Pose()
                pose.position.x = x_pos
                pose.position.y = y_pos
                pose.position.z = z_position
                pose.orientation = orientation

                waypoints.append(pose)

        # Step 3: Execute zigzag pattern using cartesian path
        rospy.loginfo(
            f"Step 3: Executing zigzag pattern with {len(waypoints)} waypoints..."
        )

        # Plan cartesian path
        plan, fraction, planning_time = env.planner.plan_cartesian_path(
            waypoints=waypoints,
            eef_step=0.001,
            jump_threshold=False,  # Allow jumps for smoother path
        )
        rospy.loginfo(
            f"Successfully planned {fraction:.1%} of the zigzag path in {planning_time:.2f} seconds"
        )

        # Execute the plan
        executor.execute_plan(plan)

        # Return to home position
        rospy.loginfo("Returning to home position...")
        success, plan, planning_time, error_code = env.planner.plan_to_named_target(
            "home"
        )
        if success and plan:
            executor.execute_plan(plan)

    except rospy.ROSInterruptException:
        rospy.loginfo("Program interrupted by user")
    except Exception as e:
        rospy.logerr(f"Error during execution: {str(e)}")
        rospy.logerr(f"Exception details: {traceback.format_exc()}")
    finally:
        rospy.loginfo("Clean shutdown. Exiting.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
