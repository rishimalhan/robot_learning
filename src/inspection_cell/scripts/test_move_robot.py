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
from shape_msgs.msg import SolidPrimitive


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
        """Create an augmented pointcloud from all persistent points"""
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
        rospy.logdebug(
            "Published persistent augmented pointcloud with %d points",
            len(self.persistent_points),
        )


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
    # Get orientation quaternion
    orientation = roi_object.pose.orientation

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

    try:
        # Step 1: Move to home position first
        rospy.loginfo("Step 1: Moving to home position...")
        success, plan, planning_time, error_code = env.planner.plan_to_named_target(
            "home"
        )
        if success and plan:
            executor.execute_plan(plan)
        else:
            rospy.logerr("Failed to plan to home position")
            return

        # Step 2: Move down along tool Z by 0.25 meters
        rospy.loginfo("Step 2: Moving down along tool Z by 0.25 meters...")
        success, plan, planning_time, error_code = env.planner.plan_to_ee_offset(
            direction=[0, 0, 1],
            distance=0.25,
        )
        if success and plan:
            executor.execute_plan(plan)
        else:
            rospy.logerr("Failed to move down along Z. Exiting.")
            return

        # Initialize the pointcloud processor
        pointcloud_processor = PointCloudProcessor()

        # Step 3: Make random movements in X-Y plane within roi_bounds
        rospy.loginfo(
            "Step 3: Making random movements in X-Y plane within ROI bounds..."
        )

        # Number of random movements to perform
        num_movements = 100
        successful_movements = 0
        max_attempts = 1000  # Prevent infinite loop if many movements fail

        attempt = 0
        while successful_movements < num_movements and attempt < max_attempts:
            if rospy.is_shutdown():
                rospy.loginfo("ROS shutdown detected. Exiting.")
                break

            attempt += 1
            rospy.loginfo(
                f"Attempting movement {attempt} (completed {successful_movements}/{num_movements})"
            )

            # Get current end effector pose
            current_pose = env.planner.move_group.get_current_pose(
                end_effector_link="tool0"
            ).pose
            current_x = current_pose.position.x
            current_y = current_pose.position.y

            # 1. Sample a random direction in X-Y plane
            angle = random.uniform(0, 2 * math.pi)  # Random angle in radians
            direction_x = math.cos(angle)
            direction_y = math.sin(angle)

            # 2. Sample a random distance between 0-0.3m
            distance_x = random.uniform(0.05, 0.3)  # Random distance between 0.05-0.3m
            distance_y = random.uniform(0.05, 0.8)  # Random distance between 0.05-0.3m

            # 3. Calculate target position
            target_x = current_x + (direction_x * distance_x)
            target_y = current_y + (direction_y * distance_y)

            # 4. Check if target is within ROI bounds
            if (
                target_x < roi_bounds["x_min"]
                or target_x > roi_bounds["x_max"]
                or target_y < roi_bounds["y_min"]
                or target_y > roi_bounds["y_max"]
            ):
                rospy.loginfo("Target outside ROI bounds, trying again...")
                continue

            # Plan and execute movement in X-Y plane only
            rospy.loginfo(
                f"Planning in direction [{direction_x:.3f}, {direction_y:.3f}] for {distance_x:.3f}m, {distance_y:.3f}m"
            )
            success, plan, planning_time, error_code = env.planner.plan_to_ee_offset(
                direction=[direction_x, direction_y, 0],
                distance=np.linalg.norm(
                    [direction_x * distance_x, direction_y * distance_y]
                ),
            )

            if success and plan:
                executor.execute_plan(plan)
                # Only create pointcloud after movement completes
                pointcloud_processor.create_augmented_pointcloud()
                successful_movements += 1
                rospy.loginfo(
                    f"Successfully completed movement {successful_movements}/{num_movements}"
                )
            else:
                rospy.logwarn(
                    f"Failed to plan movement in direction [{direction_x:.3f}, {direction_y:.3f}]"
                )

            # Check for ctrl+c between movements
            if rospy.is_shutdown():
                rospy.loginfo("Ctrl+C detected. Stopping movements.")
                break

            # Small delay between movements
            rospy.sleep(0.5)

        if successful_movements < num_movements and not rospy.is_shutdown():
            rospy.logwarn(
                f"Could only complete {successful_movements}/{num_movements} movements after {attempt} attempts"
            )

        # Return to home position
        if not rospy.is_shutdown():
            rospy.loginfo("Returning to home position...")
            success, plan, planning_time, error_code = env.planner.plan_to_named_target(
                "home"
            )
            if success and plan:
                executor.execute_plan(plan)
                pointcloud_processor.create_augmented_pointcloud()

    except rospy.ROSInterruptException:
        rospy.loginfo("Program interrupted by user")
    except Exception as e:
        rospy.logerr(f"Error during execution: {str(e)}")
    finally:
        rospy.loginfo("Clean shutdown. Exiting.")

    # No rospy.spin() to make it easier to interrupt


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
