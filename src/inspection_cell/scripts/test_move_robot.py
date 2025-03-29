#!/usr/bin/env python3

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
        rospy.loginfo(
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
                rospy.loginfo(f"Removed {removed} outlier points")

            return filtered_points
        except Exception as e:
            rospy.logwarn(f"Error filtering outliers: {e}")
            return points

    def _remove_duplicate_points(self, points, distance_threshold=0.005):
        """Remove duplicate points that are very close to each other using voxel grid"""
        rospy.loginfo(f"Deduplicating points: {len(points)} input points")

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

            rospy.loginfo(f"After deduplication: {len(unique_points)} points")
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
        rospy.loginfo(
            "Published persistent augmented pointcloud with %d points",
            len(self.persistent_points),
        )


def get_robot_roi_bounds(env):
    """Get the X,Y bounds of the robot_roi from the planning scene"""
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

    # Get position from primitive pose
    position = roi_object.primitive_poses[0].position
    position_list = [position.x, position.y, position.z]

    # Calculate bounds (half-width from center position)
    half_x, half_y, half_z = [d / 2 for d in dimensions]
    x, y, z = position_list

    x_min, x_max = x - half_x, x + half_x
    y_min, y_max = y - half_y, y + half_y

    # Create and return bounds
    roi_bounds = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z": z,  # Center Z height
    }

    rospy.loginfo(
        f"ROI bounds: X: [{x_min:.3f}, {x_max:.3f}], Y: [{y_min:.3f}, {y_max:.3f}], Z: {z:.3f}"
    )
    return roi_bounds


def main():
    # Initialize ROS node
    rospy.init_node("test_robot_motion_node", anonymous=True)
    env = EnvironmentLoader()
    executor = Executor()

    # Initialize the pointcloud processor
    pointcloud_processor = PointCloudProcessor()

    # Get the robot_roi bounds
    roi_bounds = get_robot_roi_bounds(env)
    if not roi_bounds:
        rospy.logerr("Required robot_roi object not found in the scene. Exiting.")
        return

    # Give time for subscriptions to establish
    rospy.sleep(1.0)

    # Move to home position first
    rospy.loginfo("Moving to home position...")
    success, plan, planning_time, error_code = env.planner.plan_to_named_target("home")
    if success and plan:
        executor.execute_plan(plan)
        pointcloud_processor.create_augmented_pointcloud()
    else:
        rospy.logerr("Failed to plan to home position")
        return

    success, plan, planning_time, error_code = env.planner.plan_to_ee_offset(
        direction=[0, 0, 1], distance=0.2
    )
    if success and plan:
        executor.execute_plan(plan)
        pointcloud_processor.create_augmented_pointcloud()

    # Number of random movements to perform
    num_movements = 10
    count = 0
    while count < num_movements:
        # Get current end effector pose
        current_pose = env.planner.move_group.get_current_pose().pose
        current_x = current_pose.position.x
        current_y = current_pose.position.y

        # Generate random target within ROI bounds
        target_x = random.uniform(roi_bounds["x_min"], roi_bounds["x_max"])
        target_y = random.uniform(roi_bounds["y_min"], roi_bounds["y_max"])

        # Calculate offset from current position
        offset_x = target_x - current_x
        offset_y = target_y - current_y

        # Calculate direction vector (normalized)
        distance = np.sqrt(offset_x**2 + offset_y**2)

        # Ensure minimum movement distance
        if distance < 0.05:  # If movement too small, skip
            continue

        # Calculate direction and distance
        direction_x = offset_x / distance
        direction_y = offset_y / distance

        # Plan and execute movement in X direction
        rospy.loginfo(f"Moving to X: {target_x:.3f}, Y: {target_y:.3f}")
        success, plan, planning_time, error_code = env.planner.plan_to_ee_offset(
            direction=[direction_x, direction_y, 0],
            distance=distance,
        )

        if success and plan:
            executor.execute_plan(plan)
            pointcloud_processor.create_augmented_pointcloud()
        else:
            rospy.logwarn(
                f"Failed to plan movement to [{target_x:.3f}, {target_y:.3f}]"
            )
            continue

        # Small delay between movements
        rospy.sleep(0.5)
        rospy.loginfo(f"Random movement {count+1}/{num_movements}... done")
        count += 1
    # Return to home position
    rospy.loginfo("Returning to home position...")
    success, plan, planning_time, error_code = env.planner.plan_to_named_target("home")
    if success and plan:
        executor.execute_plan(plan)
        pointcloud_processor.create_augmented_pointcloud()

    rospy.loginfo(
        "Random movement exploration complete, continuing to process pointclouds..."
    )
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
