#!/usr/bin/env python3

import rospy
import sys
import random
import math
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import traceback
import time
import argparse

from inspection_cell.load_system import EnvironmentLoader
from inspection_cell.planner import Planner
from inspection_cell.executor import Executor

from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Vector3
from std_msgs.msg import ColorRGBA
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.msg import RobotState, PlanningScene, ObjectColor
from visualization_msgs.msg import Marker, MarkerArray


class PoseSampler:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("pose_sampler", anonymous=True)

        # Initialize tracking sets for markers
        self.marker_ids = set()
        self.roi_marker_ids = set()

        # Track sampled and successful poses
        self.sampled_poses = []
        self.successful_poses = []

        # Initialize the environment loader
        self.env_loader = self._create_environment()

        # Create publishers for visualization
        self.pose_target_pub = rospy.Publisher(
            "/move_group/goal_pose", PoseStamped, queue_size=1
        )
        self.planning_scene_pub = rospy.Publisher(
            "/planning_scene", PlanningScene, queue_size=10
        )
        self.display_trajectory_pub = rospy.Publisher(
            "/move_group/display_planned_path", DisplayTrajectory, queue_size=10
        )
        # Add marker publishers for visualization without affecting collision checking
        self.marker_pub = rospy.Publisher(
            "/pose_sampler/visualization_markers", MarkerArray, queue_size=10
        )
        self.roi_markers_pub = rospy.Publisher(
            "/pose_sampler/roi_markers", MarkerArray, queue_size=1
        )

        # Get the robot_roi information from the config
        self.roi_info = self._get_roi_info()

        # Initialize the planner and executor
        self.planner = Planner(move_group_name="manipulator")
        self.executor = Executor(move_group_name="manipulator", check_collisions=True)

        rospy.loginfo("PoseSampler initialized")

    def _create_environment(self):
        """Create a modified version of EnvironmentLoader that doesn't initialize a node"""

        # Create a subclass of EnvironmentLoader
        class ModifiedEnvironmentLoader(EnvironmentLoader):
            def __init__(self, move_group_name="manipulator", clear_scene=True):
                # Skip the parent class's __init__ to avoid initializing a node
                import moveit_commander
                import rospkg
                import os
                import yaml
                from inspection_cell.collision_checker import CollisionCheck

                # Initialize attributes directly
                self.scene = moveit_commander.PlanningSceneInterface()
                self.move_group = moveit_commander.MoveGroupCommander(move_group_name)
                self.robot = moveit_commander.RobotCommander()
                self.group_name = move_group_name

                # Create RosPack instance
                self.rospack = rospkg.RosPack()

                # Get the planning frame
                self.planning_frame = self.move_group.get_planning_frame()
                rospy.loginfo(f"Planning frame: {self.planning_frame}")

                # Clear the planning scene if requested
                if clear_scene:
                    self.clear_scene()

                # Load environment configuration using rospkg
                config_path = os.path.join(
                    self.rospack.get_path("inspection_cell"),
                    "config",
                    "environment.yaml",
                )
                rospy.loginfo(f"Loading configuration from: {config_path}")
                with open(config_path, "r") as f:
                    self.config = yaml.safe_load(f)

                # Configure robot parameters
                self._configure_robot()

                # Add objects to scene
                self._add_objects_to_scene()

                # Wait for scene to update
                rospy.sleep(1.0)

                # Print current scene objects
                self._print_scene_objects()

                # Print the ACM status for debugging
                self.print_acm_status()

                # Initialize the collision checker AFTER environment is fully loaded
                rospy.loginfo(
                    "Initializing collision checker after environment setup..."
                )
                self.collision_checker = CollisionCheck(move_group_name=move_group_name)
                rospy.loginfo("Collision checker initialized")

        # Create and return the modified environment loader
        rospy.loginfo("Loading environment...")
        return ModifiedEnvironmentLoader(
            move_group_name="manipulator", clear_scene=True
        )

    def _get_roi_info(self):
        """Extract information about the robot_roi region directly from the planning scene"""
        rospy.loginfo("Retrieving robot_roi information from planning scene...")

        # Wait a bit longer for the scene to be fully loaded
        rospy.sleep(1.0)

        # Get all collision objects from the scene
        scene_objects = self.env_loader.scene.get_objects()

        # First log all available objects in the scene for debugging
        rospy.loginfo("Planning scene objects:")
        for obj_name in scene_objects:
            rospy.loginfo(f"  - {obj_name}")

        # Check if robot_roi exists in the scene
        if "robot_roi" not in scene_objects:
            rospy.logerr("robot_roi not found in planning scene objects")
            return None

        roi_object = scene_objects["robot_roi"]

        # Extract information from the collision object
        if (
            not roi_object.primitives
            or roi_object.primitives[0].type != SolidPrimitive.BOX
        ):
            rospy.logerr("robot_roi is not a box primitive")
            return None

        # Get dimensions from primitive
        dimensions = list(roi_object.primitives[0].dimensions)

        # Get pose from primitive pose
        position = roi_object.primitive_poses[0].position
        orientation = roi_object.primitive_poses[0].orientation

        # Convert position to list for easier handling
        position_list = [position.x, position.y, position.z]

        # Check if position is at origin (likely incorrect)
        if position_list == [0.0, 0.0, 0.0]:
            rospy.logwarn(
                "Warning: robot_roi position is at [0,0,0], which might be incorrect"
            )
            rospy.loginfo(
                "Attempting to get position from environment configuration..."
            )

            # Try to get position from environment configuration
            if "robot_roi" in self.env_loader.config["objects"]:
                config_position = self.env_loader.config["objects"]["robot_roi"][
                    "pose"
                ]["position"]
                rospy.loginfo(
                    f"Found robot_roi in config with position: {config_position}"
                )

                # Use the position from the config instead
                position_list = config_position
                rospy.loginfo(f"Using position from config: {position_list}")

        # Create ROI info dict
        roi_info = {
            "type": "box",
            "dimensions": dimensions,
            "pose": {
                "position": position_list,
                "orientation": [0, 0, 0],  # We assume 0 rotation for simplicity
            },
        }

        rospy.loginfo(f"Found robot_roi with dimensions: {dimensions}")
        rospy.loginfo(f"robot_roi position: {position_list}")

        # Additional debugging: Calculate the actual 3D bounds of the ROI
        half_x, half_y, half_z = [d / 2 for d in dimensions]
        x, y, z = position_list

        x_min, x_max = x - half_x, x + half_x
        y_min, y_max = y - half_y, y + half_y
        z_min, z_max = z - half_z, z + half_z

        rospy.loginfo(
            f"ROI bounds: X: [{x_min:.3f}, {x_max:.3f}], Y: [{y_min:.3f}, {y_max:.3f}], Z: [{z_min:.3f}, {z_max:.3f}]"
        )

        # Visualize the ROI as a success marker to confirm its position
        self._visualize_roi_corners(roi_info)

        return roi_info

    def _visualize_roi_corners(self, roi_info):
        """Visualize the corners of the ROI to confirm its position using visualization markers"""
        if not roi_info:
            return

        dimensions = roi_info["dimensions"]
        position = roi_info["pose"]["position"]

        # Calculate the corners of the ROI box
        half_dims = [d / 2 for d in dimensions]
        x, y, z = position

        # Create a marker array for the corners
        marker_array = MarkerArray()

        # Create markers for the 8 corners of the box
        for i, corner in enumerate(
            [
                (
                    x + half_dims[0],
                    y + half_dims[1],
                    z + half_dims[2],
                ),  # top, front, right
                (
                    x - half_dims[0],
                    y + half_dims[1],
                    z + half_dims[2],
                ),  # top, front, left
                (
                    x + half_dims[0],
                    y - half_dims[1],
                    z + half_dims[2],
                ),  # top, back, right
                (
                    x - half_dims[0],
                    y - half_dims[1],
                    z + half_dims[2],
                ),  # top, back, left
                (
                    x + half_dims[0],
                    y + half_dims[1],
                    z - half_dims[2],
                ),  # bottom, front, right
                (
                    x - half_dims[0],
                    y + half_dims[1],
                    z - half_dims[2],
                ),  # bottom, front, left
                (
                    x + half_dims[0],
                    y - half_dims[1],
                    z - half_dims[2],
                ),  # bottom, back, right
                (
                    x - half_dims[0],
                    y - half_dims[1],
                    z - half_dims[2],
                ),  # bottom, back, left
            ]
        ):
            # Create a marker for this corner
            marker = Marker()
            marker.header.frame_id = self.env_loader.planning_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "roi_corners"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Set the marker pose
            marker.pose.position.x = corner[0]
            marker.pose.position.y = corner[1]
            marker.pose.position.z = corner[2]
            marker.pose.orientation.w = 1.0

            # Set the marker scale (size)
            marker.scale = Vector3(0.02, 0.02, 0.02)  # Small sphere

            # Set the marker color (blue)
            marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.8)

            # Set marker lifetime (0 = forever)
            marker.lifetime = rospy.Duration(0)

            # Add the marker to the array
            marker_array.markers.append(marker)
            self.roi_marker_ids.add(i)

        # Create edge lines between corners to better visualize the box
        for i, (start_idx, end_idx) in enumerate(
            [
                (0, 1),
                (1, 3),
                (3, 2),
                (2, 0),  # Top face
                (4, 5),
                (5, 7),
                (7, 6),
                (6, 4),  # Bottom face
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),  # Vertical edges
            ]
        ):
            marker = Marker()
            marker.header.frame_id = self.env_loader.planning_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "roi_edges"
            marker.id = i + 8  # Start after the corner IDs
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # Add the points
            p1 = marker_array.markers[start_idx].pose.position
            p2 = marker_array.markers[end_idx].pose.position
            marker.points = [p1, p2]

            # Set the marker scale (width)
            marker.scale.x = 0.005  # Line width

            # Set the marker color (blue)
            marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.6)

            # Set marker lifetime (0 = forever)
            marker.lifetime = rospy.Duration(0)

            # Add the marker to the array
            marker_array.markers.append(marker)
            self.roi_marker_ids.add(i + 8)

        # Publish the marker array
        self.roi_markers_pub.publish(marker_array)
        rospy.loginfo("Visualized ROI corners and edges as blue markers")

    def visualize_pose(self, pose, successful=None):
        """Visualize a pose using MoveIt's target display in RViz and optional success/failure marker

        Args:
            pose: The pose to visualize
            successful: Whether the pose was successfully reached (None if not yet determined)
        """
        # Create a stamped pose for MoveIt visualization
        stamped_pose = PoseStamped()
        stamped_pose.header.frame_id = self.env_loader.planning_frame
        stamped_pose.header.stamp = rospy.Time.now()
        stamped_pose.pose = pose

        # Publish the pose target to show in RViz
        self.pose_target_pub.publish(stamped_pose)

        # If we want to color-code based on success, create a visual marker
        if successful is not None:
            # Create a marker for the result
            marker = Marker()
            marker.header.frame_id = self.env_loader.planning_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "pose_markers"
            marker.id = len(self.sampled_poses)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Set the marker pose
            marker.pose = pose

            # Set the marker scale (size)
            marker.scale = Vector3(0.025, 0.025, 0.025)  # Small sphere

            # Set the marker color based on success
            if successful:
                marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.7)  # Green
            else:
                marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.7)  # Red

            # Set marker lifetime (0 = forever)
            marker.lifetime = rospy.Duration(0)

            # Track the marker ID
            self.marker_ids.add(marker.id)

            # Create a marker array with just this marker
            marker_array = MarkerArray()
            marker_array.markers.append(marker)

            # Publish the marker
            self.marker_pub.publish(marker_array)

    def visualize_trajectory(self, plan):
        """Visualize a planned trajectory in RViz"""
        if not plan or not hasattr(plan, "joint_trajectory"):
            return

        # Create a DisplayTrajectory message for RViz
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = (
            self.planner.move_group.get_current_state()
        )
        display_trajectory.trajectory.append(plan)

        # Publish the trajectory
        self.display_trajectory_pub.publish(display_trajectory)

    def sample_pose_within_roi(self, orientation_strategy="random"):
        """
        Sample a random pose within the ROI with orientation based on the selected strategy

        Args:
            orientation_strategy: Strategy for orientation sampling
                'random' - completely random orientation
                'vertical' - tool pointing mostly downward
                'horizontal' - tool pointing mostly horizontally
                'look_at_center' - tool pointing toward the center of the ROI
                'downward' - tool Z axis pointing downward within 30 degrees

        Returns:
            geometry_msgs/Pose or None if sampling failed
        """
        if not self.roi_info:
            rospy.logerr("Cannot sample pose - ROI information not available")
            return None

        # Extract ROI dimensions and position
        roi_dimensions = self.roi_info["dimensions"]
        roi_position = self.roi_info["pose"]["position"]

        # Sample a random position within the ROI
        x_half_size = roi_dimensions[0] / 2.0
        y_half_size = roi_dimensions[1] / 2.0
        z_half_size = roi_dimensions[2] / 2.0

        # Log the sampling bounds for debugging
        rospy.loginfo(f"Sampling within bounds:")
        rospy.loginfo(
            f"  X: {roi_position[0] - x_half_size:.3f} to {roi_position[0] + x_half_size:.3f}"
        )
        rospy.loginfo(
            f"  Y: {roi_position[1] - y_half_size:.3f} to {roi_position[1] + y_half_size:.3f}"
        )
        rospy.loginfo(
            f"  Z: {roi_position[2] - z_half_size:.3f} to {roi_position[2] + z_half_size:.3f}"
        )

        # Sample random position strictly within the ROI bounds
        x = roi_position[0] + random.uniform(-x_half_size, x_half_size)
        y = roi_position[1] + random.uniform(-y_half_size, y_half_size)
        z = roi_position[2] + random.uniform(-z_half_size, z_half_size)

        # Log the sampled position for debugging
        rospy.loginfo(f"Sampled position: ({x:.3f}, {y:.3f}, {z:.3f})")

        # Verify the sampled position is within the ROI
        if (
            abs(x - roi_position[0]) > x_half_size
            or abs(y - roi_position[1]) > y_half_size
            or abs(z - roi_position[2]) > z_half_size
        ):
            rospy.logerr(f"Position outside ROI bounds: ({x:.3f}, {y:.3f}, {z:.3f})")
            return None

        # Create orientation based on strategy
        if orientation_strategy == "random":
            # Completely random orientation
            roll = random.uniform(-math.pi, math.pi)
            pitch = random.uniform(-math.pi, math.pi)
            yaw = random.uniform(-math.pi, math.pi)

        elif orientation_strategy == "vertical":
            # Tool pointing mostly downward
            roll = random.uniform(-math.pi / 4, math.pi / 4)  # Limited roll
            pitch = random.uniform(math.pi / 2, math.pi)  # Pointing downward
            yaw = random.uniform(-math.pi, math.pi)  # Any yaw

        elif orientation_strategy == "horizontal":
            # Tool pointing mostly horizontally, good for inspections along walls
            roll = random.uniform(-math.pi / 4, math.pi / 4)  # Limited roll
            pitch = random.uniform(-math.pi / 6, math.pi / 6)  # Near horizontal
            yaw = random.uniform(-math.pi, math.pi)  # Any direction

        elif orientation_strategy == "look_at_center":
            # Tool pointing toward the center of the ROI
            # Calculate vector from sampled point to ROI center
            dx = roi_position[0] - x
            dy = roi_position[1] - y
            dz = roi_position[2] - z

            # Convert vector to orientation
            # We need yaw (around z) and pitch (around y)
            yaw = math.atan2(dy, dx)

            # For pitch, we need the angle from the xy-plane
            distance_xy = math.sqrt(dx * dx + dy * dy)
            pitch = math.atan2(dz, distance_xy)

            # Roll can be random or fixed
            roll = random.uniform(-math.pi / 6, math.pi / 6)

        elif orientation_strategy == "downward":
            # Tool Z axis pointing downward with maximum 30 degree deviation
            # This means tool Z axis should be roughly aligned with world -Z axis

            # Convert 30 degrees to radians
            max_deviation = math.radians(30.0)

            # Start with a reference orientation where Z is pointing down (pitch = pi)
            # Then apply small random deviations within the 30 degree cone

            # Random deviation angles (spherical coordinates)
            # Theta is the deviation from straight down, limited to 30 degrees
            theta = random.uniform(0, max_deviation)
            # Phi is the direction of deviation (360 degrees possible)
            phi = random.uniform(0, 2 * math.pi)

            # Convert deviation angles to a direction vector
            # Note: When theta=0, this points straight down (-Z)
            dx = math.sin(theta) * math.cos(phi)
            dy = math.sin(theta) * math.sin(phi)
            dz = -math.cos(theta)  # Negative because we want to point down

            # Convert this direction vector to euler angles
            # For a Z-down orientation, we need pitch around Y axis to be approximately pi/2
            pitch = math.pi / 2 + math.asin(
                dz
            )  # offset by pi/2 because of reference orientation

            # Yaw (around Z) is determined by the X,Y components
            yaw = math.atan2(dy, dx)

            # Roll around new Z axis can be random
            # For a more stable grasp orientation, we could fix this
            roll = random.uniform(-math.pi, math.pi)

            rospy.loginfo(
                f"Downward orientation: theta={math.degrees(theta):.1f}°, "
                + f"deviation from vertical: {math.degrees(theta):.1f}°"
            )

        else:
            rospy.logwarn(
                f"Unknown orientation strategy: {orientation_strategy}, using random"
            )
            roll = random.uniform(-math.pi, math.pi)
            pitch = random.uniform(-math.pi, math.pi)
            yaw = random.uniform(-math.pi, math.pi)

        q = quaternion_from_euler(roll, pitch, yaw)

        # Create the pose
        pose = Pose()
        pose.position = Point(x, y, z)
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        return pose

    def plan_and_execute_to_pose(self, pose):
        """Plan and execute a motion to the given pose"""
        # Display the target pose in RViz
        self.visualize_pose(pose)

        # Log the target pose
        p = pose.position
        q = pose.orientation
        rpy = euler_from_quaternion([q.x, q.y, q.z, q.w])
        rospy.loginfo(
            f"Planning to pose: position=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), orientation=({rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f})"
        )

        # Plan to the pose
        success, plan, planning_time, error_code = self.planner.plan_to_pose_target(
            pose
        )

        if success and plan:
            rospy.loginfo(
                f"Plan to pose successful, planning time: {planning_time:.2f} seconds"
            )

            # Visualize the planned trajectory
            self.visualize_trajectory(plan)

            # Small delay to allow visualization
            rospy.sleep(0.5)

            # Execute the plan
            execution_success = self.executor.execute_plan(plan)

            if execution_success:
                rospy.loginfo("Successfully moved to pose")
                # Update the visualization to show success
                self.visualize_pose(pose, successful=True)
                return True
            else:
                rospy.logwarn("Failed to execute plan to pose")
                # Update the visualization to show failure
                self.visualize_pose(pose, successful=False)
                return False
        else:
            rospy.logwarn(f"Failed to plan to pose, error code: {error_code}")
            # Update the visualization to show failure
            self.visualize_pose(pose, successful=False)
            return False

    def move_to_home(self):
        """Move the robot to the home position"""
        rospy.loginfo("Moving to home position...")

        # Get available named targets
        named_targets = self.planner.get_named_targets()

        # Check if all-zeros is available
        if "all-zeros" in named_targets:
            # Plan to the named target
            success, plan, planning_time, error_code = (
                self.planner.plan_to_named_target("all-zeros")
            )
        else:
            # Plan to home
            success, plan, planning_time, error_code = self.planner.plan_to_home()

        # Visualize the trajectory if planning was successful
        if success and plan:
            self.visualize_trajectory(plan)

            # Execute the plan
            execution_success = self.executor.execute_plan(plan)

            if execution_success:
                rospy.loginfo("Successfully moved to home position")
                return True
            else:
                rospy.logwarn("Failed to execute plan to home position")
                return False
        else:
            rospy.logwarn("Failed to plan to home position")
            return False

    def sample_and_test_poses(
        self, num_poses=10, max_attempts=50, orientation_strategies=None
    ):
        """
        Sample poses and test if the robot can move to them

        Args:
            num_poses: Number of successful poses to collect
            max_attempts: Maximum number of sampling attempts
            orientation_strategies: List of orientation strategies to cycle through
                If None, uses ['downward', 'vertical', 'random', 'horizontal', 'look_at_center']
        """
        if orientation_strategies is None:
            orientation_strategies = [
                "downward",  # Add the new downward strategy as default
                "vertical",
                "random",
                "horizontal",
                "look_at_center",
            ]

        rospy.loginfo(f"Sampling and testing up to {num_poses} poses from the ROI")
        rospy.loginfo(f"Using orientation strategies: {orientation_strategies}")

        # First move to home position
        self.move_to_home()

        successful_count = 0
        attempt_count = 0

        # For cycling through strategies
        strategy_index = 0

        while successful_count < num_poses and attempt_count < max_attempts:
            attempt_count += 1

            # Get the current strategy
            current_strategy = orientation_strategies[
                strategy_index % len(orientation_strategies)
            ]
            strategy_index += 1

            rospy.loginfo(
                f"Attempt {attempt_count}: Using orientation strategy '{current_strategy}'"
            )

            # Sample a random pose within the ROI with the current strategy
            pose = self.sample_pose_within_roi(orientation_strategy=current_strategy)
            if not pose:
                continue

            # Store the sampled pose
            self.sampled_poses.append(pose)

            # Try to plan and execute a motion to the pose
            success = self.plan_and_execute_to_pose(pose)

            if success:
                successful_count += 1
                self.successful_poses.append(pose)
                rospy.loginfo(
                    f"Pose {successful_count}/{num_poses} was successful using strategy '{current_strategy}'"
                )

                # Pause at the successful pose so the user can see it
                rospy.sleep(1.0)
            else:
                rospy.loginfo(
                    f"Pose attempt {attempt_count}/{max_attempts} with strategy '{current_strategy}' failed"
                )

            # Periodically return to home to avoid getting stuck in awkward configurations
            if attempt_count % 5 == 0:
                self.move_to_home()

        # Return to home position after all attempts
        self.move_to_home()

        # Report results
        if successful_count >= num_poses:
            rospy.loginfo(
                f"Successfully found and executed {successful_count} poses in {attempt_count} attempts"
            )
        else:
            rospy.logwarn(
                f"Only found {successful_count}/{num_poses} successful poses after {attempt_count} attempts"
            )

        return successful_count

    def cleanup(self):
        """Clean up resources before shutting down"""
        rospy.loginfo("Cleaning up...")

        # Clear any visualization markers we've created
        if self.marker_ids or self.roi_marker_ids:
            # Create marker arrays for deletion
            pose_markers = MarkerArray()
            roi_markers = MarkerArray()

            # Create deletion markers for pose markers
            for marker_id in self.marker_ids:
                marker = Marker()
                marker.header.frame_id = self.env_loader.planning_frame
                marker.header.stamp = rospy.Time.now()
                marker.ns = "pose_markers"
                marker.id = marker_id
                marker.action = Marker.DELETE
                pose_markers.markers.append(marker)

            # Create deletion markers for ROI markers
            for marker_id in self.roi_marker_ids:
                # Delete corner markers
                corner_marker = Marker()
                corner_marker.header.frame_id = self.env_loader.planning_frame
                corner_marker.header.stamp = rospy.Time.now()
                corner_marker.ns = "roi_corners"
                corner_marker.id = marker_id
                corner_marker.action = Marker.DELETE
                roi_markers.markers.append(corner_marker)

                # Delete edge markers
                if marker_id >= 8:  # IDs 8-19 are edge markers
                    edge_marker = Marker()
                    edge_marker.header.frame_id = self.env_loader.planning_frame
                    edge_marker.header.stamp = rospy.Time.now()
                    edge_marker.ns = "roi_edges"
                    edge_marker.id = marker_id
                    edge_marker.action = Marker.DELETE
                    roi_markers.markers.append(edge_marker)

            # Publish the deletion markers
            if pose_markers.markers:
                self.marker_pub.publish(pose_markers)
                rospy.loginfo(
                    f"Cleared {len(pose_markers.markers)} pose visualization markers"
                )

            if roi_markers.markers:
                self.roi_markers_pub.publish(roi_markers)
                rospy.loginfo(
                    f"Cleared {len(roi_markers.markers)} ROI visualization markers"
                )

            # Allow time for deletion to process
            rospy.sleep(0.5)

    def set_roi_position(self, position_value=None):
        """
        Explicitly set the ROI position

        Args:
            position_value: Position [x, y, z] to use, or None to use default
        """
        if not self.roi_info:
            rospy.logerr("No ROI info available to update position")
            return False

        if position_value:
            self.roi_info["pose"]["position"] = position_value
            rospy.loginfo(f"Manually set ROI position to: {position_value}")

            # Recalculate bounds
            dimensions = self.roi_info["dimensions"]
            half_x, half_y, half_z = [d / 2 for d in dimensions]
            x, y, z = position_value

            x_min, x_max = x - half_x, x + half_x
            y_min, y_max = y - half_y, y + half_y
            z_min, z_max = z - half_z, z + half_z

            rospy.loginfo(
                f"Updated ROI bounds: X: [{x_min:.3f}, {x_max:.3f}], Y: [{y_min:.3f}, {y_max:.3f}], Z: [{z_min:.3f}, {z_max:.3f}]"
            )

            # Update visualization
            self._visualize_roi_corners(self.roi_info)
            return True

        return False


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="Sample and test end-effector poses in a region of interest"
        )
        parser.add_argument(
            "--num-poses",
            type=int,
            default=5,
            help="Number of successful poses to find",
        )
        parser.add_argument(
            "--max-attempts", type=int, default=30, help="Maximum sampling attempts"
        )
        parser.add_argument(
            "--strategies",
            type=str,
            default=None,
            help="Comma-separated list of orientation strategies: random,vertical,horizontal,look_at_center,downward",
        )
        parser.add_argument(
            "--roi-position",
            type=str,
            default=None,
            help="Override ROI position (format: x,y,z)",
        )

        # Parse arguments without actual sys.argv to avoid interfering with ROS
        args = parser.parse_args(rospy.myargv()[1:])

        # Process the strategies argument
        if args.strategies:
            strategies = args.strategies.split(",")
            # Validate strategies
            valid_strategies = [
                "random",
                "vertical",
                "horizontal",
                "look_at_center",
                "downward",
            ]
            for s in strategies:
                if s not in valid_strategies:
                    rospy.logwarn(
                        f"Invalid strategy '{s}', must be one of {valid_strategies}"
                    )
                    strategies = None
                    break
        else:
            strategies = None

        # Create the pose sampler
        sampler = PoseSampler()

        # Check if ROI was found
        if not sampler.roi_info:
            rospy.logerr("Unable to locate robot_roi in the planning scene. Aborting.")
            rospy.loginfo(
                "Please make sure there is a box object named 'robot_roi' in your environment configuration."
            )
            return

        # Process the roi-position argument if provided
        if args.roi_position:
            try:
                position = [float(x) for x in args.roi_position.split(",")]
                if len(position) == 3:
                    sampler.set_roi_position(position)
                else:
                    rospy.logwarn("ROI position must have 3 values (x,y,z)")
            except ValueError:
                rospy.logwarn(f"Invalid ROI position format: {args.roi_position}")

        # For debugging: If ROI position is at origin ([0,0,0]), use a fallback position
        if sampler.roi_info["pose"]["position"] == [0.0, 0.0, 0.0]:
            # Fallback to a reasonable position near the robot
            fallback_position = [
                1.0,
                0.0,
                0.75,
            ]  # Match the position in environment.yaml
            rospy.logwarn(f"Using fallback ROI position: {fallback_position}")
            sampler.set_roi_position(fallback_position)

        # Give time for publishers to connect
        rospy.sleep(1.0)

        # Sample and test poses
        sampler.sample_and_test_poses(
            num_poses=args.num_poses,
            max_attempts=args.max_attempts,
            orientation_strategies=strategies,
        )

        # Keep the visualization alive
        rospy.loginfo(
            "Sampling complete. Visualization remains active. Press Ctrl+C to exit."
        )
        rospy.spin()

    except Exception as e:
        rospy.logerr(f"Error in pose sampler: {e}")
        rospy.logerr(traceback.format_exc())
    finally:
        if "sampler" in locals():
            sampler.cleanup()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
