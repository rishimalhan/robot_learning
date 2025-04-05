#!/usr/bin/env python3

import rospy
import random
import math
import threading
import queue
import asyncio
from collections import deque
from tf.transformations import (
    quaternion_from_euler,
    quaternion_multiply,
    quaternion_from_matrix,
)
import traceback
import argparse
from inspection_cell.load_system import EnvironmentLoader
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, TransformStamped
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import PlanningScene
import rviz_tools_py as viz
import tf2_ros
import ros_numpy
import numpy as np
import time


class PoseSampler:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("pose_sampler", anonymous=True)

        # Track sampled and successful poses
        self.sampled_poses = []
        self.successful_poses = []

        # Setup queues for parallelism
        self.planning_queue = queue.Queue(maxsize=10)  # Queue of poses to plan
        self.execution_queue = queue.Queue(maxsize=10)  # Queue of plans to execute

        # Lock for visualization
        self.viz_lock = threading.Lock()

        # Event to signal shutdown
        self.shutdown_event = threading.Event()

        # Counters with locks
        self.attempt_count_lock = threading.Lock()
        self.attempt_count = 0
        self.success_count_lock = threading.Lock()
        self.success_count = 0

        # Flag for TF visualization - only show for executing poses
        self.show_tf = False
        self.current_executing_pose = None

        # Initialize the environment loader
        rospy.loginfo("Loading environment...")
        self.env = EnvironmentLoader(move_group_name="manipulator", clear_scene=True)

        # Initialize rviz_tools for visualization
        self.markers = viz.RvizMarkers("world", "viz")

        # Initialize TF broadcaster for pose visualization
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Create publishers for visualization
        self.planning_scene_pub = rospy.Publisher(
            "/planning_scene", PlanningScene, queue_size=10
        )

        # Get the robot_roi information from the config
        self.roi_info = self._get_roi_info()

        rospy.loginfo("PoseSampler initialized")

    def _get_roi_info(self):
        """Extract information about the robot_roi region directly from the planning scene"""
        rospy.loginfo("Retrieving robot_roi information from planning scene...")

        # Wait a bit longer for the scene to be fully loaded
        rospy.sleep(1.0)

        # Get all collision objects from the scene
        scene_objects = self.env.scene.get_objects()

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

        # Get pose from primitive
        position = ros_numpy.geometry.point_to_numpy(roi_object.pose.position)
        orientation = ros_numpy.geometry.quat_to_numpy(roi_object.pose.orientation)

        # Create ROI info dict
        roi_info = {
            "type": "box",
            "dimensions": dimensions,
            "pose": {
                "position": position,
                "orientation": orientation,
            },
        }

        rospy.loginfo(f"Found robot_roi with dimensions: {dimensions}")
        rospy.loginfo(f"robot_roi position: {position}")
        rospy.loginfo(f"robot_roi orientation: {orientation}")

        # Additional debugging: Calculate the actual 3D bounds of the ROI
        half_x, half_y, half_z = [d / 2 for d in dimensions]
        x, y, z = position

        x_min, x_max = x - half_x, x + half_x
        y_min, y_max = y - half_y, y + half_y
        z_min, z_max = z - half_z, z + half_z

        rospy.loginfo(
            f"ROI bounds: X: [{x_min:.3f}, {x_max:.3f}], Y: [{y_min:.3f}, {y_max:.3f}], Z: [{z_min:.3f}, {z_max:.3f}]"
        )
        return roi_info

    def visualize_pose(self, pose, successful=None, show_tf=False):
        """Visualize a pose using rviz_tools.

        Args:
            pose: The pose to visualize
            successful: Whether the pose was successfully reached (None if not yet determined)
            show_tf: Whether to publish TF frames (only for executing poses)
        """
        with self.viz_lock:
            # Create an axis marker at the pose
            axis_length = 0.1
            axis_radius = 0.01
            lifetime = 0  # 0 = forever

            # Visualize the pose with an axis marker
            self.markers.publishAxis(pose, axis_length, axis_radius, lifetime)

            # Add a sphere to indicate success/failure status if provided
            if successful is not None:
                # Use a sphere to indicate success/failure
                diameter = 0.025
                color = "green" if successful else "red"
                self.markers.publishSphere(pose, color, diameter, lifetime)

            # Publish TF frames only if requested (for executing poses)
            if show_tf:
                transforms = []

                # Create transform for TCP pose
                tcp_transform = TransformStamped()
                tcp_transform.header.stamp = rospy.Time.now()
                tcp_transform.header.frame_id = "world"
                tcp_transform.child_frame_id = (
                    "sampled_pose"  # Use fixed name to replace previous
                )

                # Set translation and rotation from the input pose
                tcp_transform.transform.translation.x = pose.position.x
                tcp_transform.transform.translation.y = pose.position.y
                tcp_transform.transform.translation.z = pose.position.z
                tcp_transform.transform.rotation = pose.orientation
                transforms.append(tcp_transform)

                # Get end-effector pose
                eef_pose = self.env.get_end_effector_transform(pose)

                # Create direct transform from TCP to end-effector
                tcp_to_eef = TransformStamped()
                tcp_to_eef.header.stamp = rospy.Time.now()
                tcp_to_eef.header.frame_id = "sampled_pose"
                tcp_to_eef.child_frame_id = (
                    "eef_pose"  # Use fixed name to replace previous
                )

                # Calculate the relative transform from TCP to end-effector
                # Get TCP and end-effector poses as numpy matrices
                tcp_matrix = ros_numpy.numpify(pose)
                eef_matrix = ros_numpy.numpify(eef_pose)

                # Calculate relative transform: tcp_T_eef = inv(tcp_T_world) * eef_T_world
                relative_transform = np.matmul(np.linalg.inv(tcp_matrix), eef_matrix)

                # Extract translation and rotation from the relative transform
                translation = relative_transform[:3, 3]
                rotation = quaternion_from_matrix(relative_transform)

                # Set the transform values
                tcp_to_eef.transform.translation.x = translation[0]
                tcp_to_eef.transform.translation.y = translation[1]
                tcp_to_eef.transform.translation.z = translation[2]
                tcp_to_eef.transform.rotation.x = rotation[0]
                tcp_to_eef.transform.rotation.y = rotation[1]
                tcp_to_eef.transform.rotation.z = rotation[2]
                tcp_to_eef.transform.rotation.w = rotation[3]

                transforms.append(tcp_to_eef)

                # Send the transforms using the class member broadcaster
                self.tf_broadcaster.sendTransform(transforms)

            # Also visualize the end-effector pose with a different color
            eef_pose = self.env.get_end_effector_transform(pose)
            self.markers.publishAxis(eef_pose, axis_length * 0.8, axis_radius, lifetime)

    def sample_pose_within_roi(self, max_angle=math.pi / 6):
        """
        Sample a random pose within the ROI with orientation within a cone around -Z axis.
        The Z-axis of the tool will point within max_angle radians of straight down.

        Args:
            max_angle: Maximum angle in radians from vertical (0 to pi)
                       Small values keep tool nearly vertical, pi/2 allows horizontal

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

        # Sample random position strictly within the ROI bounds
        x = roi_position[0] + random.uniform(-x_half_size, x_half_size)
        y = roi_position[1] + random.uniform(-y_half_size, y_half_size)
        z = roi_position[2] + random.uniform(-z_half_size, z_half_size)

        # Verify the sampled position is within the ROI
        if (
            abs(x - roi_position[0]) > x_half_size
            or abs(y - roi_position[1]) > y_half_size
            or abs(z - roi_position[2]) > z_half_size
        ):
            rospy.logerr(f"Position outside ROI bounds: ({x:.3f}, {y:.3f}, {z:.3f})")
            return None

        # Create orientation based on sampling Z first, then X, then Y
        # Clamp max_angle to valid range [0, pi]
        max_angle = max(0, min(math.pi, max_angle))

        # First sample Z direction within cone around [0,0,-1]
        # Sample random deviation angle and azimuth
        deviation_angle = random.uniform(0, max_angle)
        azimuth = random.uniform(0, 2 * math.pi)

        # Convert to Cartesian coordinates to get Z direction
        z_x = math.sin(deviation_angle) * math.cos(azimuth)
        z_y = math.sin(deviation_angle) * math.sin(azimuth)
        z_z = -math.cos(deviation_angle)  # Negative because we want -Z direction

        # Normalize Z vector
        z_norm = math.sqrt(z_x * z_x + z_y * z_y + z_z * z_z)
        z_x /= z_norm
        z_y /= z_norm
        z_z /= z_norm

        # Sample random angle for X direction in plane perpendicular to Z
        x_angle = random.uniform(0, 2 * math.pi)

        # Get arbitrary vector not parallel to Z
        temp_x = 1.0
        temp_y = 0.0
        temp_z = 0.0
        if abs(z_x) > 0.9:  # If Z is close to X axis, use Y axis instead
            temp_x = 0.0
            temp_y = 1.0
            temp_z = 0.0

        # Get perpendicular vector using cross product
        px = temp_y * z_z - temp_z * z_y
        py = temp_z * z_x - temp_x * z_z
        pz = temp_x * z_y - temp_y * z_x

        # Normalize perpendicular vector
        p_norm = math.sqrt(px * px + py * py + pz * pz)
        px /= p_norm
        py /= p_norm
        pz /= p_norm

        # Rotate perpendicular vector around Z by x_angle to get X direction
        cos_a = math.cos(x_angle)
        sin_a = math.sin(x_angle)
        x_x = px * cos_a + (z_y * pz - z_z * py) * sin_a
        x_y = py * cos_a + (z_z * px - z_x * pz) * sin_a
        x_z = pz * cos_a + (z_x * py - z_y * px) * sin_a

        # Get Y direction by cross product of Z and X
        y_x = z_y * x_z - z_z * x_y
        y_y = z_z * x_x - z_x * x_z
        y_z = z_x * x_y - z_y * x_x

        # Convert rotation matrix to quaternion using tf.transformations
        q = quaternion_from_matrix(
            [
                [x_x, x_y, x_z, 0.0],
                [y_x, y_y, y_z, 0.0],
                [z_x, z_y, z_z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Account for the TCP's 1 radian tilt around X if needed
        tcp_tilt = (
            0.0  # Set to 0 if no TCP tilt compensation needed, or to -1.0 if needed
        )
        if abs(tcp_tilt) > 1e-6:
            q_tilt = quaternion_from_euler(tcp_tilt, 0, 0)
            q = quaternion_multiply(q, q_tilt)

        # Create the pose
        pose = Pose()
        pose.position = Point(x, y, z)
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        return pose

    def sampler_worker(self, max_attempts, max_angle):
        """Worker thread that samples poses and adds them to the planning queue."""
        attempt_count = 0

        while attempt_count < max_attempts and not self.shutdown_event.is_set():
            # Sample a random pose within the ROI
            pose = self.sample_pose_within_roi(max_angle=max_angle)
            if not pose:
                continue

            # Update the attempt counter
            with self.attempt_count_lock:
                self.attempt_count += 1
                attempt_count += 1
                current_attempt = self.attempt_count

            rospy.loginfo(f"Sampled pose {current_attempt}/{max_attempts}")

            # Store the sampled pose
            self.sampled_poses.append(pose)

            # Add the pose to the planning queue (without visualizing)
            try:
                self.planning_queue.put(
                    (pose, current_attempt), block=True, timeout=1.0
                )
            except queue.Full:
                # If the planning queue is full, wait and try again
                rospy.loginfo("Planning queue full, waiting...")
                rospy.sleep(0.5)
                continue

        rospy.loginfo("Sampler worker finished")

    def planner_worker(self):
        """Worker thread that plans paths for sampled poses."""
        while not self.shutdown_event.is_set():
            try:
                # Get a pose from the queue
                pose, attempt_number = self.planning_queue.get(block=True, timeout=1.0)

                # Visualize the pose when we start planning for it (but don't show TF)
                self.visualize_pose(pose, show_tf=False)

                # Plan to the pose
                start_time = time.time()
                success, plan, planning_time, error_code = (
                    self.env.planner.plan_to_pose_target(
                        self.env.get_end_effector_transform(pose)
                    )
                )

                if success and plan:
                    total_time = time.time() - start_time
                    rospy.loginfo(
                        f"Plan for pose {attempt_number} successful, planning time: {planning_time:.2f}s, total: {total_time:.2f}s"
                    )

                    # Add plan to execution queue
                    try:
                        self.execution_queue.put(
                            (pose, plan, attempt_number), block=True, timeout=1.0
                        )
                    except queue.Full:
                        rospy.logwarn(
                            f"Execution queue full, dropping plan for pose {attempt_number}"
                        )
                        self.visualize_pose(pose, successful=False, show_tf=False)
                else:
                    rospy.logwarn(
                        f"Failed to plan for pose {attempt_number}, error code: {error_code}"
                    )
                    self.visualize_pose(pose, successful=False, show_tf=False)

                # Mark planning task as done
                self.planning_queue.task_done()

            except queue.Empty:
                # If there's nothing in the queue, sleep briefly
                rospy.sleep(0.1)
                continue

        rospy.loginfo("Planner worker finished")

    def executor_worker(self, target_success_count):
        """Worker thread that executes planned paths."""
        while not self.shutdown_event.is_set():
            try:
                # Check if we've reached our target success count
                with self.success_count_lock:
                    if self.success_count >= target_success_count:
                        rospy.loginfo(
                            f"Reached target of {target_success_count} successful poses, executor stopping"
                        )
                        break

                # Get a plan from the queue
                pose, plan, attempt_number = self.execution_queue.get(
                    block=True, timeout=1.0
                )

                # Update the currently executing pose and show TF for it
                self.visualize_pose(pose, show_tf=True)

                # Execute the plan asynchronously
                rospy.loginfo(f"Executing plan for pose {attempt_number}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Start execution
                    execution_future = loop.run_until_complete(
                        self.env.executor.execute_plan_async(plan)
                    )

                    # Wait for completion with timeout
                    execution_success = loop.run_until_complete(
                        asyncio.wait_for(execution_future, timeout=30.0)
                    )

                    if execution_success:
                        with self.success_count_lock:
                            self.success_count += 1
                            current_success = self.success_count

                        rospy.loginfo(
                            f"Successfully moved to pose {attempt_number}, success {current_success}/{target_success_count}"
                        )
                        self.successful_poses.append(pose)
                        self.visualize_pose(pose, successful=True, show_tf=True)

                        # Pause briefly at the successful pose
                        rospy.sleep(0.5)

                        # Move back to home asynchronously
                        home_future = loop.run_until_complete(
                            self.env.executor.move_to_home_async(self.env.planner)
                        )

                        # Wait for home completion with timeout
                        loop.run_until_complete(
                            asyncio.wait_for(home_future, timeout=30.0)
                        )
                    else:
                        rospy.logwarn(
                            f"Failed to execute plan for pose {attempt_number}"
                        )
                        self.visualize_pose(pose, successful=False, show_tf=False)

                except (asyncio.TimeoutError, Exception) as e:
                    rospy.logwarn(
                        f"Error executing plan for pose {attempt_number}: {e}"
                    )
                    self.visualize_pose(pose, successful=False, show_tf=False)

                finally:
                    loop.close()

                # Mark execution task as done
                self.execution_queue.task_done()

            except queue.Empty:
                # If there's nothing in the queue, sleep briefly
                rospy.sleep(0.1)
                continue

        rospy.loginfo("Executor worker finished")

    def sample_and_test_poses(
        self, num_poses=10, max_attempts=50, max_angle=math.pi / 6
    ):
        """
        Sample poses and test if the robot can move to them in parallel.

        Args:
            num_poses: Number of successful poses to collect
            max_attempts: Maximum number of sampling attempts
            max_angle: Maximum angle in radians from vertical (0 to pi)
        """
        rospy.loginfo(f"Sampling and testing up to {num_poses} poses from the ROI")
        rospy.loginfo(
            f"Using max deviation angle: {max_angle} radians ({math.degrees(max_angle):.1f} degrees)"
        )
        rospy.loginfo(f"Using parallel processing with queues")

        # Reset counters
        self.attempt_count = 0
        self.success_count = 0

        # First move to home position
        rospy.loginfo("Moving to home position before starting parallel workers")
        self.env.executor.move_to_home(self.env.planner)

        # Create and start worker threads
        threads = []

        # Start sampler worker
        sampler_thread = threading.Thread(
            target=self.sampler_worker, args=(max_attempts, max_angle), daemon=True
        )
        sampler_thread.start()
        threads.append(sampler_thread)

        # Start planner worker
        planner_thread = threading.Thread(target=self.planner_worker, daemon=True)
        planner_thread.start()
        threads.append(planner_thread)

        # Start executor worker
        executor_thread = threading.Thread(
            target=self.executor_worker, args=(num_poses,), daemon=True
        )
        executor_thread.start()
        threads.append(executor_thread)

        # Wait for all threads to complete or for Ctrl+C
        try:
            # Wait for the executor thread to finish (indicates we've reached our target)
            executor_thread.join()

            # Signal other threads to stop
            self.shutdown_event.set()

            # Wait for remaining threads with timeout
            for thread in threads:
                thread.join(timeout=2.0)

            # Report results
            with self.success_count_lock, self.attempt_count_lock:
                if self.success_count >= num_poses:
                    rospy.loginfo(
                        f"Successfully found and executed {self.success_count} poses in {self.attempt_count} attempts"
                    )
                else:
                    rospy.logwarn(
                        f"Only found {self.success_count}/{num_poses} successful poses after {self.attempt_count} attempts"
                    )

            # Display all successful poses with a different visualization
            rospy.loginfo(f"Displaying {len(self.successful_poses)} successful poses")
            for i, pose in enumerate(self.successful_poses):
                # Create pose text label
                text_pose = Pose(
                    Point(pose.position.x, pose.position.y, pose.position.z + 0.1),
                    Quaternion(0, 0, 0, 1),
                )
                scale = Vector3(0.05, 0.05, 0.05)
                self.markers.publishText(text_pose, f"Pose {i+1}", "white", scale, 0)

                # Highlight successful pose with larger axis
                self.markers.publishAxis(pose, 0.15, 0.015, 0)
                self.markers.publishSphere(pose, "green", 0.03, 0)

        except KeyboardInterrupt:
            rospy.loginfo("Interrupted by user, shutting down worker threads")
            self.shutdown_event.set()

            # Wait for threads to finish with timeout
            for thread in threads:
                thread.join(timeout=2.0)

        return self.success_count

    def cleanup(self):
        """Clean up resources before shutting down"""
        rospy.loginfo("Cleaning up...")
        # Signal all threads to stop
        self.shutdown_event.set()
        # Delete all markers
        self.markers.deleteAllMarkers()


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
            "--max-attempts", type=int, default=100, help="Maximum sampling attempts"
        )
        parser.add_argument(
            "--max-angle",
            type=float,
            default=math.pi / 6,
            help="Maximum angle in radians from vertical (0 to pi)",
        )

        # Parse arguments without actual sys.argv to avoid interfering with ROS
        args = parser.parse_args(rospy.myargv()[1:])

        # Create the pose sampler
        sampler = PoseSampler()

        # Check if ROI was found
        if not sampler.roi_info:
            rospy.logerr("Unable to locate robot_roi in the planning scene. Aborting.")
            rospy.loginfo(
                "Please make sure there is a box object named 'robot_roi' in your environment configuration."
            )
            return

        # Give time for publishers to connect
        rospy.sleep(1.0)

        # Sample and test poses
        sampler.sample_and_test_poses(
            num_poses=args.num_poses,
            max_attempts=args.max_attempts,
            max_angle=args.max_angle,
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
