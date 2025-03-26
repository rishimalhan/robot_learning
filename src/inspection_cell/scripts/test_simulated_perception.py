#!/usr/bin/env python3

import rospy
import sys
import time
import argparse
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler

from inspection_cell.load_system import EnvironmentLoader
from inspection_cell.simulated_perception import SimulatedPerception
from inspection_cell.planner import Planner
from inspection_cell.executor import Executor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test simulated perception system")
    parser.add_argument(
        "--move", action="store_true", help="Move the robot around to test perception"
    )
    return parser.parse_args()


def main():
    """Main function to test simulated perception."""
    # Initialize ROS node
    rospy.init_node("test_perception", anonymous=True)
    rospy.loginfo("Starting simulated perception test")

    # Parse arguments
    args = parse_arguments()

    # Load the environment
    rospy.loginfo("Loading environment...")
    env_loader = EnvironmentLoader(move_group_name="manipulator", clear_scene=True)

    # Pause to ensure environment is fully loaded
    rospy.sleep(2.0)

    # Initialize the simulated perception system
    rospy.loginfo("Initializing simulated perception...")
    perception = SimulatedPerception()

    # Pause to ensure perception system is ready
    rospy.sleep(2.0)

    # Get initial detected objects
    detected_objects = perception.get_detected_objects()
    rospy.loginfo(f"Initial detection: {len(detected_objects)} objects detected")
    for name, _ in detected_objects:
        rospy.loginfo(f"  - {name}")

    if args.move:
        # Initialize the planner and executor
        planner = Planner(move_group_name="manipulator")
        executor = Executor(move_group_name="manipulator", check_collisions=True)

        # Define some poses to test perception from different viewpoints
        rospy.loginfo("Moving robot to test perception from different viewpoints...")

        # Go to home position first
        rospy.loginfo("Moving to home position...")
        success, plan, _, _ = planner.plan_to_home()
        if success:
            executor.execute_plan(plan)
            rospy.sleep(3.0)  # Give time for perception to update

            # Get detected objects from home position
            detected_objects = perception.get_detected_objects()
            rospy.loginfo(f"From home: {len(detected_objects)} objects detected")
            for name, _ in detected_objects:
                rospy.loginfo(f"  - {name}")
        else:
            rospy.logwarn("Failed to plan to home position")

        # Define test poses around the perception ROI
        test_poses = [
            # Pose 1: Front view
            {
                "position": [0.8, 0.0, 1.0],
                "orientation": [0, -1.57, 0],  # Looking horizontally at ROI
            },
            # Pose 2: Side view
            {
                "position": [1.0, 0.5, 1.0],
                "orientation": [-1.57, 0, 0],  # Looking from the side
            },
            # Pose 3: Top view
            {"position": [1.0, 0.0, 1.3], "orientation": [0, 0, 3.14]},  # Looking down
        ]

        # Try to move to each test pose
        for i, pose_config in enumerate(test_poses):
            rospy.loginfo(f"Moving to test pose {i+1}...")

            # Create pose message
            pose = Pose()
            pose.position.x = pose_config["position"][0]
            pose.position.y = pose_config["position"][1]
            pose.position.z = pose_config["position"][2]

            # Convert RPY to quaternion
            q = quaternion_from_euler(
                pose_config["orientation"][0],
                pose_config["orientation"][1],
                pose_config["orientation"][2],
            )
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]

            # Plan and execute
            success, plan, _, _ = planner.plan_to_pose_target(pose)
            if success:
                executor.execute_plan(plan)
                rospy.sleep(3.0)  # Give time for perception to update

                # Get detected objects from this pose
                detected_objects = perception.get_detected_objects()
                rospy.loginfo(
                    f"From pose {i+1}: {len(detected_objects)} objects detected"
                )
                for name, _ in detected_objects:
                    rospy.loginfo(f"  - {name}")
            else:
                rospy.logwarn(f"Failed to plan to test pose {i+1}")

        # Return to home position
        rospy.loginfo("Returning to home position...")
        success, plan, _, _ = planner.plan_to_home()
        if success:
            executor.execute_plan(plan)

    # Keep running until shutdown
    rospy.loginfo("Simulated perception test running. Press Ctrl+C to stop.")
    rospy.spin()

    # Shutdown perception system
    perception.shutdown()
    rospy.loginfo("Test completed")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in perception test: {str(e)}")
        import traceback

        traceback.print_exc()
