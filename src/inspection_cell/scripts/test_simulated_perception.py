#!/usr/bin/env python3

import rospy
import argparse
from geometry_msgs.msg import PoseStamped
from inspection_cell.load_system import EnvironmentLoader
from inspection_cell.planner import Planner
from inspection_cell.executor import Executor


def main():
    """Simple test to visualize the frustum and camera outputs."""
    # Initialize ROS node
    rospy.init_node("test_frustum_basic", anonymous=True)
    rospy.loginfo("Starting basic frustum test")

    # Load the environment using the existing EnvironmentLoader
    env_loader = EnvironmentLoader(move_group_name="manipulator", clear_scene=True)
    rospy.loginfo("Environment loaded")

    planner = Planner(move_group_name="manipulator")
    executor = Executor(move_group_name="manipulator")
    success, plan, planning_time, error_code = planner.plan_to_home()
    if success and plan:
        executor.execute_plan(plan)
    else:
        rospy.logerr("Failed to plan to home")

    success, plan, planning_time, error_code = planner.plan_to_ee_offset(
        [0.0, 0.0, 1.0], 0.2
    )
    if success and plan:
        executor.execute_plan(plan)
    else:
        rospy.logerr("Failed to plan to ee offset")

    # Now import and initialize the SimulatedPerception class
    try:
        from inspection_cell.simulated_perception import SimulatedPerception

        # Initialize the frustum detection
        rospy.loginfo("Initializing simulated perception...")
        detector = SimulatedPerception()

        # Keep running until shutdown
        rospy.loginfo("Simulated perception test running. Press Ctrl+C to stop.")
        rospy.spin()

    except ImportError as e:
        rospy.logerr(f"Failed to import SimulatedPerception: {str(e)}")
    except Exception as e:
        rospy.logerr(f"Error in simulated perception test: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
