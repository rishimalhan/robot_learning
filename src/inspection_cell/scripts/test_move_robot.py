#!/usr/bin/env python3

# External

import rospy
from inspection_cell.load_system import EnvironmentLoader
from inspection_cell.executor import Executor
from inspection_cell.point_cloud_processor import PointCloudProcessor
from geometry_msgs.msg import Pose
import traceback

# Internal

from inspection_cell.utils import get_robot_roi_bounds


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
        success, plan, planning_time, _ = env.planner.plan_to_named_target("home")
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
