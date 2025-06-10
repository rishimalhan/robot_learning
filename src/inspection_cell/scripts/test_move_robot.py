#!/usr/bin/env python3

# External

import rospy
import tf2_ros
from core.load_system import EnvironmentLoader
from core.executor import Executor
from core.point_cloud_processor import PointCloudProcessor
from geometry_msgs.msg import Pose
import traceback
import numpy as np
from core.utils import (
    get_robot_roi_bounds,
    init_visualization,
    visualize_waypoints,
    clear_visualization,
)

# Internal


def main():
    # Initialize ROS node
    rospy.init_node("test_robot_motion_node", anonymous=True)
    env = EnvironmentLoader()
    executor = Executor()

    # Initialize visualization
    markers, tf_broadcaster = init_visualization()

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

        # Calculate Z position with 0.4m lower than current
        z_position = current_pose.position.z - 0.4

        # Step 2: Set up zigzag pattern waypoints
        rospy.loginfo("Step 2: Setting up zigzag pattern waypoints...")

        # Define grid size
        grid_points_x = 10  # Number of points in X direction
        grid_points_y = 10  # Number of points in Y direction

        # Calculate step sizes based on ROI bounds with margins
        margin_x = 0.2
        margin_y = 0.3
        x_min = roi_bounds["x_min"] + margin_x
        x_max = roi_bounds["x_max"] - margin_x
        y_min = roi_bounds["y_min"] + margin_y
        y_max = roi_bounds["y_max"] - margin_y

        # Generate x and y coordinates using numpy
        x_coords = np.linspace(x_min, x_max, grid_points_x)
        y_coords = np.linspace(y_min, y_max, grid_points_y)

        # Generate zigzag waypoints
        waypoints = []
        for y_idx, y_pos in enumerate(y_coords):
            # Reverse x coordinates for every other row
            x_row = x_coords[::-1] if y_idx % 2 else x_coords

            for x_pos in x_row:
                pose = Pose()
                pose.position.x = x_pos
                pose.position.y = y_pos
                pose.position.z = z_position
                pose.orientation = orientation
                waypoints.append(pose)

        # Visualize all waypoints
        visualize_waypoints(
            waypoints, markers, tf_broadcaster, show_labels=True, show_axes=True
        )

        # Step 3: Execute zigzag pattern using cartesian path
        rospy.loginfo(
            f"Step 3: Executing zigzag pattern with {len(waypoints)} waypoints..."
        )

        # Plan cartesian path with smaller step size for smoother motion
        plan, fraction, planning_time = env.planner.plan_cartesian_path(
            waypoints=waypoints,
            eef_step=0.001,  # 1mm steps for smooth motion
            jump_threshold=False,  # Allow jumps for smoother path
        )

        if fraction < 0.98:
            rospy.logwarn(
                f"Could only plan {fraction:.1%} of the path. Consider adjusting waypoints."
            )
            return

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
        else:
            rospy.logerr("Failed to plan return to home")

    except rospy.ROSInterruptException:
        rospy.loginfo("Program interrupted by user")
    except Exception as e:
        rospy.logerr(f"Error during execution: {str(e)}")
        rospy.logerr(f"Exception details: {traceback.format_exc()}")
    finally:
        clear_visualization(markers)
        rospy.loginfo("Clean shutdown. Exiting.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
