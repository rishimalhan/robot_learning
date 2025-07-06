#!/usr/bin/env python3

import rospy
from core.load_system import EnvironmentLoader
from core.executor import Executor
from neural_engine.srv import GenerateViewpoints, GenerateViewpointsRequest
from core.utils import (
    init_visualization,
    visualize_waypoints,
    clear_visualization,
)


def visualize_viewpoints(viewpoints, markers, tf_broadcaster):
    """Visualize generated viewpoints"""
    if not viewpoints:
        rospy.logwarn("No viewpoints to visualize")
        return

    # Clear previous visualization
    clear_visualization(markers)

    # Visualize viewpoints
    visualize_waypoints(
        viewpoints,
        markers,
        tf_broadcaster,
        show_labels=True,
        show_axes=True,
    )

    rospy.loginfo(f"Visualized {len(viewpoints)} viewpoints")


def main():
    # Initialize ROS node
    rospy.init_node("test_set_cover_node", anonymous=True)

    # Initialize environment and executor
    env = EnvironmentLoader()
    executor = Executor()
    markers, tf_broadcaster = init_visualization()
    try:
        # Move to home position
        rospy.loginfo("Moving to home position...")
        success, plan, planning_time, _ = env.planner.plan_to_named_target("home")
        if success and plan:
            executor.execute_plan(plan)
            rospy.loginfo("Robot moved to home position")
        else:
            rospy.logerr("Failed to plan to home position")
            return

        # Wait for SetCover service
        rospy.loginfo("Waiting for SetCover service...")
        rospy.wait_for_service("generate_viewpoints", timeout=30.0)

        # Create service proxy
        generate_viewpoints = rospy.ServiceProxy(
            "generate_viewpoints", GenerateViewpoints
        )

        # Call service with parameters
        rospy.loginfo("Calling SetCover service...")
        request = GenerateViewpointsRequest()
        request.vert_angle = 30.0  # 30 degrees vertical tilt
        request.horz_angle = 45.0  # 45 degrees horizontal rotation
        request.num_samples = 10  # Generate 1000 viewpoints

        response = generate_viewpoints(request)
        visualize_viewpoints(response.viewpoints, markers, tf_broadcaster)

        if response.success:
            rospy.loginfo(f"{response.message}")
            rospy.loginfo(f"Generated {len(response.viewpoints)} viewpoints")
        else:
            rospy.logerr(f"Service failed: {response.message}")

    except rospy.ROSInterruptException:
        rospy.loginfo("Program interrupted by user")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
    finally:
        rospy.loginfo("Test completed")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
