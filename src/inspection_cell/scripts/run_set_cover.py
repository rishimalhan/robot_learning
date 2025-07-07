#!/usr/bin/env python3

# External

import rospy
import time

# Internal

from core.point_cloud_processor import PointCloudProcessor
from core.load_system import EnvironmentLoader
from core.executor import Executor
from neural_engine.srv import GenerateViewpoints, GenerateViewpointsRequest
from core.utils import (
    init_visualization,
    visualize_waypoints,
    clear_visualization,
)
from core.greedy_set_cover import GreedySetCover

# Global variables to store data between function calls
viewpoints_data = None
pointclouds_data = None
coverage_plan = None
env = None
executor = None
markers = None
tf_broadcaster = None
greedy_set_cover = None


def initialize_system():
    """Initialize all system components"""
    global env, executor, markers, tf_broadcaster, greedy_set_cover

    rospy.loginfo("Initializing system components...")

    # Initialize environment and executor
    env = EnvironmentLoader()
    executor = Executor()
    markers, tf_broadcaster = init_visualization()
    clear_visualization(markers)
    greedy_set_cover = GreedySetCover()

    rospy.loginfo("System initialized successfully")


def generate_viewpoints(vert_angle=30.0, horz_angle=45.0, num_samples=500):
    """
    Generate viewpoints using the service

    Args:
        vert_angle: Vertical angle in degrees
        horz_angle: Horizontal angle in degrees
        num_samples: Number of samples to generate

    Returns:
        Service response containing viewpoints and pointclouds
    """
    global viewpoints_data, pointclouds_data

    rospy.loginfo("Generating viewpoints...")

    # Wait for service
    rospy.wait_for_service("generate_viewpoints", timeout=30.0)

    # Create service proxy
    generate_viewpoints_service = rospy.ServiceProxy(
        "generate_viewpoints", GenerateViewpoints
    )

    # Call service with parameters
    request = GenerateViewpointsRequest()
    request.vert_angle = vert_angle
    request.horz_angle = horz_angle
    request.num_samples = num_samples

    response = generate_viewpoints_service(request)

    if response.success:
        viewpoints_data = response.viewpoints
        pointclouds_data = response.pointclouds
        rospy.loginfo(f"Generated {len(viewpoints_data)} viewpoints successfully")
        return response
    else:
        rospy.logerr(f"Viewpoint generation failed: {response.message}")
        return None


def compute_greedy_cover():
    """
    Compute greedy set cover from existing viewpoints

    Returns:
        List of selected viewpoints for coverage
    """
    global coverage_plan, viewpoints_data, pointclouds_data

    if viewpoints_data is None or pointclouds_data is None:
        rospy.logerr("No viewpoints data available. Call generate_viewpoints first.")
        return None

    rospy.loginfo("Computing greedy set cover...")

    coverage_plan = greedy_set_cover.plan(
        viewpoints=viewpoints_data, pointclouds=pointclouds_data
    )

    rospy.loginfo(f"Greedy cover computed: {len(coverage_plan)} viewpoints selected")
    return coverage_plan


def plan_and_execute():
    """
    Plan and execute the coverage path, then return to home

    Returns:
        True if successful, False otherwise
    """
    global coverage_plan, env, executor

    if coverage_plan is None:
        rospy.logerr("No coverage plan available. Call compute_greedy_cover first.")
        return False

    rospy.loginfo("Planning and executing coverage path...")

    success_count = 0
    total_viewpoints = len(coverage_plan)

    # Execute each viewpoint in the coverage plan
    for i, viewpoint in enumerate(coverage_plan):
        rospy.loginfo(f"Executing viewpoint {i+1}/{total_viewpoints}")

        # Plan to viewpoint
        success, plan, planning_time, _ = env.planner.plan_to_pose_target(
            env.get_end_effector_transform(viewpoint)
        )

        if success and plan:
            # Execute the plan
            executor.execute_plan(plan)
            success_count += 1
            rospy.loginfo(f"Successfully executed viewpoint {i+1}")
        else:
            rospy.logwarn(f"Failed to plan/execute viewpoint {i+1}")

    # Return to home position
    rospy.loginfo("Returning to home position...")
    success, plan, planning_time, _ = env.planner.plan_to_named_target("home")
    if success and plan:
        executor.execute_plan(plan)
        rospy.loginfo("Robot returned to home position")
    else:
        rospy.logerr("Failed to return to home position")

    rospy.loginfo(
        f"Execution completed: {success_count}/{total_viewpoints} viewpoints successful"
    )
    return success_count == total_viewpoints


def visualize_viewpoints_helper(viewpoints, path=False, title="viewpoints"):
    """Helper function to visualize viewpoints"""
    global markers, tf_broadcaster

    if not viewpoints:
        rospy.logwarn(f"No {title} to visualize")
        return

    # Clear previous visualization
    clear_visualization(markers)

    # Visualize viewpoints
    visualize_waypoints(
        viewpoints if len(viewpoints) < 100 else viewpoints[:100],
        markers,
        tf_broadcaster,
        path=path,
    )

    rospy.loginfo(f"Visualized {len(viewpoints)} {title}")


def main():
    """Main function to run the complete pipeline"""
    # Initialize ROS node
    rospy.init_node("run_set_cover_node", anonymous=True)

    try:
        # Initialize system
        initialize_system()

        # Move to home position first
        rospy.loginfo("Moving to home position...")
        success, plan, planning_time, _ = env.planner.plan_to_named_target("home")
        if success:
            if plan:
                executor.execute_plan(plan)
            rospy.loginfo("Robot moved to home position")
        else:
            rospy.logerr("Failed to move to home position")
            return

        # Step 1: Generate viewpoints
        response = generate_viewpoints(
            vert_angle=30.0, horz_angle=45.0, num_samples=1000
        )

        if response is None:
            rospy.logerr("Failed to generate viewpoints")
            return

    except rospy.ROSInterruptException:
        rospy.loginfo("Program interrupted by user")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
    finally:
        rospy.loginfo("Main execution completed")


def set_cover():
    """
    Function that uses generated viewpoints without calling service
    Visualizes viewpoints, waits 2 seconds, does cover set, plans, then ends
    """
    global viewpoints_data, pointclouds_data

    PointCloudProcessor(enable_reconstruction=True)

    if viewpoints_data is None or pointclouds_data is None:
        rospy.logerr("No viewpoints data available. Run main() first.")
        return

    rospy.loginfo("=== Using generated viewpoints ===")

    try:
        rospy.sleep(2.0)

        # Visualize existing viewpoints
        visualize_viewpoints_helper(
            viewpoints_data, path=False, title="existing viewpoints"
        )

        # Wait 2 seconds
        rospy.loginfo("Waiting 2 seconds...")
        time.sleep(1.0)

        # Compute greedy cover
        coverage_plan = compute_greedy_cover()
        if coverage_plan is None:
            rospy.logerr("Failed to compute greedy cover")
            return

        # Visualize coverage plan
        visualize_viewpoints_helper(coverage_plan, path=True, title="coverage plan")

        # Plan and execute
        success = plan_and_execute()

        if success:
            rospy.loginfo("Rerun completed successfully!")
        else:
            rospy.logwarn("Rerun completed with some failures")

    except Exception as e:
        rospy.logerr(f"Error during rerun: {e}")
    finally:
        rospy.loginfo("Rerun completed")


if __name__ == "__main__":
    try:
        main()

        # Use IPython embed instead of rospy.spin()
        rospy.loginfo("Starting IPython shell...")
        rospy.loginfo("Available functions:")
        rospy.loginfo("  - main(): Run complete pipeline")
        rospy.loginfo("  - set_cover(): Set cover with generated viewpoints")
        rospy.loginfo("  - generate_viewpoints(): Generate new viewpoints")
        rospy.loginfo("  - compute_greedy_cover(): Compute greedy cover")
        rospy.loginfo("  - plan_and_execute(): Plan and execute coverage")

        # Import IPython and embed
        try:
            from IPython import embed

            embed()
        except ImportError:
            rospy.logwarn("IPython not available, falling back to rospy.spin()")
            rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Program interrupted")
    except KeyboardInterrupt:
        rospy.loginfo("Program interrupted by user")
