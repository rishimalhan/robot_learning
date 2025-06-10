#!/usr/bin/env python3

# External
import rospy
import tf2_ros
from core.load_system import EnvironmentLoader
from core.executor import Executor
from geometry_msgs.msg import Pose
from copy import deepcopy
import numpy as np
from tf.transformations import quaternion_from_euler
from core.utils import (
    create_approach_pose,
    init_visualization,
    visualize_grasp,
    clear_visualization,
)


def main():
    # Initialize ROS node
    rospy.init_node("test_picking_node", anonymous=True)
    env = EnvironmentLoader()
    executor = Executor()

    # Initialize visualization
    markers, tf_broadcaster = init_visualization()

    try:
        # Move to home
        success, plan, _, _ = env.planner.plan_to_named_target("home")
        if not success:
            rospy.logerr("Failed to plan to home")
            return
        executor.execute_plan(plan)

        # Get grasp poses from param server
        shelving_grasps = rospy.get_param("/environment/grasps/shelving/poses")
        tote_grasp = rospy.get_param("/environment/grasps/tote/poses")[
            0
        ]  # Using first tote grasp

        # Try each shelving grasp
        for grasp_idx, grasp in enumerate(shelving_grasps):
            try:
                rospy.loginfo(
                    f"\nAttempting grasp {grasp_idx + 1} of {len(shelving_grasps)}"
                )

                # Create grasp pose
                grasp_pose = Pose()
                grasp_pose.position.x = grasp["position"][0]
                grasp_pose.position.y = grasp["position"][1]
                grasp_pose.position.z = grasp["position"][2]

                # Convert euler angles to quaternion
                q = quaternion_from_euler(
                    grasp["orientation"][0],
                    grasp["orientation"][1],
                    grasp["orientation"][2],
                )
                grasp_pose.orientation.x = q[0]
                grasp_pose.orientation.y = q[1]
                grasp_pose.orientation.z = q[2]
                grasp_pose.orientation.w = q[3]

                # Create approach pose 10cm back along gripper Z
                approach_pose = create_approach_pose(grasp_pose, 0.1)

                # Visualize grasp and approach poses
                visualize_grasp(grasp_pose, approach_pose, markers, tf_broadcaster)

                # Execute pick sequence
                # First plan to approach pose (point-to-point)
                success, plan, _, _ = env.planner.plan_to_pose_target(approach_pose)
                if not success:
                    rospy.logwarn(
                        f"Failed to plan to approach pose for grasp {grasp_idx + 1}"
                    )
                    continue

                executor.execute_plan(plan)

                # Then do straight-line motion to grasp
                success, plan, _, _ = env.planner.plan_to_ee_offset(
                    direction=[0, 0, 1],  # Move forward in Z
                    distance=0.2,  # 20cm forward
                    eef_step=0.01,  # 1cm steps
                )
                if not success:
                    rospy.logwarn(
                        f"Failed to plan straight-line to grasp for grasp {grasp_idx + 1}"
                    )
                    continue

                executor.execute_plan(plan)

                rospy.sleep(0.2)  # Pause for grasp

                # Straight-line retreat
                success, plan, _, _ = env.planner.plan_to_ee_offset(
                    direction=[0, 0, -1],  # Move back in Z
                    distance=0.2,  # 20cm back
                    eef_step=0.01,  # 1cm steps
                )
                if not success:
                    rospy.logwarn(
                        f"Failed to plan straight-line retreat for grasp {grasp_idx + 1}"
                    )
                    continue

                executor.execute_plan(plan)

                # Create place pose
                place_pose = Pose()
                place_pose.position.x = tote_grasp["position"][0]
                place_pose.position.y = tote_grasp["position"][1]
                place_pose.position.z = tote_grasp["position"][2]

                # Convert euler angles to quaternion for place pose
                q = quaternion_from_euler(
                    tote_grasp["orientation"][0],
                    tote_grasp["orientation"][1],
                    tote_grasp["orientation"][2],
                )
                place_pose.orientation.x = q[0]
                place_pose.orientation.y = q[1]
                place_pose.orientation.z = q[2]
                place_pose.orientation.w = q[3]

                # Create place approach 10cm back along gripper Z
                place_approach = create_approach_pose(place_pose, 0.1)

                # Visualize place and approach poses
                visualize_grasp(place_pose, place_approach, markers, tf_broadcaster)

                # Execute place sequence
                # First plan to place approach (point-to-point)
                success, plan, _, _ = env.planner.plan_to_pose_target(place_approach)
                if not success:
                    rospy.logwarn(
                        f"Failed to plan to place approach for grasp {grasp_idx + 1}"
                    )
                    continue

                executor.execute_plan(plan)

                # Straight-line to place
                success, plan, _, _ = env.planner.plan_to_ee_offset(
                    direction=[0, 0, 1],  # Move forward in Z
                    distance=0.2,  # 20cm forward
                    eef_step=0.01,  # 1cm steps
                )
                if not success:
                    rospy.logwarn(
                        f"Failed to plan straight-line to place for grasp {grasp_idx + 1}"
                    )
                    continue

                executor.execute_plan(plan)

                rospy.sleep(0.2)  # Pause for release

                # Straight-line retreat from place
                success, plan, _, _ = env.planner.plan_to_ee_offset(
                    direction=[0, 0, -1],  # Move back in Z
                    distance=0.2,  # 20cm back
                    eef_step=0.01,  # 1cm steps
                )
                if not success:
                    rospy.logwarn(
                        f"Failed to plan straight-line retreat from place for grasp {grasp_idx + 1}"
                    )
                    continue

                executor.execute_plan(plan)

                # Return home after successful pick and place
                success, plan, _, _ = env.planner.plan_to_named_target("home")
                if not success:
                    rospy.logwarn(
                        f"Failed to plan return to home after grasp {grasp_idx + 1}"
                    )
                    continue

                executor.execute_plan(plan)
                rospy.loginfo(
                    f"Successfully completed pick and place for grasp {grasp_idx + 1}"
                )

            except Exception as e:
                rospy.logwarn(
                    f"Error during pick and place for grasp {grasp_idx + 1}: {str(e)}"
                )
                # Try to return to home position if there was an error
                try:
                    success, plan, _, _ = env.planner.plan_to_named_target("home")
                    if success:
                        executor.execute_plan(plan)
                except Exception as home_error:
                    rospy.logerr(
                        f"Failed to return home after error: {str(home_error)}"
                    )
                continue
            break

    except Exception as e:
        rospy.logerr(f"Error during pick and place sequence: {str(e)}")
    finally:
        # Final return to home
        try:
            success, plan, _, _ = env.planner.plan_to_named_target("home")
            if success:
                executor.execute_plan(plan)
        except Exception as e:
            rospy.logerr(f"Failed final return to home: {str(e)}")

        clear_visualization(markers)


if __name__ == "__main__":
    main()
