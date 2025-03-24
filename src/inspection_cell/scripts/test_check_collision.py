#!/usr/bin/env python3

"""
Standalone script to check for collisions in the current state of the robot.
This is useful for debugging when MoveIt planning reports collisions.

Usage:
    rosrun inspection_cell check_collision.py [move_group_name]
    
    move_group_name: Optional name of the move group to check. Default is "manipulator".
"""

import rospy
import sys
import moveit_commander
from inspection_cell.collision_checker import CollisionCheck


def main():
    # Parse arguments
    move_group_name = "manipulator"
    if len(sys.argv) > 1:
        move_group_name = sys.argv[1]

    # Initialize ROS node
    rospy.init_node("collision_check_node", anonymous=True)

    # Initialize MoveIt commander (needed for RobotCommander)
    moveit_commander.roscpp_initialize(sys.argv)

    try:
        # Create collision checker
        rospy.loginfo(f"Checking collisions for move group: {move_group_name}")
        checker = CollisionCheck(move_group_name)

        # Check current state
        is_valid = checker.check_state_validity()

        if is_valid:
            rospy.loginfo("Current robot state is collision-free.")
        else:
            rospy.logwarn("Current robot state is in collision!")
            collision_report = checker.get_collision_report()
            rospy.logwarn(collision_report)

            # Provide helpful tips
            rospy.loginfo("\nTips for resolving collisions:")
            rospy.loginfo(
                "1. Check if the robot model and environment objects have correct dimensions."
            )
            rospy.loginfo(
                "2. Verify that all collision objects are in the correct position."
            )
            rospy.loginfo("3. Make sure the robot is not in self-collision.")
            rospy.loginfo(
                "4. If using a real robot, make sure the joint states are correctly reported."
            )
            rospy.loginfo(
                "5. Check for potential issues with planning scene not matching reality."
            )

    except Exception as e:
        rospy.logerr(f"Error checking collisions: {e}")

    finally:
        # Shut down MoveIt commander
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
