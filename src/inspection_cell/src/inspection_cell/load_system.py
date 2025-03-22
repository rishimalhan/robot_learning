#!/usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs.msg
import random
import sys
from std_msgs.msg import String

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("test_robot_motion_node", anonymous=True)

    # Initialize robot commander and group
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    # Use the name of your MoveIt planning group here
    group_name = "manipulator"  # Replace with your planning group
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Get current joint values
    current_joints = move_group.get_current_joint_values()
    rospy.loginfo("Current joint values: %s", current_joints)

    # Get joint limits from the robot model
    joint_names = move_group.get_active_joints()
    new_joint_target = current_joints[:]

    for i, joint_name in enumerate(joint_names):
        joint = robot.get_joint(joint_name)
        if joint:
            min_pos = joint.min_bound()
            max_pos = joint.max_bound()
            new_joint_target[i] = random.uniform(min_pos, max_pos)
            rospy.loginfo("%s %.4f %.4f %d", joint_name, min_pos, max_pos, 1)

    rospy.loginfo("Planning to joint target: %s", new_joint_target)

    move_group.set_joint_value_target(new_joint_target)

    # plan() returns a tuple: (success, plan, planning_time, error_code)
    success, plan, planning_time, error_code = move_group.plan()
    
    if success and plan and len(plan.joint_trajectory.points) > 0:
        rospy.loginfo("Plan successful. Executing...")
        move_group.go(wait=True)
        rospy.loginfo("Motion execution complete.")
    else:
        rospy.logwarn("Planning failed. Error code: %s", error_code)

    move_group.stop()
    move_group.clear_pose_targets()
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
