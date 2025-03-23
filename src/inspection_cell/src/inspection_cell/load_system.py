#!/usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs.msg
import random
import sys
from std_msgs.msg import String
from neural_engine.srv import SampleConfig

def get_sample_config(current_joints):
    """
    Get a sampled configuration within 90 degrees of the current position.
    
    Args:
        current_joints (list): Current joint positions in radians
        
    Returns:
        list: Sampled joint positions in radians, or None if sampling failed
    """
    rospy.wait_for_service('sample_config')
    
    try:
        sample_config = rospy.ServiceProxy('sample_config', SampleConfig)
        response = sample_config(current_joints)
        
        if response.success:
            return response.sampled_joints
        else:
            rospy.logwarn("Failed to sample configuration")
            return None
            
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {str(e)}")
        return None

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

    # Wait for the sample_config service to be available
    rospy.wait_for_service('sample_config')
    
    try:
        # Call the service to get a valid configuration
        new_joint_target = get_sample_config(current_joints)
        
        if new_joint_target:
            rospy.loginfo("Planning to joint target: %s", new_joint_target)
        else:
            rospy.logerr("Failed to get a valid configuration from service")
            return

        move_group.set_joint_value_target(new_joint_target)

        # plan() returns a tuple: (success, plan, planning_time, error_code)
        success, plan, planning_time, error_code = move_group.plan()
        
        if success and plan and len(plan.joint_trajectory.points) > 0:
            rospy.loginfo("Plan successful. Executing...")
            move_group.go(wait=True)
            rospy.loginfo("Motion execution complete.")
        else:
            rospy.logwarn("Planning failed. Error code: %s", error_code)

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
    finally:
        move_group.stop()
        move_group.clear_pose_targets()
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
