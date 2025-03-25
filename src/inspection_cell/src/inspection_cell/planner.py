#!/usr/bin/env python3

import rospy
import numpy as np
from moveit_commander import MoveGroupCommander, RobotCommander
from inspection_cell.error_codes import get_error_code_name, get_error_description


class Planner:
    """
    Motion planning functionality for robot manipulation tasks.

    This class handles the planning of robot motions without executing them,
    which is the responsibility of the Executor class.
    """

    def __init__(self, move_group_name="manipulator"):
        """
        Initialize the planner with a specific move group.

        Args:
            move_group_name: Name of the MoveIt move group to use
        """
        self.robot = RobotCommander()
        self.move_group = MoveGroupCommander(move_group_name)
        self.group_name = move_group_name

        # Set some default planning parameters
        self.move_group.set_planning_time(5.0)  # seconds
        self.move_group.set_num_planning_attempts(10)
        self.move_group.set_max_velocity_scaling_factor(0.5)  # 50% of max velocity
        self.move_group.set_max_acceleration_scaling_factor(
            0.3
        )  # 30% of max acceleration

        rospy.loginfo(f"Planner initialized for group: {move_group_name}")
        rospy.loginfo(f"Planning frame: {self.move_group.get_planning_frame()}")
        rospy.loginfo(f"End effector link: {self.move_group.get_end_effector_link()}")

    def set_planning_parameters(
        self,
        planning_time=None,
        num_attempts=None,
        velocity_factor=None,
        acceleration_factor=None,
    ):
        """
        Set planning parameters.

        Args:
            planning_time: Planning time in seconds
            num_attempts: Number of planning attempts
            velocity_factor: Velocity scaling factor (0.0 to 1.0)
            acceleration_factor: Acceleration scaling factor (0.0 to 1.0)
        """
        if planning_time is not None:
            self.move_group.set_planning_time(planning_time)
        if num_attempts is not None:
            self.move_group.set_num_planning_attempts(num_attempts)
        if velocity_factor is not None:
            self.move_group.set_max_velocity_scaling_factor(velocity_factor)
        if acceleration_factor is not None:
            self.move_group.set_max_acceleration_scaling_factor(acceleration_factor)

    def get_current_joint_values(self):
        """
        Get the current joint values of the robot.

        Returns:
            list: Current joint positions in radians
        """
        return self.move_group.get_current_joint_values()

    def get_named_targets(self):
        """
        Get a list of all available named targets defined in the SRDF file.

        Returns:
            list: List of named target strings
        """
        named_targets = self.move_group.get_named_targets()
        rospy.loginfo(f"Available named targets: {named_targets}")
        return named_targets

    def plan_to_named_target(self, target_name):
        """
        Plan a motion to a named target position defined in the SRDF.

        Args:
            target_name: Name of the predefined target position

        Returns:
            tuple: (success, plan, planning_time, error_code)
        """
        if target_name not in self.move_group.get_named_targets():
            rospy.logerr(
                f"Named target '{target_name}' is not defined in the SRDF file"
            )
            rospy.loginfo(f"Available targets: {self.move_group.get_named_targets()}")
            return False, None, 0, 0

        rospy.loginfo(f"Planning to named target: {target_name}")

        # Set the named target
        self.move_group.set_named_target(target_name)

        # Plan to the target
        success, plan, planning_time, error_code = self.move_group.plan()

        # Log the planning result
        if success and plan:
            waypoints = len(plan.joint_trajectory.points)
            rospy.loginfo(
                f"Planning to '{target_name}' succeeded with {waypoints} waypoints in {planning_time:.2f} seconds"
            )
        else:
            error_name = get_error_code_name(error_code)
            error_desc = get_error_description(error_code)
            rospy.logwarn(
                f"Planning to '{target_name}' failed with error: {error_name} - {error_desc}"
            )

        return success, plan, planning_time, error_code

    def plan_to_home(self):
        """
        Plan a motion to the home position.
        First tries using the 'all-zeros' named target if available,
        otherwise falls back to a predefined joint position.

        Returns:
            tuple: (success, plan, planning_time, error_code)
        """
        rospy.loginfo("Planning to home position...")

        # Try to use the named target first
        named_targets = self.move_group.get_named_targets()

        if "all-zeros" in named_targets:
            rospy.loginfo("Using 'all-zeros' named target from SRDF")
            return self.plan_to_named_target("all-zeros")
        else:
            rospy.loginfo(
                "Named target 'all-zeros' not found, using hardcoded home position"
            )
            # Define a safe home position slightly above the work surface
            home_position = [0, -0.5, 0, 0, 0.5, 0]  # Adjust as needed
            return self.plan_to_joint_target(home_position)

    def plan_to_joint_target(self, joint_positions):
        """
        Plan a motion to the specified joint positions.

        Args:
            joint_positions: List of joint positions in radians

        Returns:
            tuple: (success, plan, planning_time, error_code)
        """
        rospy.loginfo(f"Planning to joint positions: {joint_positions}")

        # Set the target position
        self.move_group.set_joint_value_target(joint_positions)

        # Plan to the target
        success, plan, planning_time, error_code = self.move_group.plan()

        # Log the planning result
        if success and plan:
            waypoints = len(plan.joint_trajectory.points)
            rospy.loginfo(
                f"Planning succeeded with {waypoints} waypoints in {planning_time:.2f} seconds"
            )
        else:
            error_name = get_error_code_name(error_code)
            error_desc = get_error_description(error_code)
            rospy.logwarn(f"Planning failed with error: {error_name} - {error_desc}")

        return success, plan, planning_time, error_code

    def plan_to_pose_target(self, pose):
        """
        Plan a motion to the specified pose in Cartesian space.

        Args:
            pose: geometry_msgs/Pose target

        Returns:
            tuple: (success, plan, planning_time, error_code)
        """
        rospy.loginfo("Planning to pose target")

        # Set the target pose
        self.move_group.set_pose_target(pose)

        # Plan to the target
        success, plan, planning_time, error_code = self.move_group.plan()

        # Log the planning result
        if success and plan:
            waypoints = len(plan.joint_trajectory.points)
            rospy.loginfo(
                f"Planning succeeded with {waypoints} waypoints in {planning_time:.2f} seconds"
            )
        else:
            error_name = get_error_code_name(error_code)
            error_desc = get_error_description(error_code)
            rospy.logwarn(f"Planning failed with error: {error_name} - {error_desc}")

        return success, plan, planning_time, error_code

    def plan_cartesian_path(self, waypoints, eef_step=0.01, jump_threshold=0.0):
        """
        Plan a Cartesian path through the specified waypoints.

        Args:
            waypoints: List of geometry_msgs/Pose waypoints
            eef_step: Step size for the end effector (meters)
            jump_threshold: Jump threshold for joint space discontinuities

        Returns:
            tuple: (plan, fraction, planning_time)
        """
        rospy.loginfo(f"Planning Cartesian path with {len(waypoints)} waypoints")

        # Plan the Cartesian path
        start_time = rospy.Time.now()
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints, eef_step, jump_threshold
        )
        planning_time = (rospy.Time.now() - start_time).to_sec()

        # Log the planning result
        if fraction > 0.98:  # Consider 98% or better as success
            rospy.loginfo(
                f"Cartesian planning succeeded with {fraction:.2%} completion in {planning_time:.2f} seconds"
            )
        else:
            rospy.logwarn(f"Cartesian planning only achieved {fraction:.2%} completion")

        return plan, fraction, planning_time


if __name__ == "__main__":
    # Simple test if this file is run directly
    rospy.init_node("planner_test", anonymous=True)
    planner = Planner()
    success, plan, planning_time, error_code = planner.plan_to_home()

    if success:
        rospy.loginfo("Planning to home position succeeded!")
    else:
        rospy.logerr("Planning to home position failed!")

    rospy.spin()
