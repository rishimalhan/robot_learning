#!/usr/bin/env python3

import rospy
import torch
import numpy as np
from neural_engine.srv import SampleConfig, SampleConfigResponse
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest

# Check MPS availability
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class CollisionChecker:
    def __init__(self, group_name="manipulator"):
        """Initialize the collision checker with MoveIt interfaces"""
        # Initialize MoveIt interfaces
        self.robot = moveit_commander.RobotCommander()
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        # Wait for state validity service
        rospy.loginfo("Waiting for state validity service...")
        rospy.wait_for_service("/check_state_validity")
        self.state_validity_service = rospy.ServiceProxy(
            "/check_state_validity", GetStateValidity
        )
        rospy.loginfo("State validity service connected")

        # Get joint names for the group
        self.joint_names = self.move_group.get_active_joints()
        rospy.loginfo(f"Joint names: {self.joint_names}")

    def is_state_valid(self, joint_positions):
        """Check if a set of joint positions is collision-free"""
        # Get current robot state
        robot_state = self.robot.get_current_state()

        # Update the joint positions in the robot state
        for i, joint_name in enumerate(self.joint_names):
            try:
                index = robot_state.joint_state.name.index(joint_name)
                robot_state.joint_state.position = list(
                    robot_state.joint_state.position
                )
                robot_state.joint_state.position[index] = joint_positions[i]
            except ValueError:
                rospy.logwarn(f"Joint {joint_name} not found in robot state")
                return False

        # Create request
        req = GetStateValidityRequest()
        req.robot_state = robot_state
        req.group_name = self.move_group.get_name()

        # Call service
        try:
            res = self.state_validity_service.call(req)

            # If state is not valid, log collision information
            if not res.valid:
                rospy.logdebug("Configuration is in collision:")
                for contact in res.contacts:
                    rospy.logdebug(
                        f"  Collision between {contact.contact_body_1} and {contact.contact_body_2}"
                    )
                    rospy.logdebug(f"  Collision distance: {contact.depth}")

            return res.valid
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False


# Initialize collision checker at module level
collision_checker = None


def initialize_collision_checker():
    global collision_checker
    if collision_checker is None:
        collision_checker = CollisionChecker()


def sample_configs(req):
    try:
        # Ensure collision checker is initialized
        initialize_collision_checker()

        # Convert current joints to tensor on MPS
        current_joints = torch.tensor(req.current_joints, device=DEVICE)
        num_joints = len(current_joints)

        # Generate multiple random configurations
        num_samples = (
            100  # Generate more samples to increase chance of finding valid config
        )
        max_iterations = 3  # Maximum number of batches to try

        for iteration in range(max_iterations):
            rospy.loginfo(f"Sampling iteration {iteration+1}/{max_iterations}...")

            # Generate random configurations
            random_configs = torch.rand((num_samples, num_joints), device=DEVICE)

            # Get joint limits (assuming standard joint limits for ABB IRB2400)
            min_limits = torch.tensor(
                [-3.1416, -1.7453, -1.0472, -3.49, -2.0944, -6.9813], device=DEVICE
            )
            max_limits = torch.tensor(
                [3.1416, 1.9199, 1.1345, 3.49, 2.0944, 6.9813], device=DEVICE
            )

            # Scale random values to joint limits
            random_configs = min_limits + (max_limits - min_limits) * random_configs

            # Calculate absolute differences from current position
            joint_diffs = torch.abs(random_configs - current_joints)

            # Convert 90 degrees to radians
            max_diff = torch.tensor(1.5708, device=DEVICE)  # 90 degrees in radians

            # Find configurations where all joints are within 90 degrees
            valid_configs = torch.all(joint_diffs <= max_diff, dim=1)

            if torch.any(valid_configs):
                # Get all valid configuration indices
                valid_indices = torch.where(valid_configs)[0]

                # Check each valid configuration for collisions
                for idx in valid_indices:
                    sampled_config = random_configs[idx]
                    sampled_joints = sampled_config.cpu().numpy().tolist()

                    # Check for collisions
                    rospy.loginfo(
                        f"Checking configuration for collisions: {sampled_joints}"
                    )
                    if collision_checker.is_state_valid(sampled_joints):
                        rospy.loginfo(
                            f"Found collision-free configuration: {sampled_joints}"
                        )
                        return SampleConfigResponse(
                            sampled_joints=sampled_joints, success=True
                        )
                    else:
                        rospy.logdebug("Configuration is in collision, trying next...")

                rospy.logwarn(
                    "All valid configurations are in collision, trying more samples..."
                )
            else:
                rospy.logwarn(
                    "No valid configuration found within 90 degrees in this batch"
                )

        rospy.logerr(
            "Failed to find a collision-free configuration after all iterations"
        )
        return SampleConfigResponse(sampled_joints=[], success=False)

    except Exception as e:
        rospy.logerr(f"Error in sample_configs: {str(e)}")
        import traceback

        rospy.logerr(traceback.format_exc())
        return SampleConfigResponse(sampled_joints=[], success=False)


def main():
    rospy.init_node("sample_config_service")

    # Initialize MoveIt
    moveit_commander.roscpp_initialize([])

    # Initialize collision checker
    initialize_collision_checker()

    # Create service
    s = rospy.Service("sample_config", SampleConfig, sample_configs)
    rospy.loginfo("Sample Config Service Ready")
    rospy.loginfo(f"Using device: {DEVICE}")

    # Keep the node running
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
