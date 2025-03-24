#! /usr/bin/env python3

# External

import rospy
import moveit_commander
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from moveit_msgs.msg import RobotState


class CollisionCheck:
    """
    Optimized utility class for checking collisions between the robot and environment
    """

    def __init__(self, move_group_name="manipulator"):
        # Get robot commander and the move group
        self.robot = moveit_commander.RobotCommander()
        self.group_name = move_group_name
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        # Cache joint names to avoid repeated calls
        self.joint_names = self.move_group.get_active_joints()

        # Keep track of contacts for more detailed reporting
        self.last_contacts = []

        # Wait for state validity service during initialization
        rospy.loginfo("Waiting for state validity service...")
        rospy.wait_for_service("/check_state_validity")
        self.state_validity_service = rospy.ServiceProxy(
            "/check_state_validity", GetStateValidity
        )
        rospy.loginfo("State validity service connected")

    def check_state_validity(self, joint_values=None):
        """
        Efficiently check if a robot state is valid (collision-free).

        Args:
            joint_values: Optional list of joint values to check. If None, the current state is checked.

        Returns:
            bool: True if state is valid, False otherwise.
        """
        # Clear previous contacts
        self.last_contacts = []

        if joint_values is None:
            # Use current state directly - most efficient path
            robot_state = self.robot.get_current_state()
        else:
            # Validate joint values length
            if len(self.joint_names) != len(joint_values):
                rospy.logerr(
                    f"Joint count mismatch: {len(self.joint_names)} names vs {len(joint_values)} values"
                )
                return False

            # Create robot state with provided joint values
            robot_state = RobotState()
            robot_state.joint_state.header.stamp = rospy.Time.now()
            robot_state.joint_state.name = self.joint_names
            robot_state.joint_state.position = joint_values

        # Create request - reuse the same request structure
        req = GetStateValidityRequest()
        req.robot_state = robot_state
        req.group_name = self.group_name

        # Call service
        try:
            res = self.state_validity_service.call(req)

            # Store contacts for future reference
            is_valid = res.valid

            # Check if response has contacts and store them if present
            if not is_valid and hasattr(res, "contacts") and res.contacts:
                self.last_contacts = res.contacts
                self._log_collision_details()

            return is_valid
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def _log_collision_details(self):
        """Log detailed collision information to ROS log."""
        if not self.last_contacts:
            return

        rospy.logwarn("State is in collision!")
        rospy.logwarn("Collision Details:")

        for contact in self.last_contacts:
            # Use correct attribute names
            body1 = (
                contact.contact_body_1
                if hasattr(contact, "contact_body_1")
                else contact.body_name_1
            )
            body2 = (
                contact.contact_body_2
                if hasattr(contact, "contact_body_2")
                else contact.body_name_2
            )

            rospy.logwarn(f"- Collision between '{body1}' and '{body2}':")
            if hasattr(contact, "position"):
                rospy.logwarn(f"  * Contact position: {contact.position}")
            if hasattr(contact, "depth"):
                rospy.logwarn(f"  * Penetration depth: {contact.depth:.4f}")
            if hasattr(contact, "normal"):
                rospy.logwarn(f"  * Normal: {contact.normal}")

    def get_collision_report(self):
        """
        Get a detailed report of collisions from the last check.

        Returns:
            str: A formatted report of collisions.
        """
        if not self.last_contacts:
            return "No collisions detected."

        report = ["Collision Details:"]

        # Group contacts by collision pair to avoid redundancy
        collision_pairs = {}

        for contact in self.last_contacts:
            # Use correct attribute names
            body1 = (
                contact.contact_body_1
                if hasattr(contact, "contact_body_1")
                else contact.body_name_1
            )
            body2 = (
                contact.contact_body_2
                if hasattr(contact, "contact_body_2")
                else contact.body_name_2
            )
            depth = contact.depth if hasattr(contact, "depth") else 0.0

            pair = (body1, body2)

            if pair in collision_pairs:
                collision_pairs[pair].append(depth)
            else:
                collision_pairs[pair] = [depth]

        # Report each collision pair
        for (body1, body2), depths in collision_pairs.items():
            max_depth = max(depths)
            avg_depth = sum(depths) / len(depths)
            num_points = len(depths)

            report.append(f"- Collision between '{body1}' and '{body2}':")
            report.append(f"  * Contact points: {num_points}")
            report.append(f"  * Max penetration depth: {max_depth:.4f}")
            report.append(f"  * Avg penetration depth: {avg_depth:.4f}")

        return "\n".join(report)


def check_current_state_collisions(move_group_name="manipulator"):
    """
    Check and report collisions for the current robot state.

    Args:
        move_group_name: The name of the move group to check

    Returns:
        tuple: (is_valid, collision_report)
    """
    checker = CollisionCheck(move_group_name)
    is_valid = checker.check_state_validity()

    collision_report = ""
    if not is_valid:
        collision_report = checker.get_collision_report()

    return is_valid, collision_report
