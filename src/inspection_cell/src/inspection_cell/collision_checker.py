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

        # Log collision bodies
        self.log_collision_bodies()

    def log_collision_bodies(self):
        """Log collision bodies in the planning scene with clear categorization"""
        rospy.loginfo("=============== COLLISION CONFIGURATION ===============")

        # Get all collision objects in the world
        try:
            # Create planning scene interface to get objects
            scene = moveit_commander.PlanningSceneInterface()
            rospy.sleep(0.5)  # Allow interface to connect

            # Get objects and attached objects
            world_objects = scene.get_objects()
            attached_objects = scene.get_attached_objects()

            # Get disabled collision objects from environment configuration
            disabled_objects = []
            try:
                env_config = rospy.get_param("/environment")
                for name, config in env_config.items():
                    if name in world_objects and not config.get(
                        "collision_enabled", True
                    ):
                        disabled_objects.append(name)
            except Exception as e:
                rospy.logwarn(f"Error getting disabled objects from config: {e}")

            # Get active objects (those not in disabled list)
            active_objects = [
                name for name in world_objects if name not in disabled_objects
            ]

            # Get robot links
            robot_links = self.robot.get_link_names()

            # Print a neat summary
            rospy.loginfo(f"Robot: {len(robot_links)} links")

            rospy.loginfo(f"Active Collision Objects ({len(active_objects)}):")
            for obj in active_objects:
                rospy.loginfo(f"  • {obj}")

            rospy.loginfo(f"Disabled Collision Objects ({len(disabled_objects)}):")
            for obj in disabled_objects:
                rospy.loginfo(f"  • {obj}")

            rospy.loginfo(f"Attached Objects ({len(attached_objects)}):")
            for obj in attached_objects:
                rospy.loginfo(f"  • {obj}")

            rospy.loginfo("=====================================================")

        except Exception as e:
            rospy.logwarn(f"Error getting collision objects: {e}")

    def _get_current_acm(self):
        """Get the current Allowed Collision Matrix from the planning scene."""
        try:
            rospy.wait_for_service("/get_planning_scene", timeout=1.0)
            from moveit_msgs.srv import GetPlanningScene
            from moveit_msgs.msg import PlanningSceneComponents

            get_planning_scene = rospy.ServiceProxy(
                "/get_planning_scene", GetPlanningScene
            )

            # Create proper PlanningSceneComponents message
            components = PlanningSceneComponents()
            components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX

            # Call with the proper message
            response = get_planning_scene(components)
            return response.scene.allowed_collision_matrix
        except Exception as e:
            rospy.logwarn(f"Failed to get ACM: {e}")
            return None

    def check_collision(self, joint_values=None):
        """
        Check if a robot state is in collision with the environment.
        This is a compatibility method that returns results in the same format
        as the original method from EnvironmentLoader.

        Args:
            joint_values: Optional list of joint values to check. If None, the current state is checked.

        Returns:
            tuple: (is_valid, contacts) where is_valid is a boolean and contacts is a list of collision contacts
        """
        # Use the state validity checker
        is_valid = self.check_state_validity(joint_values)

        # Return the result and contacts in the expected format
        return is_valid, self.last_contacts

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
