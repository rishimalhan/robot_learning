#!/usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs.msg
import random
import sys
import time
import traceback
from std_msgs.msg import String
from neural_engine.srv import SampleConfig
from inspection_cell.load_system import EnvironmentLoader
from inspection_cell.collision_checker import CollisionCheck
from inspection_cell.error_codes import get_error_code_name, get_error_description
import rospkg

HOME = [0, 0.0, 0, 0, 0.0, 0]


class RobotMover:
    def __init__(self):
        # Initialize MoveIt commander
        moveit_commander.roscpp_initialize(sys.argv)

        # Create a modified EnvironmentLoader that doesn't initialize the node
        self.env_loader = self._create_environment()

        # Get the move group from the environment loader
        self.move_group = self.env_loader.get_move_group()

        # Display information about the robot
        self.robot = moveit_commander.RobotCommander()
        rospy.loginfo(
            "Planning reference frame: %s", self.move_group.get_planning_frame()
        )
        rospy.loginfo("End effector link: %s", self.move_group.get_end_effector_link())
        rospy.loginfo(
            "Available planning groups: %s", ", ".join(self.robot.get_group_names())
        )

    def _create_environment(self):
        """Create a modified version of EnvironmentLoader that doesn't initialize a node"""

        # Create a subclass of EnvironmentLoader
        class ModifiedEnvironmentLoader(EnvironmentLoader):
            def __init__(self, move_group_name="manipulator", clear_scene=True):
                # Skip the parent class's __init__ to avoid initializing a node
                # Initialize attributes directly
                self.scene = moveit_commander.PlanningSceneInterface()
                self.move_group = moveit_commander.MoveGroupCommander(move_group_name)
                self.robot = moveit_commander.RobotCommander()
                self.group_name = move_group_name

                # Create RosPack instance
                self.rospack = rospkg.RosPack()

                # Get the planning frame
                self.planning_frame = self.move_group.get_planning_frame()
                rospy.loginfo(f"Planning frame: {self.planning_frame}")

                # Clear the planning scene if requested
                if clear_scene:
                    self.clear_scene()

                # Load environment configuration using rospkg
                import os
                import yaml

                config_path = os.path.join(
                    self.rospack.get_path("inspection_cell"),
                    "config",
                    "environment.yaml",
                )
                rospy.loginfo(f"Loading configuration from: {config_path}")
                with open(config_path, "r") as f:
                    self.config = yaml.safe_load(f)

                # Configure robot parameters
                self._configure_robot()

                # Add objects to scene
                self._add_objects_to_scene()

                # Wait for scene to update
                rospy.sleep(1.0)

                # Print current scene objects
                self._print_scene_objects()

                # Initialize the collision checker AFTER environment is fully loaded
                rospy.loginfo(
                    "Initializing collision checker after environment setup..."
                )
                self.collision_checker = CollisionCheck(move_group_name=move_group_name)
                rospy.loginfo("Collision checker initialized")

            def check_collision(self, joint_positions=None, group_name=None):
                """Check if a robot state is in collision using the collision checker.

                Args:
                    joint_positions: List of joint positions to check. If None, the current robot state is used.
                    group_name: Name of the move group to check. Ignored if using the optimized checker.

                Returns:
                    tuple: (is_valid, contacts) where is_valid is a boolean and contacts is a list of collision contacts
                """
                # Use the collision checker
                is_valid = self.collision_checker.check_state_validity(joint_positions)

                # Return the result and contacts
                return is_valid, self.collision_checker.last_contacts

        # Create and return the modified environment loader
        rospy.loginfo("Loading environment...")
        return ModifiedEnvironmentLoader(
            move_group_name="manipulator", clear_scene=True
        )

    def check_collision(self, joints=None):
        """Check if the robot is in collision at the specified joint values"""
        # Use the collision checker from the environment loader
        is_valid, contacts = self.env_loader.check_collision(joints)

        # Get collision report if in collision
        if not is_valid:
            collision_report = self.env_loader.collision_checker.get_collision_report()
            rospy.logwarn("Robot state is in collision!")
            rospy.logwarn(collision_report)
        else:
            rospy.loginfo("Robot state is collision-free")

        return is_valid

    def generate_plans(self, num_plans=20):
        """Generate and execute multiple plans"""
        # Check current state for collisions
        rospy.loginfo("Checking current state for collisions...")
        self.env_loader.check_collision(HOME)

        move_group = self.env_loader.get_move_group()
        joint_names = move_group.get_active_joints()
        robot = self.env_loader.get_robot()

        joint_limits = []
        for joint_name in joint_names:
            joint = robot.get_joint(joint_name)
            joint_limits.append((joint.min_bound(), joint.max_bound()))

        # Get current joint values
        current_joints = self.move_group.get_current_joint_values()
        rospy.loginfo("Current joint values: %s", current_joints)

        # Generate multiple plans
        rospy.loginfo(f"Generating {num_plans} sample plans...")
        successful_plans = 0
        attempts = 0
        max_attempts = (
            num_plans * 2
        )  # Try more times than needed to get enough successful plans

        while successful_plans < num_plans and attempts < max_attempts:
            attempts += 1
            try:
                # Generate a random joint configuration within joint limits
                new_joint_target = [
                    random.uniform(min_limit, max_limit)
                    for min_limit, max_limit in joint_limits
                ]

                rospy.loginfo(
                    f"Attempt {attempts}/{max_attempts} - Planning to joint target: %s",
                    new_joint_target,
                )
                self.move_group.set_joint_value_target(new_joint_target)

                # Create a plan
                rospy.loginfo(f"Attempt {attempts}/{max_attempts} - Planning...")
                plan_success, plan, planning_time, error_code = self.move_group.plan()

                if plan_success and plan and len(plan.joint_trajectory.points) > 0:
                    successful_plans += 1
                    rospy.loginfo(
                        f"Plan {successful_plans}/{num_plans} - Planning successful with {len(plan.joint_trajectory.points)} waypoints."
                    )

                    # Execute the plan
                    rospy.loginfo(f"Plan {successful_plans}/{num_plans} - Executing...")
                    self.move_group.execute(plan, wait=True)
                    rospy.loginfo(
                        f"Plan {successful_plans}/{num_plans} - Motion execution complete."
                    )

                    # Update current joints for next sample
                    current_joints = self.move_group.get_current_joint_values()

                    # Wait a short time between plans
                    time.sleep(1.0)
                else:
                    # Get and log the error description using our new error codes module
                    error_name = get_error_code_name(error_code)
                    error_description = get_error_description(error_code)
                    rospy.logwarn(
                        f"Attempt {attempts}/{max_attempts} - Planning failed. Error: {error_name} - {error_description}"
                    )

            except rospy.ServiceException as e:
                rospy.logerr(
                    f"Attempt {attempts}/{max_attempts} - Service call failed: {e}"
                )
                rospy.logerr(traceback.format_exc())

            except Exception as e:
                rospy.logerr(f"Attempt {attempts}/{max_attempts} - Error: {e}")
                rospy.logerr(traceback.format_exc())

            finally:
                self.move_group.stop()
                self.move_group.clear_pose_targets()

        if successful_plans < num_plans:
            rospy.logwarn(
                f"Could only complete {successful_plans}/{num_plans} plans after {attempts} attempts"
            )

    def cleanup(self):
        """Clean up resources before shutting down"""
        rospy.loginfo("Cleaning up...")
        self.env_loader.clear_scene()
        moveit_commander.roscpp_shutdown()


def main():
    # Initialize ROS node
    rospy.init_node("test_robot_motion_node", anonymous=True)

    try:
        # Create robot mover and generate plans
        robot_mover = RobotMover()
        robot_mover.generate_plans(num_plans=5)

    except Exception as e:
        rospy.logerr(f"Error in main: {e}")
        rospy.logerr(traceback.format_exc())
    finally:
        if "robot_mover" in locals():
            robot_mover.cleanup()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception: {e}")
        rospy.logerr(traceback.format_exc())
