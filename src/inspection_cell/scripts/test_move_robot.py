#!/usr/bin/env python3

import rospy
import sys
import time
import traceback
from inspection_cell.load_system import EnvironmentLoader
from inspection_cell.planner import Planner
from inspection_cell.executor import Executor


class RobotMover:
    def __init__(self):
        # Initialize the environment loader
        self.env_loader = self._create_environment()

        # Initialize the planner and executor
        self.planner = Planner(move_group_name="manipulator")
        self.executor = Executor(move_group_name="manipulator", check_collisions=True)

        # Get available named targets
        self.named_targets = self.planner.get_named_targets()

        rospy.loginfo("RobotMover initialized with Planner and Executor")

    def _create_environment(self):
        """Create a modified version of EnvironmentLoader that doesn't initialize a node"""

        # Create a subclass of EnvironmentLoader
        class ModifiedEnvironmentLoader(EnvironmentLoader):
            def __init__(self, move_group_name="manipulator", clear_scene=True):
                # Skip the parent class's __init__ to avoid initializing a node
                import moveit_commander
                import rospkg
                import os
                import yaml
                from inspection_cell.collision_checker import CollisionCheck

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

        # Create and return the modified environment loader
        rospy.loginfo("Loading environment...")
        return ModifiedEnvironmentLoader(
            move_group_name="manipulator", clear_scene=True
        )

    def move_to_home(self):
        """Move the robot to the home position using named target if available"""
        rospy.loginfo("Planning motion to home position...")

        # Plan to the home position (planner will try to use named target)
        success, plan, planning_time, error_code = self.planner.plan_to_home()

        if success and plan:
            rospy.loginfo(
                f"Plan to home position successful, execution time: {planning_time:.2f} seconds"
            )

            # Execute the plan
            execution_success = self.executor.execute_plan(plan, verify_safety=True)

            if execution_success:
                rospy.loginfo("Successfully moved to home position")
                return True
            else:
                rospy.logerr("Failed to execute plan to home position")
                return False
        else:
            rospy.logerr("Failed to plan to home position")
            return False

    def move_to_named_target(self, target_name):
        """Move the robot to a named target from the SRDF file"""
        if target_name not in self.named_targets:
            rospy.logerr(
                f"Named target '{target_name}' not available. Available targets: {self.named_targets}"
            )
            return False

        rospy.loginfo(f"Planning motion to named target: {target_name}")

        # Plan to the named target
        success, plan, planning_time, error_code = self.planner.plan_to_named_target(
            target_name
        )

        if success and plan:
            rospy.loginfo(
                f"Plan to '{target_name}' successful, execution time: {planning_time:.2f} seconds"
            )

            # Execute the plan
            execution_success = self.executor.execute_plan(plan, verify_safety=True)

            if execution_success:
                rospy.loginfo(f"Successfully moved to named target: {target_name}")
                return True
            else:
                rospy.logerr(f"Failed to execute plan to named target: {target_name}")
                return False
        else:
            rospy.logerr(f"Failed to plan to named target: {target_name}")
            return False

    def generate_plans(self, num_plans=5):
        """Generate and execute multiple random plans"""
        rospy.loginfo(f"Generating {num_plans} sample plans...")

        # Try to move to home position first
        self.move_to_home()

        # Get joint limits for random sampling
        robot = self.env_loader.get_robot()
        joint_names = self.planner.move_group.get_active_joints()

        joint_limits = []
        for joint_name in joint_names:
            joint = robot.get_joint(joint_name)
            joint_limits.append((joint.min_bound(), joint.max_bound()))

        # Track our progress
        successful_plans = 0
        attempts = 0
        max_attempts = num_plans * 5  # Allow multiple attempts per desired plan

        while successful_plans < num_plans and attempts < max_attempts:
            attempts += 1
            try:
                # Generate a random joint configuration within limits
                import random

                random_joints = [
                    random.uniform(min_limit, max_limit)
                    for min_limit, max_limit in joint_limits
                ]

                rospy.loginfo(
                    f"Attempt {attempts}/{max_attempts} - Planning to random configuration"
                )

                # Plan to the random configuration
                success, plan, planning_time, error_code = (
                    self.planner.plan_to_joint_target(random_joints)
                )

                if success and plan:
                    successful_plans += 1
                    rospy.loginfo(
                        f"Plan {successful_plans}/{num_plans} successful, executing..."
                    )

                    # Execute the plan
                    execution_success = self.executor.execute_plan(
                        plan, verify_safety=True
                    )

                    if execution_success:
                        rospy.loginfo(
                            f"Plan {successful_plans}/{num_plans} executed successfully"
                        )
                    else:
                        rospy.logwarn(
                            f"Plan {successful_plans}/{num_plans} execution failed"
                        )
                        successful_plans -= 1  # Don't count failed executions

                    # Short pause between plans
                    time.sleep(1.0)
                else:
                    rospy.logwarn(
                        f"Attempt {attempts}/{max_attempts} - Planning failed"
                    )

            except Exception as e:
                rospy.logerr(f"Error during planning/execution: {str(e)}")
                rospy.logerr(traceback.format_exc())

        if successful_plans < num_plans:
            rospy.logwarn(
                f"Only completed {successful_plans}/{num_plans} plans after {attempts} attempts"
            )
        else:
            rospy.loginfo(
                f"Successfully completed all {num_plans} plans in {attempts} attempts"
            )

        # Return to home position
        self.move_to_home()

    def cleanup(self):
        """Clean up resources before shutting down"""
        rospy.loginfo("Cleaning up...")
        self.env_loader.clear_scene()


def main():
    # Initialize ROS node
    rospy.init_node("test_robot_motion_node", anonymous=True)

    try:
        # Create robot mover and generate plans
        robot_mover = RobotMover()

        # Check if specific named targets are available
        if "all-zeros" in robot_mover.named_targets:
            # Move to all-zeros named target
            robot_mover.move_to_named_target("all-zeros")
        else:
            # Fall back to generic home
            robot_mover.move_to_home()

        # Generate and execute random plans
        robot_mover.generate_plans(num_plans=3)

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
