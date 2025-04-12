#!/usr/bin/env python3

# External

import rospy
import numpy as np
from moveit_commander import MoveGroupCommander, RobotCommander
import ros_numpy
from typing import List, Dict, Optional, Tuple, Callable, Any, Union
from functools import partial

# Internal

from inspection_cell.error_codes import get_error_code_name, get_error_description
from inspection_cell.stats_reporter import report_planning_stats, stats_reporter


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
        self.move_group.set_planning_time(1.0)  # seconds
        self.move_group.set_num_planning_attempts(1)
        self.move_group.set_max_velocity_scaling_factor(1.0)
        self.move_group.set_max_acceleration_scaling_factor(1.0)

        # Define available planners for different types of motion
        self.point_to_point_planners = [
            "RRTkConfigDefault",
            "RRTConnectkConfigDefault",
            "RRTstarkConfigDefault",
        ]
        self.cartesian_planners = [
            "TRRTkConfigDefault",
            "ESTkConfigDefault",
        ]

        rospy.loginfo(f"Planner initialized for group: {move_group_name}")
        rospy.loginfo(f"Planning frame: {self.move_group.get_planning_frame()}")
        rospy.loginfo(f"End effector link: {self.move_group.get_end_effector_link()}")

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

    def plan_to_home(self, planner_id: str = None):
        """
        Plan a motion to the home position.
        First tries using the 'home' named target if available,
        otherwise falls back to a predefined joint position.

        Args:
            planner_id: If specified, use this single planner. If None, try multiple planners in parallel.

        Returns:
            tuple: (success, plan, planning_time, error_code)
        """
        rospy.loginfo("Planning to home position...")
        return self.plan_to_named_target("home", planner_id)

    @report_planning_stats
    def _execute_planning_task(
        self, planning_func: Callable, planner_id: str, *args, **kwargs
    ) -> Tuple[bool, Optional[object], float, int]:
        """
        Execute a planning task with a specific planner.

        Args:
            planning_func: The planning function to execute
            planner_id: The planner to use
            *args: Positional arguments for the planning function
            **kwargs: Keyword arguments for the planning function

        Returns:
            tuple: (success, plan, planning_time, error_code)
        """
        try:
            self.move_group.set_planner_id(planner_id)
            return planning_func(*args, **kwargs)
        except Exception as e:
            rospy.logerr(f"Error in planning with {planner_id}: {str(e)}")
            return False, None, 0, 0

    def _execute_cartesian_task(
        self, planning_func: Callable, planner_id: str, *args, **kwargs
    ) -> Tuple[object, float, float]:
        """
        Execute a cartesian planning task with a specific planner.

        Args:
            planning_func: The planning function to execute
            planner_id: The planner to use
            *args: Positional arguments for the planning function
            **kwargs: Keyword arguments for the planning function

        Returns:
            tuple: (plan, fraction, planning_time)
        """
        try:
            self.move_group.set_planner_id(planner_id)
            start_time = rospy.Time.now()
            result = planning_func(*args, **kwargs)
            planning_time = (rospy.Time.now() - start_time).to_sec()

            if isinstance(result, tuple) and len(result) == 2:
                plan, fraction = result
                if fraction > 0.9:
                    rospy.loginfo(
                        f"Cartesian plan found with {planner_id} (fraction: {fraction})"
                    )
                return plan, fraction, planning_time
            return None, 0.0, planning_time
        except Exception as e:
            rospy.logerr(f"Error in cartesian planning with {planner_id}: {str(e)}")
            return None, 0.0, 0.0

    def _run_sequential_planning(
        self,
        planners: List[str],
        planning_func: Callable,
        is_cartesian: bool = False,
        all_planners: bool = False,
        *args,
        **kwargs,
    ) -> Union[Tuple[bool, object, float, int], Tuple[object, float, float]]:
        """
        Run multiple planners sequentially and return the first successful result.

        Args:
            planners: List of planner IDs to try
            planning_func: The planning function to execute
            is_cartesian: Whether this is a cartesian planning task
            all_planners: If True, try all planners even after finding a successful one
            *args: Positional arguments for the planning function
            **kwargs: Keyword arguments for the planning function

        Returns:
            For point-to-point: (success, plan, planning_time, error_code)
            For cartesian: (plan, fraction, planning_time)
        """
        best_result = None
        best_planner = None

        for planner_id in planners:
            rospy.loginfo(f"Planning with: {planner_id}")

            if is_cartesian:
                plan, fraction, planning_time = self._execute_cartesian_task(
                    planning_func, planner_id, *args, **kwargs
                )
                if fraction > 0.9:
                    rospy.loginfo(
                        f"Cartesian plan found with {planner_id} (fraction: {fraction})"
                    )
                    if not all_planners:
                        return plan, fraction, planning_time
                    if best_result is None or fraction > best_result[1]:
                        best_result = (plan, fraction, planning_time)
                        best_planner = planner_id
            else:
                success, plan, planning_time, error_code = self._execute_planning_task(
                    planning_func, planner_id, *args, **kwargs
                )
                rospy.loginfo(
                    f"Planning with {planner_id} succeeded: {success} in time {planning_time} with error code {error_code}"
                )
                if success:
                    if not all_planners:
                        return True, plan, planning_time, error_code
                    # Store the result as (planning_time, plan, error_code) for easier comparison
                    if best_result is None or planning_time < best_result[0]:
                        best_result = (planning_time, plan, error_code)
                        best_planner = planner_id

        if best_result is not None:
            if is_cartesian:
                plan, fraction, planning_time = best_result
                rospy.loginfo(
                    f"Using cartesian plan from {best_planner} with fraction {fraction}"
                )
                return plan, fraction, planning_time
            else:
                planning_time, plan, error_code = best_result
                rospy.loginfo(f"Using plan from {best_planner}")
                return True, plan, planning_time, error_code
        else:
            if is_cartesian:
                return None, 0.0, 0.0
            else:
                return False, None, 0, 0

    def _log_planning_result(
        self,
        success: bool,
        plan: Optional[object],
        planning_time: float,
        error_code: int = 0,
        fraction: float = 1.0,
        target_name: str = None,
    ):
        """
        Log the result of a planning operation.

        Args:
            success: Whether planning was successful
            plan: The resulting plan
            planning_time: Time taken for planning
            error_code: Error code if planning failed
            fraction: Fraction of path achieved (for cartesian planning)
            target_name: Name of the target (for named target planning)
        """
        if success and plan:
            waypoints = len(plan.joint_trajectory.points)
            if target_name:
                rospy.loginfo(
                    f"Planning to '{target_name}' succeeded with {waypoints} waypoints in {planning_time:.2f} seconds"
                )
            else:
                rospy.loginfo(
                    f"Planning succeeded with {waypoints} waypoints in {planning_time:.2f} seconds"
                )
        else:
            if error_code:
                error_name = get_error_code_name(error_code)
                error_desc = get_error_description(error_code)
                rospy.logwarn(
                    f"Planning failed with error: {error_name} - {error_desc}"
                )
            elif fraction < 0.98:
                rospy.logwarn(
                    f"Cartesian planning only achieved {fraction:.2%} completion"
                )

    def plan_to_named_target(
        self, target_name: str, planner_id: str = None, all_planners: bool = False
    ):
        """
        Plan a motion to a named target position defined in the SRDF.
        Can use either a single planner or try multiple planners sequentially.

        Args:
            target_name: Name of the predefined target position
            planner_id: If specified, use this single planner. If None, try multiple planners sequentially.
            all_planners: If True, try all planners even after finding a successful one

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

        def planning_func():
            self.move_group.set_named_target(target_name)
            return self.move_group.plan()

        if planner_id:
            success, plan, planning_time, error_code = self._execute_planning_task(
                planning_func, planner_id
            )
        else:
            success, plan, planning_time, error_code = self._run_sequential_planning(
                self.point_to_point_planners, planning_func, all_planners=all_planners
            )

        self._log_planning_result(
            success, plan, planning_time, error_code, target_name=target_name
        )
        return success, plan, planning_time, error_code

    def plan_to_joint_target(
        self,
        joint_positions: List[float],
        planner_id: str = None,
        all_planners: bool = False,
    ):
        """
        Plan a motion to the specified joint positions.
        Can use either a single planner or try multiple planners sequentially.

        Args:
            joint_positions: List of joint positions in radians
            planner_id: If specified, use this single planner. If None, try multiple planners sequentially.
            all_planners: If True, try all planners even after finding a successful one

        Returns:
            tuple: (success, plan, planning_time, error_code)
        """
        rospy.loginfo(f"Planning to joint positions: {joint_positions}")

        def planning_func():
            self.move_group.set_joint_value_target(joint_positions)
            return self.move_group.plan()

        if planner_id:
            success, plan, planning_time, error_code = self._execute_planning_task(
                planning_func, planner_id
            )
        else:
            success, plan, planning_time, error_code = self._run_sequential_planning(
                self.point_to_point_planners, planning_func, all_planners=all_planners
            )

        self._log_planning_result(success, plan, planning_time, error_code)
        return success, plan, planning_time, error_code

    def plan_to_pose_target(
        self, pose, planner_id: str = None, all_planners: bool = False
    ):
        """
        Plan a motion to the specified pose in Cartesian space.
        Can use either a single planner or try multiple planners sequentially.

        Args:
            pose: geometry_msgs/Pose target
            planner_id: If specified, use this single planner. If None, try multiple planners sequentially.
            all_planners: If True, try all planners even after finding a successful one

        Returns:
            tuple: (success, plan, planning_time, error_code)
        """
        rospy.loginfo("Planning to pose target")

        def planning_func():
            self.move_group.set_pose_target(pose)
            return self.move_group.plan()

        if planner_id:
            success, plan, planning_time, error_code = self._execute_planning_task(
                planning_func, planner_id
            )
        else:
            success, plan, planning_time, error_code = self._run_sequential_planning(
                self.point_to_point_planners, planning_func, all_planners=all_planners
            )

        self._log_planning_result(success, plan, planning_time, error_code)
        return success, plan, planning_time, error_code

    def plan_cartesian_path(
        self,
        waypoints,
        eef_step=0.01,
        jump_threshold=0.0,
        planner_id: str = None,
        all_planners: bool = False,
    ):
        """
        Plan a Cartesian path through the specified waypoints.
        Can use either a single planner or try multiple planners sequentially.

        Args:
            waypoints: List of geometry_msgs/Pose waypoints
            eef_step: Step size for the end effector (meters)
            jump_threshold: Jump threshold for joint space discontinuities
            planner_id: If specified, use this single planner. If None, try multiple planners sequentially.
            all_planners: If True, try all planners even after finding a successful one

        Returns:
            tuple: (plan, fraction, planning_time)
        """
        rospy.loginfo(f"Planning Cartesian path with {len(waypoints)} waypoints")

        def planning_func():
            return self.move_group.compute_cartesian_path(
                waypoints, eef_step, jump_threshold
            )

        if planner_id:
            plan, fraction, planning_time = self._execute_cartesian_task(
                planning_func, planner_id
            )
        else:
            plan, fraction, planning_time = self._run_sequential_planning(
                self.cartesian_planners,
                planning_func,
                is_cartesian=True,
                all_planners=all_planners,
            )

        self._log_planning_result(
            fraction > 0.98, plan, planning_time, fraction=fraction
        )
        return plan, fraction, planning_time

    def plan_to_ee_offset(
        self,
        direction,
        distance,
        eef_step=0.01,
        avoid_collisions=True,
        planner_id: str = None,
        all_planners: bool = False,
    ):
        """
        Plan a linear motion from current end effector position along the specified direction.
        The direction is specified in the end-effector frame.
        Can use either a single planner or try multiple planners sequentially.

        Args:
            direction: 3D vector [x, y, z] indicating the direction of motion in the end-effector frame
            distance: Distance to move in meters
            eef_step: Step size for the end effector (meters)
            avoid_collisions: Whether to avoid collisions during planning
            planner_id: If specified, use this single planner. If None, try multiple planners sequentially.
            all_planners: If True, try all planners even after finding a successful one

        Returns:
            tuple: (success, plan, planning_time, error_code)
        """
        # Normalize the direction vector
        direction = np.array(direction, dtype=float)
        unit_direction = direction / np.linalg.norm(direction)
        rospy.loginfo(
            f"Planning linear motion in end-effector frame direction {unit_direction} for {distance} meters"
        )

        # Get current end effector pose
        current_pose = self.move_group.get_current_pose().pose

        # Convert pose to transformation matrix
        current_transform = ros_numpy.numpify(current_pose)

        # Convert direction from ee frame to world frame using rotation matrix
        world_direction = current_transform[:3, :3] @ unit_direction
        world_direction = world_direction / np.linalg.norm(world_direction)

        rospy.loginfo(f"Converted to world frame direction: {world_direction}")

        # Create target pose by applying the offset in world frame
        target_pose = current_pose
        target_pose.position.x += world_direction[0] * distance
        target_pose.position.y += world_direction[1] * distance
        target_pose.position.z += world_direction[2] * distance

        # Create waypoints for cartesian path
        waypoints = [target_pose]

        def planning_func():
            return self.move_group.compute_cartesian_path(
                waypoints, eef_step, avoid_collisions
            )

        if planner_id:
            plan, fraction, planning_time = self._execute_cartesian_task(
                planning_func, planner_id
            )
        else:
            plan, fraction, planning_time = self._run_sequential_planning(
                self.cartesian_planners,
                planning_func,
                is_cartesian=True,
                all_planners=all_planners,
            )

        # Determine success based on the fraction of path achieved
        success = fraction > 0.98  # Consider 98% or better as success
        error_code = 0  # We don't have an actual error code from cartesian planning

        self._log_planning_result(
            success, plan, planning_time, error_code, fraction=fraction
        )
        return success, plan, planning_time, error_code


if __name__ == "__main__":
    # Simple test if this file is run directly
    rospy.init_node("planner_test", anonymous=True)
    planner = Planner()

    # Clear any existing stats
    stats_reporter.clear()

    # Test planning to home position with all planners
    success, plan, planning_time, error_code = planner.plan_to_named_target(
        "all-zeros", all_planners=True
    )

    if success:
        rospy.loginfo("Planning to home position succeeded!")
    else:
        rospy.logerr("Planning to home position failed!")

    # Print statistics
    rospy.loginfo("\nPlanning Statistics:")
    rospy.loginfo(f"Total attempts: {stats_reporter.get_total_attempts()}")
    rospy.loginfo(f"Successful attempts: {stats_reporter.get_successful_attempts()}")
    rospy.loginfo(f"Overall success rate: {stats_reporter.get_success_rate():.2%}")

    # Print statistics for each planner
    for planner_id in [
        "RRTkConfigDefault",
        "RRTConnectkConfigDefault",
        "RRTstarkConfigDefault",
    ]:
        stats = stats_reporter.get_stats(planner_id)
        if stats:
            success_rate = stats_reporter.get_success_rate(planner_id)
            avg_time = stats_reporter.get_average_planning_time(planner_id)
            rospy.loginfo(f"\n{planner_id}:")
            rospy.loginfo(f"  Success rate: {success_rate:.2%}")
            rospy.loginfo(f"  Average planning time: {avg_time:.3f} seconds")
            rospy.loginfo(f"  Total attempts: {len(stats)}")
