#!/usr/bin/env python3

import rospy
from inspection_cell.load_system import EnvironmentLoader
from inspection_cell.planner import Planner
from inspection_cell.executor import Executor


def main():
    # Initialize ROS node
    rospy.init_node("test_robot_motion_node", anonymous=True)
    env = EnvironmentLoader()
    executor = Executor()

    # Move to home position
    success, plan, planning_time, error_code = env.planner.plan_to_named_target("home")
    if success and plan:
        executor.execute_plan(plan)
    success, plan, planning_time, error_code = env.planner.plan_to_ee_offset(
        direction=[0, 0, 1],
        distance=0.2,
    )
    if success and plan:
        executor.execute_plan(plan)

    rospy.loginfo("Test complete")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
