#!/usr/bin/env python3

import rospy
import numpy as np
import time
from core.load_system import EnvironmentLoader


def test_collision_checker():
    """Test the optimized collision checker with various configurations."""
    # Initialize the environment loader which already sets up the collision checker
    start_time = time.time()
    env = EnvironmentLoader(move_group_name="manipulator")
    init_time = time.time() - start_time
    rospy.loginfo(
        f"Environment and collision checker initialized in {init_time:.4f} seconds"
    )

    # Get the collision checker from the environment
    checker = env.collision_checker

    # Test current state
    rospy.loginfo("\nChecking current robot state for collisions...")
    start_time = time.time()
    is_valid, contacts = env.check_collision()
    check_time = time.time() - start_time

    rospy.loginfo(
        f"Current state collision check completed in {check_time:.4f} seconds"
    )
    if is_valid:
        rospy.loginfo("Result: Current state is collision-free")
    else:
        rospy.logwarn("Result: Current state is in collision")
        rospy.loginfo(checker.get_collision_report())


if __name__ == "__main__":
    try:
        test_collision_checker()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in collision testing: {e}")
