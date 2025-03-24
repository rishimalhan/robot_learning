#!/usr/bin/env python3

"""
This module defines MoveIt error code enumerations and utility functions for error handling.
"""

# External

from enum import Enum


class MoveItErrorCodes(Enum):
    """
    Enumeration of MoveIt error codes based on moveit_msgs/MoveItErrorCodes.msg
    """

    SUCCESS = 1
    PLANNING_FAILED = 99999
    INVALID_MOTION_PLAN = -1
    MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE = -2
    CONTROL_FAILED = -3
    UNABLE_TO_ACQUIRE_SENSOR_DATA = -4
    TIMED_OUT = -5
    PREEMPTED = -6
    START_STATE_IN_COLLISION = -10
    START_STATE_VIOLATES_PATH_CONSTRAINTS = -11
    GOAL_IN_COLLISION = -12
    GOAL_VIOLATES_PATH_CONSTRAINTS = -13
    GOAL_CONSTRAINTS_VIOLATED = -14
    INVALID_GROUP_NAME = -15
    INVALID_GOAL_CONSTRAINTS = -16
    INVALID_ROBOT_STATE = -17
    INVALID_LINK_NAME = -18
    INVALID_OBJECT_NAME = -19
    FRAME_TRANSFORM_FAILURE = -21
    COLLISION_CHECKING_UNAVAILABLE = -22
    ROBOT_STATE_STALE = -23
    SENSOR_INFO_STALE = -24
    COMMUNICATION_FAILURE = -25
    NO_IK_SOLUTION = -31
    FAILURE = 99999


def get_error_code_name(error_code):
    """
    Get the name of a MoveIt error code.

    Args:
        error_code: The MoveIt error code (either as int or MoveItErrorCodes object)

    Returns:
        str: The name of the error code
    """
    # Handle the case where error_code is a MoveItErrorCodes object from moveit_msgs
    if hasattr(error_code, "val"):
        code_value = error_code.val
    else:
        code_value = error_code

    # Try to find the corresponding enum value
    for code in MoveItErrorCodes:
        if code.value == code_value:
            return code.name

    return f"UNKNOWN_ERROR_{code_value}"


def get_error_description(error_code):
    """
    Get a human-readable description of a MoveIt error code.

    Args:
        error_code: The MoveIt error code (either as int or MoveItErrorCodes object)

    Returns:
        str: A description of the error code
    """
    descriptions = {
        MoveItErrorCodes.SUCCESS: "Success",
        MoveItErrorCodes.PLANNING_FAILED: "Planning failed",
        MoveItErrorCodes.INVALID_MOTION_PLAN: "Invalid motion plan",
        MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE: "Motion plan invalidated by environment change",
        MoveItErrorCodes.CONTROL_FAILED: "Control failed",
        MoveItErrorCodes.UNABLE_TO_ACQUIRE_SENSOR_DATA: "Unable to acquire sensor data",
        MoveItErrorCodes.TIMED_OUT: "Timed out",
        MoveItErrorCodes.PREEMPTED: "Preempted",
        MoveItErrorCodes.START_STATE_IN_COLLISION: "Start state in collision",
        MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS: "Start state violates path constraints",
        MoveItErrorCodes.GOAL_IN_COLLISION: "Goal in collision",
        MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS: "Goal violates path constraints",
        MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: "Goal constraints violated",
        MoveItErrorCodes.INVALID_GROUP_NAME: "Invalid group name",
        MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: "Invalid goal constraints",
        MoveItErrorCodes.INVALID_ROBOT_STATE: "Invalid robot state",
        MoveItErrorCodes.INVALID_LINK_NAME: "Invalid link name",
        MoveItErrorCodes.INVALID_OBJECT_NAME: "Invalid object name",
        MoveItErrorCodes.FRAME_TRANSFORM_FAILURE: "Frame transform failure",
        MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE: "Collision checking unavailable",
        MoveItErrorCodes.ROBOT_STATE_STALE: "Robot state is stale",
        MoveItErrorCodes.SENSOR_INFO_STALE: "Sensor information is stale",
        MoveItErrorCodes.COMMUNICATION_FAILURE: "Communication failure",
        MoveItErrorCodes.NO_IK_SOLUTION: "No IK solution found",
        MoveItErrorCodes.FAILURE: "Failure",
    }

    # Handle the case where error_code is a MoveItErrorCodes object from moveit_msgs
    if hasattr(error_code, "val"):
        code_value = error_code.val
    else:
        code_value = error_code

    # Try to find the corresponding enum value
    for code in MoveItErrorCodes:
        if code.value == code_value:
            return descriptions.get(code, f"Unknown error code: {code_value}")

    return f"Unknown error code: {code_value}"
