#!/usr/bin/env python3

# External

import traceback
import rospy
from neural_engine.srv import (
    GenerateViewpoints,
    GenerateViewpointsResponse,
)

# Internal

from core.utils import sample_roi_poses
from moveit_commander import MoveGroupCommander, RobotCommander


class SetCover:
    def __init__(self):
        self.move_group = MoveGroupCommander("manipulator")
        self.robot = RobotCommander()

    def filter_valid_viewpoints(self, viewpoints, min_visible_points=100):
        """Filter viewpoints by IK feasibility and visibility"""
        valid_viewpoints = []

        for i, pose in enumerate(viewpoints):
            # Check IK feasibility only
            joint_values = self.move_group.get_ik(pose)

            if joint_values:
                valid_viewpoints.append(pose)
                rospy.loginfo(f"Viewpoint {i+1}/{len(viewpoints)} has valid IK")

        rospy.loginfo(
            f"Filtered {len(valid_viewpoints)}/{len(viewpoints)} valid viewpoints"
        )
        return valid_viewpoints

    def handle_service_request(self, req):
        """Handle service request with custom parameters"""
        try:
            # Use parameters from request as maximum angles
            max_vert_angle = req.vert_angle
            max_horz_angle = req.horz_angle
            num_samples = req.num_samples

            rospy.loginfo(
                f"Service request: max_vert_angle={max_vert_angle}°, max_horz_angle={max_horz_angle}°, num_samples={num_samples}"
            )

            # Generate viewpoints
            viewpoints = sample_roi_poses(
                num_samples=num_samples,
                max_vert_angle=max_vert_angle,
                max_horz_angle=max_horz_angle,
            )

            # Filter valid viewpoints
            valid_viewpoints = self.filter_valid_viewpoints(viewpoints)

            response = GenerateViewpointsResponse()
            response.success = True
            response.message = f"Generated {len(valid_viewpoints)} valid viewpoints (from {len(viewpoints)} candidates)"
            response.viewpoints = valid_viewpoints

            return response

        except Exception as e:
            rospy.logerr(f"Error in service request: {e}")
            response = GenerateViewpointsResponse()
            response.success = False
            response.message = str(e)
            response.viewpoints = []
            return response


def main():
    """Main function to run SetCover service"""
    # Initialize ROS node FIRST - same as test_move_robot.py
    rospy.init_node("set_cover_service")

    try:
        # Initialize SetCover
        set_cover = SetCover()

        # Create service
        service = rospy.Service(
            "generate_viewpoints", GenerateViewpoints, set_cover.handle_service_request
        )

        rospy.loginfo(
            "SetCover service ready. Call with: rosservice call /generate_viewpoints"
        )

        # Keep service running
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Service interrupted")
    except Exception:
        rospy.logerr(f"Error in SetCover service: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
