#!/usr/bin/env python3

# External

import traceback
import rospy
from neural_engine.srv import (
    GenerateViewpoints,
    GenerateViewpointsResponse,
)
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from geometry_msgs.msg import PoseStamped

# Internal

from core.utils import sample_roi_poses
from moveit_commander import MoveGroupCommander, RobotCommander


class SetCover:
    def __init__(self):
        self.move_group = MoveGroupCommander("manipulator")
        self.robot = RobotCommander()

        # Setup IK service
        rospy.wait_for_service("/compute_ik")
        self.ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)

    def filter_valid_viewpoints(self, viewpoints):
        """Filter viewpoints by IK feasibility and visibility"""
        valid_viewpoints = []

        req = GetPositionIKRequest()
        req.ik_request.group_name = "manipulator"
        req.ik_request.pose_stamped.header.frame_id = "world"
        req.ik_request.timeout = rospy.Duration(1.0)

        for i, pose in enumerate(viewpoints):
            # Check IK feasibility using service
            req.ik_request.pose_stamped.pose = pose

            try:
                resp = self.ik_srv(req)
                if resp.error_code.val == resp.error_code.SUCCESS:
                    valid_viewpoints.append(pose)
            except Exception as e:
                rospy.logwarn(f"IK service call failed for viewpoint {i+1}: {e}")

        rospy.loginfo(
            f"Filtered {len(valid_viewpoints)}/{len(viewpoints)} valid viewpoints"
        )
        return valid_viewpoints

    def generate_viewpoints(self, request):
        """Generate viewpoints service callback"""
        try:
            rospy.loginfo(
                f"Generating {request.num_samples} viewpoints with "
                f"vert_angle={request.vert_angle}, horz_angle={request.horz_angle}"
            )

            # Generate candidate viewpoints
            candidate_poses = sample_roi_poses(
                request.num_samples * 5,  # Generate more candidates for filtering
                request.vert_angle,
                request.horz_angle,
            )

            # Filter by IK feasibility
            valid_poses = self.filter_valid_viewpoints(candidate_poses)

            # Limit to requested number
            final_poses = valid_poses[: request.num_samples]

            response = GenerateViewpointsResponse()
            response.success = len(final_poses) > 0
            response.viewpoints = final_poses

            if response.success:
                response.message = f"Generated {len(final_poses)} valid viewpoints"
                rospy.loginfo(response.message)
            else:
                response.message = "No valid viewpoints found"
                rospy.logwarn(response.message)

            return response

        except Exception as e:
            rospy.logerr(f"Error generating viewpoints: {e}")
            rospy.logerr(traceback.format_exc())
            response = GenerateViewpointsResponse()
            response.success = False
            response.message = f"Error: {str(e)}"
            return response


def main():
    rospy.init_node("set_cover_service")
    set_cover = SetCover()

    service = rospy.Service(
        "generate_viewpoints", GenerateViewpoints, set_cover.generate_viewpoints
    )

    rospy.loginfo("SetCover service ready")
    rospy.spin()


if __name__ == "__main__":
    main()
