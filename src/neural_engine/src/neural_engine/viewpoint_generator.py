#!/usr/bin/env python3

# External

import random
import traceback
import rospy
import tf2_ros, tf
import numpy as np
from neural_engine.srv import (
    GenerateViewpoints,
    GenerateViewpointsResponse,
)
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf.transformations import quaternion_matrix
import ros_numpy

# Internal

from core.utils import sample_roi_poses
from neural_engine.simulated_perception import SimulatedPerception

# Constants
EVALUATION_CAMERA_FRAME = (
    "evaluation_camera"  # Dedicated frame for viewpoint evaluation
)


class ViewpointGenerator:
    def __init__(self):
        self._initialize = False
        self.timeout = 10.0
        self._ik_srv = None
        self._perception = None
        self._tf_broadcaster = None
        self._original_camera_frame = None

    def initialize(self):
        # Wait for environment to be ready
        rospy.wait_for_service("/compute_ik", timeout=self.timeout)
        # Setup IK service
        self._ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)

        # Setup simulated perception for pointcloud generation
        self._perception = SimulatedPerception()

        # Setup TF broadcaster for publishing evaluation camera poses
        self._tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        import collections
        import collections.abc

        collections.Sequence = collections.abc.Sequence
        listener = tf.TransformListener()
        (trans, rot) = listener.lookupTransform("tool0", "tcp", rospy.Time(0))
        T = quaternion_matrix(rot)
        T[:3, 3] = trans

        self._tcp_transform = T

        # Store original camera frame to restore later
        self._original_camera_frame = self._perception.camera_params.frame_id

        # Temporarily override the camera frame for evaluation
        self._perception.camera_params.frame_id = EVALUATION_CAMERA_FRAME

        self._initialize = True

    def _publish_evaluation_camera_transform(self, pose):
        """
        Publish a camera transform to the dedicated evaluation frame
        """

        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "world"
        transform.child_frame_id = EVALUATION_CAMERA_FRAME

        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation = pose.orientation

        self._tf_broadcaster.sendTransform(transform)
        rospy.sleep(0.1)  # Allow transform to propagate

    def _generate_pointcloud_from_pose(self, pose):
        """
        Generate pointcloud from a specific camera pose without affecting real robot
        """

        try:
            # Publish to dedicated evaluation frame (not the real robot frame)
            self._publish_evaluation_camera_transform(pose)

            # Generate pointcloud from this viewpoint
            pointcloud = self._perception.generate_pointcloud()
            if pointcloud is None:
                return None
            pointcloud = self._perception._to_pointcloud_msg(pointcloud)
            return pointcloud

        except Exception as e:
            rospy.logwarn(f"Failed to generate pointcloud from pose: {e}")
            return None

    def _ik_frame_transform(self, pose):
        # ToDo: Make a generic environment class that constructs
        # itself from the moveit scene and provides the necessary transforms
        """
        Transform pose from world to TCP frame
        """
        # Convert to Pose if PoseStamped
        if isinstance(pose, PoseStamped):
            pose = pose.pose
        return ros_numpy.geometry.numpy_to_pose(
            np.matmul(ros_numpy.numpify(pose), np.linalg.inv(self._tcp_transform))
        )

    def filter_valid_viewpoints(self, viewpoints):
        """
        Filter viewpoints by IK feasibility and pointcloud visibility
        """

        valid_viewpoints = []
        valid_pointclouds = []
        random.shuffle(viewpoints)

        req = GetPositionIKRequest()
        req.ik_request.group_name = "manipulator"
        req.ik_request.pose_stamped.header.frame_id = "world"
        req.ik_request.timeout = rospy.Duration(1.0)

        for i, pose in enumerate(viewpoints):
            # Check IK feasibility using service
            req.ik_request.pose_stamped.pose = self._ik_frame_transform(pose)

            try:
                resp = self._ik_srv(req)
                if resp.error_code.val == resp.error_code.SUCCESS:
                    # IK is valid, now check pointcloud
                    pointcloud = self._generate_pointcloud_from_pose(pose)
                    if pointcloud is not None:
                        valid_viewpoints.append(pose)
                        valid_pointclouds.append(pointcloud)

            except Exception as e:
                rospy.logwarn(f"Viewpoint validation failed for viewpoint {i+1}: {e}")

        return valid_viewpoints, valid_pointclouds

    def generate_viewpoints(self, request):
        """
        Generate viewpoints service callback
        """

        if not self._initialize:
            self.initialize()

        try:
            rospy.loginfo(
                f"Generating {request.num_samples} viewpoints with "
                f"vert_angle={request.vert_angle}, horz_angle={request.horz_angle}"
            )
            rospy.loginfo(f"Using evaluation camera frame: {EVALUATION_CAMERA_FRAME}")

            # Generate candidate viewpoints
            candidate_poses = sample_roi_poses(
                request.num_samples,  # Generate more candidates for filtering
                request.vert_angle,
                request.horz_angle,
            )

            # Filter by IK feasibility and pointcloud visibility
            valid_viewpoints, valid_pointclouds = self.filter_valid_viewpoints(
                candidate_poses
            )

            response = GenerateViewpointsResponse()
            response.success = len(valid_viewpoints) > 0
            response.viewpoints = valid_viewpoints
            response.pointclouds = valid_pointclouds

            if response.success:
                response.message = f"Generated {len(valid_viewpoints)} valid viewpoints"
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

    def cleanup(self):
        """
        Restore original camera frame and cleanup
        """

        if self._original_camera_frame is not None:
            self._perception.camera_params.frame_id = self._original_camera_frame
            rospy.loginfo(
                f"Restored original camera frame: {self._original_camera_frame}"
            )


def main():
    rospy.init_node("set_cover_service")
    viewpoint_generator = ViewpointGenerator()

    rospy.Service(
        "generate_viewpoints",
        GenerateViewpoints,
        viewpoint_generator.generate_viewpoints,
    )
    rospy.loginfo("ViewpointGenerator service ready")

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        viewpoint_generator.cleanup()


if __name__ == "__main__":
    main()
