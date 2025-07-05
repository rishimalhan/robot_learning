#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from neural_engine.srv import (
    GenerateViewpoints,
    GenerateViewpointsRequest,
    GenerateViewpointsResponse,
)
from core.load_system import EnvironmentLoader
from core.utils import (
    get_robot_roi_bounds,
    init_visualization,
    visualize_waypoints,
    clear_visualization,
)
import tf.transformations as tf_trans


class SetCover:
    def __init__(self):
        """Initialize SetCover service for viewpoint generation"""
        rospy.loginfo("Initializing SetCover service...")

        # Load environment
        self.env = EnvironmentLoader()

        # Get robot ROI bounds
        self.roi_bounds = get_robot_roi_bounds(self.env)
        if not self.roi_bounds:
            rospy.logerr("Required robot_roi object not found in the scene")
            raise ValueError("robot_roi not found")

        # Initialize visualization
        self.markers, self.tf_broadcaster = init_visualization()

        # Store generated viewpoints
        self.viewpoints = []

        rospy.loginfo("SetCover initialized successfully")

    def generate_viewpoints(self, vert_angle, horz_angle, num_samples):
        """
        Generate viewpoints within robot ROI with specified camera orientations

        Args:
            vert_angle (float): Vertical angle in degrees (camera tilt from vertical)
            horz_angle (float): Horizontal angle in degrees (camera rotation around Z)
            num_samples (int): Number of random samples to generate

        Returns:
            list: Generated viewpoints as Pose objects
        """
        rospy.loginfo(
            f"Generating {num_samples} viewpoints with vert_angle={vert_angle}°, horz_angle={horz_angle}°"
        )

        # Clear previous viewpoints
        self.viewpoints = []

        # Define margins to stay within ROI
        margin_x = 0.2
        margin_y = 0.3
        margin_z = 0.1

        # Calculate sampling bounds
        x_min = self.roi_bounds["x_min"] + margin_x
        x_max = self.roi_bounds["x_max"] - margin_x
        y_min = self.roi_bounds["y_min"] + margin_y
        y_max = self.roi_bounds["y_max"] - margin_y
        z_min = self.roi_bounds["z_min"] + margin_z
        z_max = self.roi_bounds["z_max"] - margin_z

        # Generate random positions within ROI
        x_positions = np.random.uniform(x_min, x_max, num_samples)
        y_positions = np.random.uniform(y_min, y_max, num_samples)
        z_positions = np.random.uniform(z_min, z_max, num_samples)

        # Convert angles to radians
        vert_rad = np.radians(vert_angle)
        horz_rad = np.radians(horz_angle)

        # Generate viewpoints
        for i in range(num_samples):
            pose = Pose()

            # Set position
            pose.position.x = x_positions[i]
            pose.position.y = y_positions[i]
            pose.position.z = z_positions[i]

            # Calculate orientation
            # Start with camera facing down (looking at part)
            # Apply vertical angle (tilt from vertical)
            # Apply horizontal angle (rotation around Z)

            # Euler angles: roll, pitch, yaw
            # Camera facing down with tilt and rotation
            roll = 0.0  # Keep X axis aligned with global X
            pitch = vert_rad  # Tilt from vertical
            yaw = horz_rad  # Rotation around Z

            # Convert to quaternion
            quaternion = tf_trans.quaternion_from_euler(roll, pitch, yaw)
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]

            self.viewpoints.append(pose)

        rospy.loginfo(f"Generated {len(self.viewpoints)} viewpoints")
        return self.viewpoints

    def visualize_viewpoints(self):
        """Visualize generated viewpoints"""
        if not self.viewpoints:
            rospy.logwarn("No viewpoints to visualize")
            return

        # Clear previous visualization
        clear_visualization(self.markers)

        # Visualize viewpoints
        visualize_waypoints(
            self.viewpoints,
            self.markers,
            self.tf_broadcaster,
            show_labels=True,
            show_axes=True,
        )

        rospy.loginfo(f"Visualized {len(self.viewpoints)} viewpoints")

    def handle_service_request(self, req):
        """Handle service request with custom parameters"""
        try:
            # Use parameters from request
            vert_angle = req.vert_angle
            horz_angle = req.horz_angle
            num_samples = req.num_samples

            rospy.loginfo(
                f"Service request: vert_angle={vert_angle}°, horz_angle={horz_angle}°, num_samples={num_samples}"
            )

            # Generate viewpoints
            viewpoints = self.generate_viewpoints(vert_angle, horz_angle, num_samples)

            # Visualize viewpoints
            self.visualize_viewpoints()

            response = GenerateViewpointsResponse()
            response.success = True
            response.message = f"Generated {len(viewpoints)} viewpoints successfully"
            response.viewpoints = viewpoints

            return response

        except Exception as e:
            rospy.logerr(f"Error in service request: {e}")
            response = GenerateViewpointsResponse()
            response.success = False
            response.message = str(e)
            response.viewpoints = []
            return response

    def cleanup(self):
        """Clean up resources"""
        clear_visualization(self.markers)


def main():
    """Main function to run SetCover service"""
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
    except Exception as e:
        rospy.logerr(f"Error in SetCover service: {e}")
    finally:
        if "set_cover" in locals():
            set_cover.cleanup()


if __name__ == "__main__":
    main()
