#!/usr/bin/env python3

# External

import rospy
import math
from tf.transformations import quaternion_from_euler
import argparse
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# Internal

from inspection_cell.load_system import EnvironmentLoader
from inspection_cell.planner import Planner
from inspection_cell.executor import Executor
from inspection_cell.utils import get_roi_info


class ReachabilityAnalysis:
    """Analyze robot reachability by testing specific orientations at positions within a ROI."""

    def __init__(self):
        # Initialize ROS node
        rospy.init_node("reachability_analysis", anonymous=True)

        # Create visualization publisher for results
        self.marker_pub = rospy.Publisher(
            "/reachability_analysis/markers", MarkerArray, queue_size=10
        )

        # Load robot environment
        self.env_loader = self._load_environment()

        # Initialize the planner
        self.planner = Planner(move_group_name="manipulator")

        # Get ROI information from the planning scene using utility function
        self.roi_info = get_roi_info(self.env_loader)
        if not self.roi_info:
            rospy.logerr("Failed to get ROI information. Aborting.")
            return

        # Track markers for cleanup
        self.marker_ids = set()

        rospy.loginfo("ReachabilityAnalysis initialized successfully")

    def _load_environment(self):
        """Load the robot environment."""
        rospy.loginfo("Loading robot environment...")
        return EnvironmentLoader(move_group_name="manipulator", clear_scene=False)

    def move_to_home(self):
        """Move the robot to the home position."""
        rospy.loginfo("Moving to home position...")

        # Get available named targets
        named_targets = self.planner.get_named_targets()

        # Plan to home or all-zeros
        if "all-zeros" in named_targets:
            success, plan, _, _ = self.planner.plan_to_named_target("all-zeros")
        else:
            success, plan, _, _ = self.planner.plan_to_home()

        if success and plan:
            # Execute the plan
            executor = Executor(move_group_name="manipulator")
            success = executor.execute_plan(plan)
            return success
        else:
            rospy.logwarn("Failed to plan to home position")
            return False

    def _create_pose_with_orientation(self, position, roll, pitch, yaw):
        """Create a pose with the given position and orientation."""
        q = quaternion_from_euler(roll, pitch, yaw)

        pose = Pose()
        pose.position = Point(*position)
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        return pose

    def _test_orientations_at_position(self, position):
        """Test 5 specific orientations at a given position and return reachability score."""
        # Define the 5 orientations:
        # 1. Tool Z pointing directly down (0 degrees from vertical)
        # 2. Tool Z at +30 degrees in pitch from vertical
        # 3. Tool Z at -30 degrees in pitch from vertical
        # 4. Tool Z at +30 degrees in roll from vertical
        # 5. Tool Z at -30 degrees in roll from vertical

        # Convert degrees to radians
        angle_rad = math.radians(60)
        orientations = [
            (0, math.pi / 2, 0),
            (0, math.pi / 2 + angle_rad, 0),
            (0, math.pi / 2 + angle_rad / 2, 0),
            (0, math.pi / 2 - angle_rad, 0),
            (0, math.pi / 2 - angle_rad / 2, 0),
            (angle_rad, math.pi / 2, 0),
            (angle_rad / 2, math.pi / 2, 0),
            (-angle_rad, math.pi / 2, 0),
            (-angle_rad / 2, math.pi / 2, 0),
        ]

        reachable_count = 0

        for i, (roll, pitch, yaw) in enumerate(orientations):
            # Create the pose
            pose = self._create_pose_with_orientation(position, roll, pitch, yaw)

            # Check if pose is reachable using planner's existing methods
            # First set the pose target
            self.planner.move_group.set_pose_target(pose)

            # Try to plan to the pose - this will automatically check IK
            success, _, _, _ = self.planner.move_group.plan()

            # Clear the target to avoid affecting future plans
            self.planner.move_group.clear_pose_targets()

            if success:
                reachable_count += 1
                rospy.logdebug(
                    f"Position {position} with orientation {i+1} is reachable"
                )
            else:
                rospy.logdebug(
                    f"Position {position} with orientation {i+1} is NOT reachable"
                )

        # Return score as percentage of reachable orientations
        return reachable_count / len(orientations)

    def _get_color_for_score(self, score):
        """Get color based on reachability score."""
        if score == 0:
            return ColorRGBA(1.0, 0.0, 0.0, 0.7)  # Red - unreachable
        elif score == 1.0:
            return ColorRGBA(0.0, 1.0, 0.0, 0.7)  # Green - fully reachable
        else:
            return ColorRGBA(1.0, 1.0, 0.0, 0.7)  # Yellow - partially reachable

    def visualize_reachability(self, positions, scores):
        """Visualize the reachability analysis results."""
        marker_array = MarkerArray()

        for i, (position, score) in enumerate(zip(positions, scores)):
            # Create a cube marker for this position
            marker = Marker()
            marker.header.frame_id = self.env_loader.planning_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "reachability"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Set position and orientation
            marker.pose.position = Point(*position)
            marker.pose.orientation.w = 1.0

            # Set size (cube size)
            cube_size = min(self.roi_info["dimensions"]) / (
                self.num_samples_per_dim + 1
            )
            marker.scale = Vector3(cube_size, cube_size, cube_size)

            # Set color based on score
            marker.color = self._get_color_for_score(score)

            # Set marker persistence
            marker.lifetime = rospy.Duration(0)

            # Add marker to array
            marker_array.markers.append(marker)
            self.marker_ids.add(i)

        # Publish all markers
        self.marker_pub.publish(marker_array)
        rospy.loginfo(f"Published {len(positions)} reachability markers")

    def analyze_reachability(self, num_samples_per_dim=4):
        """Analyze reachability in the ROI with a uniform grid of positions."""
        if not self.roi_info:
            rospy.logerr("Cannot analyze reachability - ROI information not available")
            return

        # Store for visualization
        self.num_samples_per_dim = num_samples_per_dim

        # Move to home position first
        if not self.move_to_home():
            rospy.logwarn("Failed to move to home position, continuing analysis anyway")

        # Extract ROI dimensions and position
        dimensions = self.roi_info["dimensions"]
        roi_position = self.roi_info["position"]

        # Calculate the step size for each dimension
        step_sizes = [dim / (num_samples_per_dim + 1) for dim in dimensions]

        # Calculate the starting corner (minimum coordinates)
        start_corner = [
            roi_position[i] - dimensions[i] / 2 + step_sizes[i] for i in range(3)
        ]

        # Generate uniform grid of positions
        positions = []
        for x_idx in range(num_samples_per_dim):
            for y_idx in range(num_samples_per_dim):
                for z_idx in range(num_samples_per_dim):
                    pos = [
                        start_corner[0] + x_idx * step_sizes[0],
                        start_corner[1] + y_idx * step_sizes[1],
                        start_corner[2] + z_idx * step_sizes[2],
                    ]
                    positions.append(pos)

        rospy.loginfo(f"Testing {len(positions)} positions with 5 orientations each")

        # Test reachability at each position
        scores = []
        total_positions = len(positions)

        for i, position in enumerate(positions):
            # Log progress
            if i % 10 == 0 or i == total_positions - 1:
                rospy.loginfo(f"Testing position {i+1}/{total_positions}")

            # Test orientations at this position
            score = self._test_orientations_at_position(position)
            scores.append(score)

            # Periodically visualize results
            if (i + 1) % 10 == 0 or i == total_positions - 1:
                self.visualize_reachability(positions[: i + 1], scores[: i + 1])

        # Final visualization update
        self.visualize_reachability(positions, scores)

        # Calculate statistics
        fully_reachable = sum(1 for s in scores if s == 1.0)
        partially_reachable = sum(1 for s in scores if 0 < s < 1.0)
        unreachable = sum(1 for s in scores if s == 0)

        rospy.loginfo("Reachability analysis completed:")
        rospy.loginfo(
            f"  - Fully reachable positions: {fully_reachable} ({fully_reachable/total_positions*100:.1f}%)"
        )
        rospy.loginfo(
            f"  - Partially reachable positions: {partially_reachable} ({partially_reachable/total_positions*100:.1f}%)"
        )
        rospy.loginfo(
            f"  - Unreachable positions: {unreachable} ({unreachable/total_positions*100:.1f}%)"
        )

        # Move back to home position
        self.move_to_home()

        return positions, scores

    def cleanup(self):
        """Clean up resources before shutting down."""
        # Delete all published markers
        if self.marker_ids:
            marker_array = MarkerArray()
            for marker_id in self.marker_ids:
                marker = Marker()
                marker.header.frame_id = self.env_loader.planning_frame
                marker.header.stamp = rospy.Time.now()
                marker.ns = "reachability"
                marker.id = marker_id
                marker.action = Marker.DELETE
                marker_array.markers.append(marker)

            self.marker_pub.publish(marker_array)
            rospy.loginfo(f"Cleared {len(self.marker_ids)} visualization markers")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze robot reachability in a region of interest"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="Number of samples per dimension (total positions = samples^3)",
    )

    # Parse arguments
    args = parser.parse_args(rospy.myargv()[1:])

    try:
        # Create reachability analysis
        analyzer = ReachabilityAnalysis()

        # Wait for publishers to connect
        rospy.sleep(1.0)

        # Analyze reachability
        analyzer.analyze_reachability(num_samples_per_dim=args.samples)

        # Keep visualization active
        rospy.loginfo(
            "Analysis complete. Visualization remains active. Press Ctrl+C to exit."
        )
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted by user")
    except Exception as e:
        rospy.logerr(f"Error in reachability analysis: {e}")
        import traceback

        rospy.logerr(traceback.format_exc())
    finally:
        if "analyzer" in locals():
            analyzer.cleanup()


if __name__ == "__main__":
    main()
