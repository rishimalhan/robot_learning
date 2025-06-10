#! /usr/bin/env python3

import rospkg
import os
import yaml
import rospy
import numpy as np
from typing import Dict, Any, List, Tuple
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import (
    Pose,
    Point,
    Quaternion,
    Vector3,
    TransformStamped,
)
from tf.transformations import (
    euler_matrix,
    quaternion_from_matrix,
)
from copy import deepcopy
import rviz_tools_py as viz
import ros_numpy


def resolve_package_path(pkg_uri: str) -> str:
    """Resolve a package:// URI to an absolute path."""
    if not pkg_uri.startswith("package://"):
        # If it's not a package URI, return the original path
        return pkg_uri

    # Strip the URI and get package + relative path
    path = pkg_uri[len("package://") :]
    pkg_name, rel_path = path.split("/", 1)

    # Get absolute path to the package
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(pkg_name)

    return os.path.join(pkg_path, rel_path)


def load_yaml_to_params(file_path: str, param_namespace: str) -> Dict:
    """
    Load a YAML file and put its contents on the ROS parameter server.

    Args:
        file_path: Path to the YAML file
        param_namespace: Namespace to use for parameters

    Returns:
        The loaded configuration dictionary
    """
    # Resolve package paths if needed
    if file_path.startswith("package://"):
        file_path = resolve_package_path(file_path)

    # Load YAML file
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    # Add to parameter server
    for key, value in config.items():
        param_path = f"{param_namespace}/{key}"
        rospy.set_param(param_path, value)

    return config


def get_param(param_name: str, default: Any = None) -> Any:
    """
    Get a parameter from the ROS parameter server with type checking.

    Args:
        param_name: Name of the parameter to get
        default: Default value if parameter doesn't exist

    Returns:
        Parameter value or default
    """
    return rospy.get_param(param_name, default)


def get_param_dict(namespace: str) -> Dict:
    """
    Get all parameters under a namespace as a dictionary.

    Args:
        namespace: Namespace to get parameters from

    Returns:
        Dictionary of parameters
    """
    # Make sure namespace starts with /
    if not namespace.startswith("/"):
        namespace = "/" + namespace

    # Get all parameters in namespace
    params = {}
    try:
        param_names = rospy.get_param_names()

        for name in param_names:
            if name.startswith(namespace):
                # Extract the key (without the namespace)
                key = name[len(namespace) + 1 :] if namespace != "/" else name[1:]
                params[key] = rospy.get_param(name)

    except Exception as e:
        rospy.logwarn(f"Failed to get parameters under {namespace}: {e}")

    return params


def convert_stl_to_obj(stl_path: str, temp_dir: str) -> str:
    """
    Convert an STL file to OBJ format for rendering.

    Args:
        stl_path: Path to the STL file
        temp_dir: Directory to store temporary files

    Returns:
        Path to the converted OBJ file
    """
    import trimesh
    from trimesh.exchange.obj import export_obj

    # Load the STL file
    mesh = trimesh.load(stl_path)

    # Create an OBJ file path
    obj_path = os.path.join(
        temp_dir, os.path.basename(stl_path).replace(".stl", ".obj")
    )

    # Export as OBJ
    with open(obj_path, "w") as f:
        f.write(export_obj(mesh))

    return obj_path


def generate_camera_trajectory(
    center: List[float],
    radius: float,
    n_points: int,
    height_range: Tuple[float, float] = (-0.5, 0.5),
    target_point: List[float] = [0, 0, 0],
) -> List[np.ndarray]:
    """
    Generate a circular camera trajectory around a center point.

    Args:
        center: Center point of the trajectory [x, y, z]
        radius: Radius of the circle
        n_points: Number of points to generate
        height_range: Min/max height variation
        target_point: Point to look at

    Returns:
        List of 4x4 camera transformation matrices
    """
    trajectory = []

    for i in range(n_points):
        angle = 2 * np.pi * i / n_points

        # Calculate position on circle with height variation
        height_param = np.sin(angle * 2) if height_range[1] > height_range[0] else 0
        height_offset = (
            height_range[0]
            + (height_range[1] - height_range[0]) * (height_param + 1) / 2
        )

        x = center[0] + radius * np.cos(angle)
        y = center[1] + height_offset
        z = center[2] + radius * np.sin(angle)

        # Create a translation matrix
        camera_pos = np.array([x, y, z])
        translation = np.eye(4)
        translation[:3, 3] = camera_pos

        # Create a rotation matrix (look at target_point)
        forward = np.array(target_point) - camera_pos
        forward = forward / np.linalg.norm(forward)

        # Compute right and up vectors for camera orientation
        world_up = np.array([0, 1, 0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Create rotation matrix
        rotation = np.eye(4)
        rotation[:3, 0] = right
        rotation[:3, 1] = up
        rotation[:3, 2] = -forward  # Negative because camera looks along -Z

        # Combine rotation and translation
        transform = rotation @ translation

        # Invert because we need world_T_camera
        transform_inv = np.linalg.inv(transform)

        trajectory.append(transform_inv)

    return trajectory


def get_roi_info(env):
    """
    Get comprehensive information about the robot_roi from the planning scene.

    Args:
        env: Environment loader instance with scene attribute

    Returns:
        dict: Dictionary containing:
            - dimensions: [x, y, z] dimensions of the ROI box
            - position: [x, y, z] center position of the ROI
            - bounds: Dictionary with x_min, x_max, y_min, y_max, z_min, z_max
        None: If robot_roi not found or not a box primitive
    """
    rospy.loginfo("Retrieving robot_roi information from planning scene...")

    # Wait a bit for the scene to be fully loaded
    rospy.sleep(1.0)

    # Get all collision objects from the scene
    scene_objects = env.scene.get_objects()

    # Check if robot_roi exists
    if "robot_roi" not in scene_objects:
        rospy.logerr("robot_roi not found in planning scene objects")
        return None

    roi_object = scene_objects["robot_roi"]

    # Check if it's a box
    if not roi_object.primitives or roi_object.primitives[0].type != 1:  # BOX is type 1
        rospy.logerr("robot_roi is not a box primitive")
        return None

    # Get dimensions from primitive
    dimensions = list(roi_object.primitives[0].dimensions)

    # Get position from pose
    position = roi_object.pose.position
    position_list = [position.x, position.y, position.z]

    # Calculate bounds (half-width from center position)
    half_dims = [d / 2 for d in dimensions]
    bounds = {}
    for i, axis in enumerate(["x", "y", "z"]):
        bounds[f"{axis}_min"] = position_list[i] - half_dims[i]
        bounds[f"{axis}_max"] = position_list[i] + half_dims[i]

    # Create comprehensive result
    roi_info = {"dimensions": dimensions, "position": position_list, "bounds": bounds}

    rospy.loginfo(f"ROI dimensions: {dimensions}")
    rospy.loginfo(f"ROI position: {position_list}")
    rospy.loginfo(
        f"ROI bounds: X: [{bounds['x_min']:.3f}, {bounds['x_max']:.3f}], "
        f"Y: [{bounds['y_min']:.3f}, {bounds['y_max']:.3f}], "
        f"Z: [{bounds['z_min']:.3f}, {bounds['z_max']:.3f}]"
    )

    return roi_info


def get_robot_roi_bounds(env):
    """
    Get the X,Y bounds of the robot_roi from the planning scene.
    A simplified version of get_roi_info() that returns only X,Y bounds.
    """
    roi_info = get_roi_info(env)
    if roi_info is None:
        return None

    return {k: v for k, v in roi_info["bounds"].items() if not k.startswith("z_")}


def create_approach_pose(target_pose: Pose, distance: float) -> Pose:
    """Create approach pose by moving back along the Z-axis of the orientation."""
    # Convert quaternion to rotation matrix
    rot_mat = ros_numpy.numpify(target_pose)

    # Get the Z axis (third column of rotation matrix)
    z_axis = rot_mat[0:3, 2]

    # Create approach pose
    approach = deepcopy(target_pose)
    # Move back along Z axis
    approach.position.x = target_pose.position.x - distance * z_axis[0]
    approach.position.y = target_pose.position.y - distance * z_axis[1]
    approach.position.z = target_pose.position.z - distance * z_axis[2]

    return approach


def publish_pose_as_transform(
    pose: Pose,
    parent_frame: str,
    child_frame: str,
    broadcaster: tf2_ros.TransformBroadcaster,
    eef_pose: Pose = None,
    markers=None,
):
    """
    Publish a pose as TF frames with optional visualization markers.

    Args:
        pose: The pose to visualize
        parent_frame: Parent frame ID
        child_frame: Child frame ID
        broadcaster: TF broadcaster to use
        eef_pose: Optional end-effector pose to visualize
        markers: Optional RvizMarkers instance for additional visualization
    """
    transforms = []

    # Create transform for TCP pose
    tcp_transform = TransformStamped()
    tcp_transform.header.stamp = rospy.Time.now()
    tcp_transform.header.frame_id = parent_frame
    tcp_transform.child_frame_id = child_frame

    # Set translation and rotation from the input pose
    tcp_transform.transform.translation.x = pose.position.x
    tcp_transform.transform.translation.y = pose.position.y
    tcp_transform.transform.translation.z = pose.position.z
    tcp_transform.transform.rotation = pose.orientation
    transforms.append(tcp_transform)

    if eef_pose:
        # Create direct transform from TCP to end-effector
        tcp_to_eef = TransformStamped()
        tcp_to_eef.header.stamp = rospy.Time.now()
        tcp_to_eef.header.frame_id = child_frame
        tcp_to_eef.child_frame_id = f"{child_frame}_eef"

        # Calculate relative transform: tcp_T_eef = inv(tcp_T_world) * eef_T_world
        tcp_matrix = ros_numpy.numpify(pose)
        eef_matrix = ros_numpy.numpify(eef_pose)
        relative_transform = np.matmul(np.linalg.inv(tcp_matrix), eef_matrix)

        # Extract translation and rotation
        translation = relative_transform[:3, 3]
        rotation = quaternion_from_matrix(relative_transform)

        # Set the transform values
        tcp_to_eef.transform.translation.x = translation[0]
        tcp_to_eef.transform.translation.y = translation[1]
        tcp_to_eef.transform.translation.z = translation[2]
        tcp_to_eef.transform.rotation.x = rotation[0]
        tcp_to_eef.transform.rotation.y = rotation[1]
        tcp_to_eef.transform.rotation.z = rotation[2]
        tcp_to_eef.transform.rotation.w = rotation[3]

        transforms.append(tcp_to_eef)

    # Send the transforms
    broadcaster.sendTransform(transforms)

    # Add visualization markers if requested
    if markers:
        # Axis markers
        axis_length = 0.1
        axis_radius = 0.01
        lifetime = 0  # 0 = forever
        markers.publishAxis(pose, axis_length, axis_radius, lifetime)

        if eef_pose:
            markers.publishAxis(eef_pose, axis_length * 0.8, axis_radius, lifetime)


def create_pose_stamped(pose: Pose, frame_id: str) -> geometry_msgs.msg.PoseStamped:
    """Create a PoseStamped from a Pose and frame_id."""
    pose_stamped = geometry_msgs.msg.PoseStamped()
    pose_stamped.header.frame_id = frame_id
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose = pose
    return pose_stamped


def visualize_pose_with_status(
    pose: Pose,
    markers,
    successful: bool = None,
    show_tf: bool = False,
    tf_broadcaster=None,
    parent_frame: str = "world",
    eef_pose: Pose = None,
):
    """
    Visualize a pose with success/failure status and optional TF frames.

    Args:
        pose: The pose to visualize
        markers: RvizMarkers instance for visualization
        successful: Optional success/failure status
        show_tf: Whether to show TF frames
        tf_broadcaster: TF broadcaster to use if showing frames
        parent_frame: Parent frame ID for TF frames
        eef_pose: Optional end-effector pose to visualize
    """
    # Create an axis marker at the pose
    axis_length = 0.1
    axis_radius = 0.01
    lifetime = 0  # 0 = forever
    markers.publishAxis(pose, axis_length, axis_radius, lifetime)

    # Add success/failure indicator if status is provided
    if successful is not None:
        diameter = 0.025
        color = "green" if successful else "red"
        markers.publishSphere(pose, color, diameter, lifetime)

    # Publish TF frames if requested
    if show_tf and tf_broadcaster:
        publish_pose_as_transform(
            pose, parent_frame, "sampled_pose", tf_broadcaster, eef_pose=eef_pose
        )


def add_pose_label(pose: Pose, text: str, markers, color: str = "white"):
    """Add a text label above a pose."""
    text_pose = Pose(
        Point(pose.position.x, pose.position.y, pose.position.z + 0.1),
        Quaternion(0, 0, 0, 1),
    )
    scale = Vector3(0.05, 0.05, 0.05)
    markers.publishText(text_pose, text, color, scale, 0)


def init_visualization(frame_id="world", node_name="viz"):
    """Initialize visualization tools.

    Args:
        frame_id: Frame ID for visualization markers
        node_name: Name for the visualization node

    Returns:
        tuple: (RvizMarkers instance, TF broadcaster)
    """
    markers = viz.RvizMarkers(frame_id, node_name)
    tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
    return markers, tf_broadcaster


def visualize_waypoints(
    waypoints,
    markers,
    tf_broadcaster=None,
    parent_frame="world",
    show_labels=True,
    show_axes=True,
    show_tf=False,
):
    """Visualize a list of waypoints with optional labels and transforms.

    Args:
        waypoints: List of geometry_msgs/Pose waypoints
        markers: RvizMarkers instance for visualization
        tf_broadcaster: Optional TF broadcaster to publish transforms
        parent_frame: Parent frame ID for TF frames
        show_labels: Whether to show waypoint labels
        show_axes: Whether to show axis markers
        show_tf: Whether to publish TF frames
    """
    for i, pose in enumerate(waypoints):
        if show_axes:
            markers.publishAxis(pose, 0.1, 0.01, 0)  # lifetime=0 means forever

        if show_labels:
            text_pose = Pose()
            text_pose.position.x = pose.position.x
            text_pose.position.y = pose.position.y
            text_pose.position.z = pose.position.z + 0.1
            text_pose.orientation.w = 1.0
            scale = Vector3(0.05, 0.05, 0.05)
            markers.publishText(text_pose, f"Point {i+1}", "white", scale, 0)

        if show_tf and tf_broadcaster:
            publish_pose_as_transform(
                pose, parent_frame, f"waypoint_{i}", tf_broadcaster
            )


def visualize_path_points(points, markers, color="white", size=0.01, show_labels=True):
    """Visualize a list of points as spheres with optional labels.

    Args:
        points: List of geometry_msgs/Point or [x,y,z] points
        markers: RvizMarkers instance for visualization
        color: Color name for the spheres
        size: Diameter of the spheres
        show_labels: Whether to show point labels
    """
    for i, point in enumerate(points):
        # Create pose at point
        pose = Pose()
        if hasattr(point, "x"):  # geometry_msgs/Point
            pose.position = point
        else:  # [x,y,z] list
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = point[2]
        pose.orientation.w = 1.0

        # Add sphere
        markers.publishSphere(pose, color, size, 0)

        if show_labels:
            text_pose = Pose()
            text_pose.position = pose.position
            text_pose.position.z += 0.05
            text_pose.orientation.w = 1.0
            scale = Vector3(0.05, 0.05, 0.05)
            markers.publishText(text_pose, f"Point {i+1}", color, scale, 0)


def visualize_grasp(
    grasp_pose, approach_pose, markers, tf_broadcaster=None, parent_frame="world"
):
    """Visualize a grasp pose with its approach pose.

    Args:
        grasp_pose: The final grasp pose
        approach_pose: The approach pose
        markers: RvizMarkers instance for visualization
        tf_broadcaster: Optional TF broadcaster to publish transforms
        parent_frame: Parent frame ID for TF frames
    """
    # Constants matching the original visualization
    axis_length = 0.1
    axis_radius = 0.01
    lifetime = 0  # 0 = forever

    # Visualize grasp pose with axis marker
    markers.publishAxis(grasp_pose, axis_length, axis_radius, lifetime)
    markers.publishSphere(grasp_pose, "green", 0.025, lifetime)

    # Visualize approach pose with slightly smaller axis
    markers.publishAxis(approach_pose, axis_length * 0.8, axis_radius, lifetime)
    markers.publishSphere(approach_pose, "blue", 0.025, lifetime)

    # Publish TF frames if broadcaster provided
    if tf_broadcaster:
        transforms = []

        # Create transform for grasp pose
        grasp_transform = TransformStamped()
        grasp_transform.header.stamp = rospy.Time.now()
        grasp_transform.header.frame_id = parent_frame
        grasp_transform.child_frame_id = "grasp_pose"
        grasp_transform.transform.translation.x = grasp_pose.position.x
        grasp_transform.transform.translation.y = grasp_pose.position.y
        grasp_transform.transform.translation.z = grasp_pose.position.z
        grasp_transform.transform.rotation = grasp_pose.orientation
        transforms.append(grasp_transform)

        # Create transform for approach pose
        approach_transform = TransformStamped()
        approach_transform.header.stamp = rospy.Time.now()
        approach_transform.header.frame_id = parent_frame
        approach_transform.child_frame_id = "approach_pose"
        approach_transform.transform.translation.x = approach_pose.position.x
        approach_transform.transform.translation.y = approach_pose.position.y
        approach_transform.transform.translation.z = approach_pose.position.z
        approach_transform.transform.rotation = approach_pose.orientation
        transforms.append(approach_transform)

        # Send the transforms
        tf_broadcaster.sendTransform(transforms)


def clear_visualization(markers):
    """Clear all visualization markers.

    Args:
        markers: RvizMarkers instance to clear
    """
    markers.deleteAllMarkers()


def add_text_annotation(
    pose: Pose, text: str, markers, color="white", height_offset=0.1
):
    """Add a text annotation above a pose.

    Args:
        pose: The pose to annotate
        text: Text to display
        markers: RvizMarkers instance for visualization
        color: Color of the text
        height_offset: How far above the pose to place the text
    """
    text_pose = Pose()
    text_pose.position.x = pose.position.x
    text_pose.position.y = pose.position.y
    text_pose.position.z = pose.position.z + height_offset
    text_pose.orientation.w = 1.0
    scale = Vector3(0.05, 0.05, 0.05)
    markers.publishText(text_pose, text, color, scale, 0)
