#! /usr/bin/env python3

import rospkg
import os
import yaml
import rospy
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple


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
