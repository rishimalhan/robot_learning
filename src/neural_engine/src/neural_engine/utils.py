#! /usr/bin/env python3

import rospkg
import os
import yaml
import rospy
from typing import Dict, Any
from rospkg.common import ResourceNotFound


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
    try:
        pkg_path = rospack.get_path(pkg_name)
    except ResourceNotFound:
        rospy.logwarn(f"Package '{pkg_name}' not found")
        return None

    return os.path.join(pkg_path, rel_path)


def set_param(param_name: str, value: Any):
    """
    Set a parameter on the ROS parameter server.
    """
    rospy.set_param(param_name, value)


def bootstrap():
    """
    Load a YAML file and put its contents on the ROS parameter server.
    """
    # Resolve package paths if needed
    for package_name in ["neural_engine", "inspection_cell"]:
        file_path = resolve_package_path(f"package://{package_name}/config")
        if file_path is None:
            continue

        if not os.path.exists(file_path):
            rospy.logwarn(f"Config directory {file_path} does not exist")
            continue

        # Load all YAML files in the directory
        for file_name in os.listdir(file_path):
            if file_name.endswith(".yaml"):
                rospy.loginfo(f"Loading config from {file_name}")
                with open(os.path.join(file_path, file_name), "r") as f:
                    config = yaml.safe_load(f)
                # Add to parameter server
                param_name = "/" + file_name.split(".")[0]
                rospy.loginfo(f"Setting parameter {param_name}")
                set_param(param_name, config)


def get_param(param_name: str, default: Any = None) -> Any:
    """
    Get a parameter from the ROS parameter server.

    Args:
        param_name: Name of the parameter to get

    Returns:
        Parameter value

    Raises:
        KeyError: If parameter does not exist on parameter server
    """
    if not rospy.has_param(param_name):
        if default is not None:
            return default
        raise KeyError(f"Parameter '{param_name}' not found on parameter server")

    value = rospy.get_param(param_name)
    if value is None:
        raise ValueError(f"Parameter '{param_name}' exists but has None value")

    return value


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
