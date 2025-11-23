#! /usr/bin/env python3

# External

import numpy as np
import rospy
from tf.transformations import euler_matrix, euler_from_matrix, quaternion_matrix
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


def wrap_angles(angles):
    """Wrap Euler angles to [-pi, pi]."""
    angles = np.asarray(angles, dtype=np.float32)
    return ((angles + np.pi) % (2 * np.pi)) - np.pi


def sample_pose_within_roi(roi_bounds, max_tilt, rng=None, shrink_scale=0.0):
    """Sample a position/orientation pair within ROI bounds and tilt cone."""
    rng = rng or np.random.default_rng()
    x_min, x_max = roi_bounds["x_min"], roi_bounds["x_max"]
    y_min, y_max = roi_bounds["y_min"], roi_bounds["y_max"]
    z_min, z_max = roi_bounds["z_min"], roi_bounds["z_max"]
    if shrink_scale > 0.0:
        x_min = x_min + (x_max - x_min) * shrink_scale / 2.0
        x_max = x_max - (x_max - x_min) * shrink_scale / 2.0
        y_min = y_min + (y_max - y_min) * shrink_scale / 2.0
        y_max = y_max - (y_max - y_min) * shrink_scale / 2.0
        z_min = z_min + (z_max - z_min) * shrink_scale / 2.0
        z_max = z_max - (z_max - z_min) * shrink_scale / 2.0
    position = np.array(
        [
            rng.uniform(x_min, x_max),
            rng.uniform(y_min, y_max),
            rng.uniform(z_min, z_max),
        ],
        dtype=np.float32,
    )

    tilt = rng.uniform(0.0, max_tilt)
    azimuth = rng.uniform(0.0, 2 * np.pi)
    forward = np.array(
        [
            np.sin(tilt) * np.cos(azimuth),
            np.sin(tilt) * np.sin(azimuth),
            -np.cos(tilt),
        ],
        dtype=np.float32,
    )
    forward /= np.linalg.norm(forward) + 1e-9
    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(np.dot(forward, up_world)) > 0.95:
        up_world = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    right = np.cross(up_world, forward)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(forward, right)

    rot = np.eye(3)
    rot[:, 0] = right
    rot[:, 1] = up
    rot[:, 2] = forward

    euler = wrap_angles(np.array(euler_from_matrix(rot, axes="sxyz"), dtype=np.float32))
    euler[2] = 0.0  # enforce zero yaw
    return position, euler


def pose_within_bounds(position, orientation, max_tilt, roi_bounds):
    """Check if pose lies inside ROI box and tilt cone."""
    within_box = (
        roi_bounds["x_min"] <= position[0] <= roi_bounds["x_max"]
        and roi_bounds["y_min"] <= position[1] <= roi_bounds["y_max"]
        and roi_bounds["z_min"] <= position[2] <= roi_bounds["z_max"]
    )
    if not within_box:
        return False

    rot = euler_matrix(*orientation)[:3, :3]
    cam_dir = rot[:, 2]  # +Z axis (optical) in world coordinates
    cam_dir /= np.linalg.norm(cam_dir) + 1e-9
    angle = np.arccos(np.clip(cam_dir @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0))
    return angle <= max_tilt


def publish_pointcloud(pub, cloud_msg):
    """Publish a PointCloud2 message if publisher and message are valid."""
    if pub is None or cloud_msg is None:
        return
    if not isinstance(cloud_msg, PointCloud2):
        return
    pub.publish(cloud_msg)
    rospy.sleep(0.3)


def publish_voxel_grid(marker_pub, voxel_grid, frame_id="world"):
    """Publish occupied voxels as cubes in the specified frame."""
    if marker_pub is None or voxel_grid is None:
        return
    centers = voxel_grid.get_occupied_voxel_centers()
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "voxel_grid"
    marker.id = 0
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    scale = float(np.min(voxel_grid.voxel_size))
    if scale <= 0:
        scale = 0.01
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    marker.color.r = 0.1
    marker.color.g = 0.8
    marker.color.b = 0.2
    marker.color.a = 0.4
    marker.points = [
        Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in centers
    ]
    marker_pub.publish(marker)
    rospy.sleep(0.3)


def publish_frame_marker(
    marker_pub,
    origin,
    quaternion,
    ns,
    marker_id,
    length=0.15,
    frame_id="world",
):
    """Publish XYZ axes for a frame as a Marker.LINE_LIST."""
    if marker_pub is None:
        return
    origin = np.asarray(origin, dtype=np.float32)
    quaternion = np.asarray(quaternion, dtype=np.float32)
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.01
    rot = quaternion_matrix(quaternion)[:3, :3]
    axes = [
        (np.array([1, 0, 0], dtype=np.float32), ColorRGBA(1.0, 0.0, 0.0, 1.0)),
        (np.array([0, 1, 0], dtype=np.float32), ColorRGBA(0.0, 1.0, 0.0, 1.0)),
        (np.array([0, 0, 1], dtype=np.float32), ColorRGBA(0.0, 0.0, 1.0, 1.0)),
    ]
    points = []
    colors = []
    for axis, color in axes:
        start = Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
        end_vec = origin + rot @ axis * length
        end = Point(x=float(end_vec[0]), y=float(end_vec[1]), z=float(end_vec[2]))
        points.extend([start, end])
        colors.extend([color, color])
    marker.points = points
    marker.colors = colors
    marker.pose.orientation.w = 1.0
    marker_pub.publish(marker)
    rospy.sleep(0.3)
