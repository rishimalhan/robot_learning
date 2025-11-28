#! /usr/bin/env python3

# External

import numpy as np
import rospy
from tf.transformations import (
    euler_matrix,
    euler_from_matrix,
    quaternion_matrix,
    quaternion_from_euler,
)
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


def wrap_angles(angles):
    """Wrap Euler angles to [-pi, pi]."""
    angles = np.asarray(angles, dtype=np.float32)
    return ((angles + np.pi) % (2 * np.pi)) - np.pi


def sample_pose_within_roi(roi_bounds, max_tilt, shrink_scale=0.2):
    """Sample a position/orientation pair within ROI bounds and tilt cone."""
    shrink = float(np.clip(shrink_scale, 0.0, 0.99))
    x_min, x_max = roi_bounds["x_min"], roi_bounds["x_max"]
    y_min, y_max = roi_bounds["y_min"], roi_bounds["y_max"]
    z_min, z_max = roi_bounds["z_min"], roi_bounds["z_max"]
    if shrink > 0.0:
        x_span = x_max - x_min
        y_span = y_max - y_min
        z_span = z_max - z_min
        x_min = x_min + 0.5 * shrink * x_span
        x_max = x_max - 0.5 * shrink * x_span
        y_min = y_min + 0.5 * shrink * y_span
        y_max = y_max - 0.5 * shrink * y_span
        z_min = z_min + 0.5 * shrink * z_span
        z_max = z_max - 0.5 * shrink * z_span
    position = np.array(
        [
            np.random.uniform(low=x_min, high=x_max),
            np.random.uniform(low=y_min, high=y_max),
            np.random.uniform(low=z_min, high=z_max),
        ],
        dtype=np.float32,
    )
    phi = float(np.random.uniform(low=0.0, high=max_tilt))
    theta = float(np.random.uniform(low=-np.pi, high=np.pi))

    z_axis = np.array(
        [
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            -np.cos(phi),
        ],
        dtype=np.float32,
    )
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        z_axis /= z_norm

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(up, z_axis)) > 0.995:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    x_axis = np.cross(up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_axis = np.cross(up, z_axis)
        x_norm = np.linalg.norm(x_axis)
    x_axis /= x_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-9
    rot = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float64, copy=False)
    euler = np.array(euler_from_matrix(rot, axes="sxyz"), dtype=np.float32)
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
    rospy.sleep(0.05)


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
    rospy.sleep(0.05)


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
    rospy.sleep(0.05)


def publish_camera_frustum(
    marker_pub,
    origin,
    quaternion,
    depth_min,
    depth_max,
    radius,
    marker_id=60,
    frame_id="world",
    segments=32,
):
    """Render a truncated frustum that visualizes perception constraints."""
    if marker_pub is None:
        return
    origin = np.asarray(origin, dtype=np.float32)
    quaternion = np.asarray(quaternion, dtype=np.float32)
    rot = quaternion_matrix(quaternion)[:3, :3]
    near_depth = depth_min
    far_depth = depth_max - 0.3
    far_radius = 0.1
    near_radius = max(far_radius * (near_depth / far_depth), 1e-3)

    marker_mesh = Marker()
    marker_mesh.header.frame_id = frame_id
    marker_mesh.header.stamp = rospy.Time.now()
    marker_mesh.ns = "camera_frustum"
    marker_mesh.id = marker_id
    marker_mesh.type = Marker.TRIANGLE_LIST
    marker_mesh.action = Marker.ADD
    marker_mesh.color = ColorRGBA(0.1, 0.5, 1.0, 0.2)

    marker_edges = Marker()
    marker_edges.header.frame_id = frame_id
    marker_edges.header.stamp = rospy.Time.now()
    marker_edges.ns = "camera_frustum_edges"
    marker_edges.id = marker_id + 1
    marker_edges.type = Marker.LINE_LIST
    marker_edges.action = Marker.ADD
    marker_edges.scale.x = 0.003
    marker_edges.color = ColorRGBA(0.1, 0.6, 1.0, 0.2)

    def to_point(vec):
        return Point(x=float(vec[0]), y=float(vec[1]), z=float(vec[2]))

    def append_triangle(p_a, p_b, p_c, color):
        marker_mesh.points.extend([to_point(p_a), to_point(p_b), to_point(p_c)])
        marker_mesh.colors.extend([color, color, color])

    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    near_pts = []
    far_pts = []
    for ang in angles:
        base_near = np.array(
            [near_radius * np.cos(ang), near_radius * np.sin(ang), near_depth],
            dtype=np.float32,
        )
        base_far = np.array(
            [far_radius * np.cos(ang), far_radius * np.sin(ang), far_depth],
            dtype=np.float32,
        )
        near_pts.append(origin + rot @ base_near)
        far_pts.append(origin + rot @ base_far)

    # Near and far rings
    for pts in (near_pts, far_pts):
        for i in range(segments):
            marker_edges.points.extend(
                [to_point(pts[i]), to_point(pts[(i + 1) % segments])]
            )
    # Walls + rays from apex
    for i in range(segments):
        marker_edges.points.extend([to_point(origin), to_point(far_pts[i])])
        marker_edges.points.extend([to_point(near_pts[i]), to_point(far_pts[i])])

        next_idx = (i + 1) % segments
        color = ColorRGBA(0.1, 0.5, 1.0, 0.15)
        append_triangle(origin, far_pts[i], far_pts[next_idx], color)
        append_triangle(near_pts[i], far_pts[i], far_pts[next_idx], color)
        append_triangle(near_pts[i], far_pts[next_idx], near_pts[next_idx], color)

    marker_pub.publish(marker_mesh)
    marker_pub.publish(marker_edges)


def publish_part_marker(marker_pub, part_spec, marker_id=50, frame_id="world"):
    """Publish the inspection part mesh as a marker."""
    if marker_pub is None or part_spec is None:
        return
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "inspection_part"
    marker.id = marker_id
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.ADD
    mesh_path = part_spec.get("mesh_path")
    if not mesh_path:
        return
    marker.mesh_resource = f"file://{mesh_path}"
    marker.mesh_use_embedded_materials = False
    scale = part_spec.get("scale", [1.0, 1.0, 1.0])
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    color = ColorRGBA(0.8, 0.2, 0.2, 0.6)
    marker.color = color
    pose = part_spec.get("pose", {})
    position = pose.get("position", [0.0, 0.0, 0.0])
    orientation = pose.get("orientation", [0.0, 0.0, 0.0])
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    quat = quaternion_from_euler(*orientation)
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    marker_pub.publish(marker)
    rospy.sleep(0.05)


def publish_camera_mesh(marker_pub, mesh_path, pose, marker_id=70, frame_id="world"):
    """Publish the depth camera mesh (if available) at the provided pose."""
    if marker_pub is None or not mesh_path:
        return
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "camera_mesh"
    marker.id = marker_id
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.ADD
    marker.mesh_resource = f"file://{mesh_path}"
    marker.mesh_use_embedded_materials = False

    scale = pose.get("scale", [1.0, 1.0, 1.0])
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]

    color = ColorRGBA(0.6, 0.3, 3.0, 0.8)
    marker.color = color

    position = pose.get("position", [0.0, 0.0, 0.0])
    orientation = pose.get("orientation", [0.0, 0.0, 0.0])
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    quat = quaternion_from_euler(*orientation)
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    marker_pub.publish(marker)
    rospy.sleep(0.05)
