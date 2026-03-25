#! /usr/bin/env python3

import rospy
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import std_msgs.msg
from collections import deque


class PointCloudProcessor:
    def __init__(
        self,
        enable_reconstruction=False,
        enable_mesh=False,
        enable_cad=False,
        voxel_size=0.001,
        buffer_size=1000,
        mesh_interval=10,
        alpha=1.0,
    ):
        """
        Initialize PointCloudProcessor with configurable features

        Args:
            enable_reconstruction (bool): Enable point cloud reconstruction
            enable_mesh (bool): Enable mesh generation
            enable_cad (bool): Enable CAD feature extraction
            voxel_size (float): Voxel size for downsampling
            buffer_size (int): Point cloud buffer size
            mesh_interval (int): Mesh update interval
            alpha (float): Alpha value for mesh reconstruction
        """
        self.enable_reconstruction = enable_reconstruction
        self.enable_mesh = enable_mesh
        self.enable_cad = enable_cad
        self.voxel_size = voxel_size
        self.alpha = alpha
        self.mesh_interval = mesh_interval

        # Initialize based on enabled features
        self.latest_pointcloud = None
        self.point_count = 0

        if enable_reconstruction or enable_mesh or enable_cad:
            self.pointcloud_buffer = deque(maxlen=buffer_size)
            self.global_pcd = o3d.geometry.PointCloud()
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        else:
            self.pointcloud_buffer = None
            self.global_pcd = None

        # Publishers
        self.reconstruction_pub = rospy.Publisher(
            "reconstruction", PointCloud2, queue_size=1, latch=True
        )
        if enable_mesh:
            self.mesh_pub = rospy.Publisher(
                "reconstructed_mesh", MarkerArray, queue_size=1, latch=True
            )
        if enable_cad:
            self.cad_pub = rospy.Publisher(
                "cad_features", MarkerArray, queue_size=1, latch=True
            )

        # Subscriber
        self.pointcloud_sub = rospy.Subscriber(
            "camera/points", PointCloud2, self.pointcloud_callback, queue_size=1
        )
        rospy.wait_for_message("camera/points", PointCloud2, timeout=10.0)

        features = []
        if enable_reconstruction:
            features.append("reconstruction")
        if enable_mesh:
            features.append("mesh")
        if enable_cad:
            features.append("cad")
        rospy.loginfo(
            f"PointCloud processor initialized with: {', '.join(features) if features else 'basic publishing'}"
        )

    def pointcloud_callback(self, cloud_msg):
        """Process incoming pointcloud data"""
        self.latest_pointcloud = cloud_msg

        # Basic publishing (always enabled)
        self._publish_basic_pointcloud()

        # Advanced processing only if enabled
        if self.enable_reconstruction or self.enable_mesh or self.enable_cad:
            points = pc2.read_points(cloud_msg, skip_nans=True)
            points_array = np.array(list(points))

            if len(points_array) == 0:
                return

            self.pointcloud_buffer.append(points_array)

            # Convert to Open3D and integrate
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points_array[:, :3])
            self._integrate_pointcloud(new_pcd)

            if self.enable_reconstruction:
                self._publish_reconstruction()
            if self.enable_mesh:
                self._publish_mesh()
            if self.enable_cad:
                self._publish_cad_features()

    def _publish_basic_pointcloud(self):
        """Publish basic pointcloud with world frame"""
        if self.latest_pointcloud is None:
            return

        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = rospy.Time.now()
        cloud_msg.header.frame_id = "world"
        cloud_msg.height = self.latest_pointcloud.height
        cloud_msg.width = self.latest_pointcloud.width
        cloud_msg.fields = self.latest_pointcloud.fields
        cloud_msg.is_bigendian = self.latest_pointcloud.is_bigendian
        cloud_msg.point_step = self.latest_pointcloud.point_step
        cloud_msg.row_step = self.latest_pointcloud.row_step
        cloud_msg.data = self.latest_pointcloud.data
        cloud_msg.is_dense = self.latest_pointcloud.is_dense

        self.reconstruction_pub.publish(cloud_msg)

    def _integrate_pointcloud(self, new_pcd):
        """Integrate new points into global point cloud"""
        if len(self.global_pcd.points) == 0:
            self.global_pcd = new_pcd
        else:
            combined_pcd = self.global_pcd + new_pcd
            self.global_pcd = combined_pcd.voxel_down_sample(voxel_size=self.voxel_size)

    def _publish_reconstruction(self):
        """Publish reconstructed point cloud"""
        if len(self.global_pcd.points) == 0:
            return

        points_array = np.asarray(self.global_pcd.points)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"

        augmented_pointcloud = pc2.create_cloud(header, fields, points_array)
        self.reconstruction_pub.publish(augmented_pointcloud)

    def _publish_mesh(self):
        """Create and publish mesh"""
        if len(self.global_pcd.points) < 100:
            return

        self.point_count += 1
        if self.point_count % self.mesh_interval != 0:
            return

        try:
            pcd = self.global_pcd.voxel_down_sample(voxel_size=0.001)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            if len(pcd.points) < 100:
                return

            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=10)

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, self.alpha
            )
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            if len(triangles) == 0:
                return

            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "reconstructed_mesh"
            marker.id = 0
            marker.type = Marker.TRIANGLE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.8
            marker.color.b = 0.3
            marker.color.a = 1.0

            for triangle in triangles:
                for vertex_idx in triangle:
                    point = Point()
                    point.x = vertices[vertex_idx, 0]
                    point.y = vertices[vertex_idx, 1]
                    point.z = vertices[vertex_idx, 2]
                    marker.points.append(point)

            marker_array = MarkerArray()
            marker_array.markers.append(marker)
            self.mesh_pub.publish(marker_array)

        except Exception as e:
            if "operator()" in str(e):
                self.alpha = max(0.01, self.alpha * 0.9)
            rospy.logwarn(f"Mesh generation error: {e}")

    def _publish_cad_features(self):
        """Publish CAD features (placeholder)"""
        if len(self.global_pcd.points) < 100:
            return
        # Placeholder for CAD feature extraction
        rospy.logdebug("CAD feature extraction not implemented")
