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
    def __init__(self):
        # Initialize collections to store point clouds
        self.pointcloud_buffer = deque(maxlen=1000)
        self.latest_pointcloud = None
        self.augmented_pointcloud = None

        # Initialize Open3D objects for point cloud management
        self.global_pcd = o3d.geometry.PointCloud()

        # Parameters for point cloud processing
        self.voxel_size = 0.001  # 1mm voxel size for deduplication

        # Subscribe to the camera pointcloud topic
        self.pointcloud_sub = rospy.Subscriber(
            "camera/points", PointCloud2, self.pointcloud_callback, queue_size=1
        )

        # Set up publisher for the augmented pointcloud
        self.reconstruction_pub = rospy.Publisher(
            "reconstruction", PointCloud2, queue_size=1
        )

        # Set up publisher for the reconstructed mesh
        self.mesh_pub = rospy.Publisher("reconstructed_mesh", MarkerArray, queue_size=1)

        # Mesh reconstruction parameters
        self.alpha = 1.0  # Alpha value for Alpha shape reconstruction
        self.downsample_voxel_size = (
            0.001  # Downsample to 1mm voxels before mesh creation
        )
        self.min_mesh_points = 100  # Minimum points needed for mesh reconstruction
        self.mesh_color = [0.0, 0.8, 0.3, 1.0]  # RGBA (green)
        self.mesh_publish_interval = 10
        self.point_count = 0

        # Suppress Open3D console output to avoid flooding
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

        rospy.loginfo(
            "PointCloud processor initialized with Open3D-based point storage"
        )

    def pointcloud_callback(self, cloud_msg):
        """Process incoming pointcloud data using efficient point cloud management"""
        # Store the message
        self.latest_pointcloud = cloud_msg

        # Convert PointCloud2 to a numpy array of points
        points = pc2.read_points(cloud_msg, skip_nans=True)
        points_array = np.array(list(points))

        if len(points_array) == 0:
            rospy.logwarn("Received empty pointcloud")
            return

        # Convert numpy array to Open3D point cloud
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points_array[:, :3])

        # Add to buffer for reference
        self.pointcloud_buffer.append(points_array)

        # Integrate the new points with the global point cloud
        self._integrate_point_cloud(new_pcd)

        # Create augmented pointcloud
        self.create_augmented_pointcloud()

    def _integrate_point_cloud(self, new_pcd):
        """Integrate new points into the global point cloud using voxel grid merging"""
        # If global point cloud is empty, just use the new points
        if len(self.global_pcd.points) == 0:
            self.global_pcd = new_pcd
            rospy.loginfo(
                f"Initialized global point cloud with {len(new_pcd.points)} points"
            )
            return

        # Combine global and new point clouds
        combined_pcd = self.global_pcd + new_pcd

        # Use voxel downsampling to merge nearby points
        # This maintains the point density without duplicating points
        self.global_pcd = combined_pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Log the result
        rospy.logdebug(
            f"Integrated {len(new_pcd.points)} new points. Global cloud now has {len(self.global_pcd.points)} points"
        )

    def _get_persistent_points_as_numpy(self):
        """Convert the global point cloud to a numpy array for publishing"""
        if len(self.global_pcd.points) == 0:
            return None

        # Convert Open3D point cloud to numpy array
        return np.asarray(self.global_pcd.points)

    def create_augmented_pointcloud(self):
        """Create an augmented pointcloud from the global point cloud and generate mesh"""
        # Get points from global point cloud
        points_array = self._get_persistent_points_as_numpy()

        if points_array is None or len(points_array) == 0:
            rospy.logwarn("No points available to create augmented pointcloud")
            return

        # Create pointcloud fields (XYZ)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Create header with world frame
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"  # Explicitly set to world frame

        # Create PointCloud2 message from persistent points
        self.augmented_pointcloud = pc2.create_cloud(header, fields, points_array)

        # Publish the augmented pointcloud
        self.reconstruction_pub.publish(self.augmented_pointcloud)

        # Create and publish mesh from the point cloud
        self.create_and_publish_mesh()

    def create_and_publish_mesh(self):
        """Create a mesh from the point cloud and publish it as markers"""
        if len(self.global_pcd.points) < self.min_mesh_points:
            return

        # Only regenerate mesh periodically to avoid overwhelming RViz and console
        self.point_count += 1
        if self.point_count % self.mesh_publish_interval != 0:
            return

        try:
            # Make a copy of the global point cloud to avoid modifying it
            pcd = self.global_pcd.voxel_down_sample(
                voxel_size=self.downsample_voxel_size
            )

            # Statistical outlier removal to clean up the point cloud
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            # Ensure we still have enough points after filtering
            if len(pcd.points) < self.min_mesh_points:
                rospy.logwarn(f"Too few points after preprocessing: {len(pcd.points)}")
                return

            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=10)

            # Use Alpha shape reconstruction for the visible surfaces
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, self.alpha
            )

            # Clean up the mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            # Create marker array
            marker_array = MarkerArray()

            # Create mesh marker
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "reconstructed_mesh"
            marker.id = 0
            marker.type = Marker.TRIANGLE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color.r = self.mesh_color[0]
            marker.color.g = self.mesh_color[1]
            marker.color.b = self.mesh_color[2]
            marker.color.a = self.mesh_color[3]

            # Add triangles to marker
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            # Skip if no triangles were created
            if len(triangles) == 0:
                rospy.logwarn("No triangles generated in the mesh")
                return

            # Add all triangles to the marker
            for triangle in triangles:
                for vertex_idx in triangle:
                    point = Point()
                    point.x = vertices[vertex_idx, 0]
                    point.y = vertices[vertex_idx, 1]
                    point.z = vertices[vertex_idx, 2]
                    marker.points.append(point)

            # Add marker to array
            marker_array.markers.append(marker)

            # Publish marker array
            self.mesh_pub.publish(marker_array)
            rospy.logdebug(
                f"Published reconstructed mesh with {len(triangles)} triangles"
            )

        except RuntimeError as e:
            # Handle Open3D specific errors
            error_msg = str(e)
            if "operator()" in error_msg or "Failed to close loop" in error_msg:
                rospy.logwarn(
                    "Open3D mesh construction error - using different alpha value next time"
                )
                # Adjust alpha value slightly to avoid the same error next time
                self.alpha = max(0.01, self.alpha * 0.9)
            else:
                rospy.logwarn(f"Error creating mesh: {e}")
        except Exception as e:
            rospy.logwarn(f"Error creating mesh: {e}")
