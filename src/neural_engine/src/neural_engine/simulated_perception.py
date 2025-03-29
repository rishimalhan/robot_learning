#! /usr/bin/env python3

# External

import torch
import numpy as np
import os
import rospkg
import rospy
import struct
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftPhongShader,
    TexturesVertex,
    PointLights,
    BlendParams,
)
from pytorch3d.structures import Meshes
from tf.transformations import (
    euler_matrix,
    quaternion_from_euler,
    euler_from_quaternion,
)
import matplotlib.pyplot as plt
import tempfile
from moveit_commander import PlanningSceneInterface
import tf2_ros
import geometry_msgs.msg
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import cv2
from cv_bridge import CvBridge
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import sys
import time
import signal
import threading
import atexit

# Internal

from neural_engine.utils import bootstrap, get_param

bootstrap()

# Global flag for clean shutdown
shutdown_flag = threading.Event()


def cleanup_handler():
    """Cleanup handler to ensure proper shutdown"""
    rospy.loginfo("Cleaning up resources...")
    shutdown_flag.set()
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Clear MPS cache if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def signal_handler(signum, frame):
    """Signal handler for graceful shutdown"""
    rospy.loginfo(f"Received signal {signum}")
    cleanup_handler()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
# Register cleanup on normal exit
atexit.register(cleanup_handler)


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, fov_degrees: float):
        """Create intrinsics from field of view."""
        fx = fy = (width / 2) / np.tan(np.radians(fov_degrees) / 2)
        cx = width / 2
        cy = height / 2
        return cls(width, height, fx, fy, cx, cy)

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def to_camera_info(self, frame_id: str) -> CameraInfo:
        """Convert to ROS CameraInfo message."""
        camera_info = CameraInfo()
        camera_info.header.frame_id = frame_id
        camera_info.height = self.height
        camera_info.width = self.width
        camera_info.distortion_model = "plumb_bob"
        camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
        camera_info.K = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        camera_info.R = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]  # Identity rotation
        camera_info.P = [
            self.fx,
            0.0,
            self.cx,
            0.0,
            0.0,
            self.fy,
            self.cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return camera_info


class SimulatedCamera:
    """A simulated camera using PyTorch3D rendering."""

    def __init__(
        self,
        device: str = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """Initialize a simulated camera."""
        rospy.loginfo("Initializing SimulatedCamera...")
        rospy.loginfo(f"Using device: {device}")

        # Get camera configuration from parameter server
        self.fov_config = get_param("/perception/camera/field_of_view")
        self.sensing_config = get_param("/perception/camera/sensing")
        self.image_size = get_param("/perception/camera/image_size")
        rospy.loginfo(
            f"Camera config - FOV: {self.fov_config}, Image size: {self.image_size}"
        )

        # Configure from parameters
        self.fov_degrees = self.fov_config.get("horizontal_fov", 60.0)
        self.min_range = self.fov_config.get("min_range", 0.1)
        self.max_range = self.fov_config.get("max_range", 1.0)
        self.noise_level = self.sensing_config.get("noise_level", 0.0)
        rospy.loginfo(
            f"Camera params - FOV: {self.fov_degrees}°, Range: [{self.min_range}, {self.max_range}]m, Noise: {self.noise_level}"
        )

        # Set device and initialize other attributes
        self.device = device
        self.temp_dir = tempfile.mkdtemp()
        self.mesh = None
        self.camera_frame_id = get_param("/perception/camera/frame", "tool0")
        self.bridge = CvBridge()

        self.shader_type = rospy.get_param(
            "/perception/camera/rendering/shader_type", "soft_phong"
        )
        rospy.loginfo(f"Using shader type: {self.shader_type}")

        # Create intrinsics
        self.intrinsics = CameraIntrinsics.from_fov(
            width=self.image_size[1],
            height=self.image_size[0],
            fov_degrees=self.fov_degrees,
        )
        rospy.loginfo(
            f"Camera intrinsics created: fx={self.intrinsics.fx}, fy={self.intrinsics.fy}, cx={self.intrinsics.cx}, cy={self.intrinsics.cy}"
        )

        # Rasterizer settings
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
        )
        rospy.loginfo(
            f"Rasterizer settings configured with image size: {self.image_size}"
        )

        # Create default lighting
        self.lights = PointLights(
            device=device,
            location=[[0.0, 0.0, -3.0]],
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.7, 0.7, 0.7),),
            specular_color=((0.3, 0.3, 0.3),),
        )
        rospy.loginfo("Lighting configured")

        # Blend parameters for rendering
        self.blend_params = BlendParams(
            sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0)
        )

        # Initialize TF buffer and listener
        rospy.loginfo("Initializing TF listener...")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize publishers
        rospy.loginfo("Setting up ROS publishers...")
        self.rgb_pub = rospy.Publisher("camera/rgb/image_raw", Image, queue_size=1)
        self.rgb_info_pub = rospy.Publisher(
            "camera/rgb/camera_info", CameraInfo, queue_size=1
        )
        self.depth_pub = rospy.Publisher("camera/depth/image_raw", Image, queue_size=1)
        self.depth_info_pub = rospy.Publisher(
            "camera/depth/camera_info", CameraInfo, queue_size=1
        )
        self.pointcloud_pub = rospy.Publisher(
            "camera/points", PointCloud2, queue_size=1
        )

        # Create base camera with identity transform
        rospy.loginfo("Creating base camera...")
        self.base_camera = FoVPerspectiveCameras(
            R=torch.eye(3, device=self.device).unsqueeze(0),
            T=torch.zeros(3, device=self.device).unsqueeze(0),
            fov=self.fov_degrees,
            znear=self.min_range,
            zfar=self.max_range,
            device=self.device,
        )

        # Create renderer with base camera
        rospy.loginfo("Creating renderer...")
        self.renderer = self._create_renderer(self.base_camera)

        rospy.loginfo("Loading mesh from scene...")
        self.mesh = self._load_mesh_from_scene()
        rospy.loginfo("SimulatedCamera initialization complete")

    def _load_mesh_from_scene(self) -> Meshes:
        """Load a mesh directly from the planning scene."""
        rospy.loginfo("Loading mesh from planning scene...")

        # Get object name from parameter server
        mesh_config = rospy.get_param("/perception/mesh", {})
        object_name = mesh_config.get("scene_object_name", "part")
        rospy.loginfo(f"Looking for object: {object_name}")

        # Connect to planning scene and wait for connection
        scene = PlanningSceneInterface()
        rospy.sleep(1.0)
        rospy.loginfo("Connected to planning scene")

        # Get object from scene
        scene_objects = scene.get_objects()
        if not scene_objects:
            rospy.logerr("No objects found in planning scene")
            raise ValueError("No objects found in planning scene")

        rospy.loginfo(f"Found objects in scene: {list(scene_objects.keys())}")

        obj = scene_objects.get(object_name)
        if not obj:
            rospy.logerr(f"Required object '{object_name}' not found")
            raise ValueError(
                f"Required object '{object_name}' not found. Available: {', '.join(scene_objects.keys())}"
            )

        # Validate mesh data exists
        if not getattr(obj, "meshes", None):
            rospy.logerr(f"Object '{object_name}' has no mesh data")
            raise ValueError(f"Object '{object_name}' has no mesh data")

        mesh_data = obj.meshes[0]
        rospy.loginfo("Processing mesh vertices and faces...")

        # Extract vertices and faces efficiently using list comprehension
        verts = [[float(p.x), float(p.y), float(p.z)] for p in mesh_data.vertices]
        faces = [[int(idx) for idx in t.vertex_indices] for t in mesh_data.triangles]

        rospy.loginfo(f"Mesh has {len(verts)} vertices and {len(faces)} faces")

        if not verts or not faces:
            rospy.logerr(f"Object '{object_name}' has empty mesh data")
            raise ValueError(f"Object '{object_name}' has empty mesh data")

        # Convert to tensors
        rospy.loginfo("Converting mesh data to tensors...")
        verts_tensor = torch.tensor(verts, dtype=torch.float32, device=self.device)
        faces_tensor = torch.tensor(faces, dtype=torch.int64, device=self.device)

        # Create textures
        verts_rgb = torch.full_like(verts_tensor, 0.7, device=self.device)
        textures = TexturesVertex(verts_features=verts_rgb.unsqueeze(0))
        rospy.loginfo("Created vertex textures")

        # Get pose data
        if not getattr(obj, "mesh_poses", None):
            rospy.logerr(f"Object '{object_name}' has no mesh pose data")
            raise ValueError(f"Object '{object_name}' has no mesh pose data")

        pose = obj.mesh_poses[0]
        translation = torch.tensor(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=torch.float32,
            device=self.device,
        )
        rospy.loginfo(
            f"Mesh translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]"
        )

        # Convert quaternion to rotation matrix directly
        quat = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        rot_matrix = np.array(
            euler_matrix(*euler_from_quaternion(quat), "sxyz")[:3, :3], dtype=np.float32
        )
        rospy.loginfo("Applied rotation matrix to mesh")

        # Apply transformations efficiently
        if not np.allclose(rot_matrix, np.eye(3)):
            rot_tensor = torch.tensor(
                rot_matrix, dtype=torch.float32, device=self.device
            )
            verts_tensor = torch.matmul(verts_tensor, rot_tensor.transpose(0, 1))

        verts_tensor += translation
        rospy.loginfo("Applied transformations to mesh vertices")

        # Create and return mesh
        mesh = Meshes(verts=[verts_tensor], faces=[faces_tensor], textures=textures)
        rospy.loginfo("Successfully created PyTorch3D mesh")
        return mesh

    def _update_camera_transform(self, camera_transform: np.ndarray):
        """Update the camera transform."""
        rospy.loginfo("Updating camera transform...")
        # Convert 4x4 numpy to 3x3 rotation + 3x1 translation for PyTorch3D camera
        R = torch.tensor(camera_transform[:3, :3], device=self.device).unsqueeze(0)
        T = torch.tensor(camera_transform[:3, 3], device=self.device).unsqueeze(0)

        # Update camera parameters
        self.base_camera.R = R
        self.base_camera.T = T
        rospy.loginfo(f"Camera transform updated - Translation: {T.cpu().numpy()}")

    def _create_renderer(self, cameras: FoVPerspectiveCameras) -> MeshRenderer:
        """Create a renderer with the specified shader type."""
        rospy.loginfo(f"Creating renderer with {self.shader_type} shader...")
        rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=self.raster_settings
        )

        if self.shader_type == "hard_phong":
            shader = HardPhongShader(
                device=self.device, cameras=cameras, lights=self.lights
            )
        else:  # Default to soft_phong
            shader = SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=self.lights,
                blend_params=self.blend_params,
            )

        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        rospy.loginfo("Renderer created successfully")
        return renderer

    def _render_scene(self, camera_transform: np.ndarray):
        """Render the scene from the given camera transform."""
        rospy.loginfo("Rendering scene...")

        # Update camera transform
        self._update_camera_transform(camera_transform.astype(np.float32))

        # Render RGB
        rospy.loginfo("Rendering RGB image...")
        rgba = self.renderer(self.mesh)
        rgb = rgba[0, ..., :3].detach().cpu().numpy().astype(np.float32)
        rospy.loginfo(
            f"RGB image shape: {rgb.shape}, range: [{rgb.min():.2f}, {rgb.max():.2f}]"
        )

        # Render depth
        rospy.loginfo("Rendering depth image...")
        fragments = self.renderer.rasterizer(self.mesh)
        depth = fragments.zbuf[0, ..., 0].detach().cpu().numpy().astype(np.float32)
        rospy.loginfo(
            f"Raw depth shape: {depth.shape}, range: [{depth.min():.2f}, {depth.max():.2f}]"
        )

        # Process depth
        mask = (depth > 0) & (depth < self.max_range)
        depth = np.where(mask, depth, 0).astype(np.float32)
        rospy.loginfo(f"Valid depth points: {np.sum(mask)}")

        if self.noise_level > 0:
            rospy.loginfo(f"Adding noise with level {self.noise_level}")
            noise = np.random.normal(0, self.noise_level, depth.shape).astype(
                np.float32
            )
            depth[mask] += noise[mask]
            depth = np.where((depth > 0) & (depth < self.max_range), depth, 0).astype(
                np.float32
            )
            rospy.loginfo(
                f"Depth range after noise: [{depth.min():.2f}, {depth.max():.2f}]"
            )

        return rgb, depth

    def create_pointcloud(
        self,
        depth: np.ndarray,
        return_colors: bool = False,
        rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Create a point cloud from a depth image."""
        rospy.loginfo("Creating point cloud from depth image...")
        H, W = depth.shape
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.cx, self.intrinsics.cy

        # Generate pixel grid
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x = (x - cx) / fx
        y = (y - cy) / fy
        z = depth

        # Convert to 3D points
        X = x * z
        Y = y * z
        Z = z

        # Filter out invalid points (zero depth)
        valid = z > 0
        xyz = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
        rospy.loginfo(f"Generated {len(xyz)} valid 3D points")

        if return_colors and rgb is not None:
            colors = rgb[valid]
            rospy.loginfo(f"Added colors to point cloud")
            return xyz, colors

        return xyz

    def get_camera_transform(self):
        """Get the 4x4 camera transform matrix in world coordinates."""
        try:
            # Get transform from world to camera frame
            transform = self.tf_buffer.lookup_transform(
                "world", self.camera_frame_id, rospy.Time(0), rospy.Duration(0.01)
            )

            # Create 4x4 transform matrix directly from translation and quaternion
            trans = transform.transform.translation
            rot = transform.transform.rotation

            transform_matrix = np.eye(4, dtype=np.float32)
            transform_matrix[:3, :3] = np.array(
                euler_matrix(*euler_from_quaternion([rot.x, rot.y, rot.z, rot.w]))[
                    :3, :3
                ],
                dtype=np.float32,
            )
            transform_matrix[:3, 3] = [trans.x, trans.y, trans.z]

            rospy.loginfo(
                f"Got camera transform from TF: {trans.x:.3f}, {trans.y:.3f}, {trans.z:.3f}"
            )
            return transform_matrix

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"Could not get camera transform from TF: {e}")

            # Create a default transform
            transform_matrix = np.eye(4, dtype=np.float32)
            # Position in front of the scene
            transform_matrix[:3, 3] = np.array([1.0, 0.0, 1.0], dtype=np.float32)

            return transform_matrix

    def publish_camera_data(self):
        """Render and publish camera data based on current scene and camera position."""
        if self.mesh is None:
            rospy.logwarn("No mesh loaded, cannot publish camera data")
            return

        rospy.loginfo("Publishing camera data...")

        # Get camera transform and render scene
        camera_transform = self.get_camera_transform()
        rgb, depth = self._render_scene(camera_transform)
        rospy.loginfo("Scene rendered successfully")

        # Get current timestamp
        timestamp = rospy.Time.now()

        # Prepare camera info
        camera_info = self.intrinsics.to_camera_info(self.camera_frame_id)
        camera_info.header.stamp = timestamp

        # Publish RGB image
        rospy.loginfo("Publishing RGB image...")
        rgb_msg = self.bridge.cv2_to_imgmsg(
            cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            encoding="bgr8",
        )
        rgb_msg.header.frame_id = self.camera_frame_id
        rgb_msg.header.stamp = timestamp
        self.rgb_pub.publish(rgb_msg)
        self.rgb_info_pub.publish(camera_info)

        # Publish depth image
        rospy.loginfo("Publishing depth image...")
        depth_msg = self.bridge.cv2_to_imgmsg(
            depth.astype(np.float32), encoding="32FC1"
        )
        depth_msg.header.frame_id = self.camera_frame_id
        depth_msg.header.stamp = timestamp
        self.depth_pub.publish(depth_msg)
        self.depth_info_pub.publish(camera_info)

        # Create and publish point cloud
        rospy.loginfo("Creating point cloud...")
        pointcloud, colors = self.create_pointcloud(depth, return_colors=True, rgb=rgb)
        rospy.loginfo(f"Point cloud created with {len(pointcloud)} points")

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        if len(pointcloud) > 0:
            rospy.loginfo("Packing RGB colors for point cloud...")
            # Pack RGB into a single float32
            rgb_packed = np.zeros(len(colors), dtype=np.float32)
            for i in range(len(colors)):
                r, g, b = colors[i]
                rgb_packed[i] = struct.unpack(
                    "I",
                    struct.pack("BBBB", int(b * 255), int(g * 255), int(r * 255), 0),
                )[0]

            cloud_points = np.hstack([pointcloud, rgb_packed.reshape(-1, 1)])
            rospy.loginfo(f"Point cloud data shape: {cloud_points.shape}")
        else:
            rospy.logwarn("No valid points in point cloud, creating dummy point")
            cloud_points = np.array([[0.0, 0.0, 1000.0, 0.0]], dtype=np.float32)

        cloud_msg = pc2.create_cloud(
            std_msgs.msg.Header(frame_id=self.camera_frame_id, stamp=timestamp),
            fields,
            cloud_points,
        )
        rospy.loginfo("Publishing point cloud...")
        self.pointcloud_pub.publish(cloud_msg)

    def cleanup(self):
        """Clean up resources."""
        rospy.loginfo("Cleaning up SimulatedCamera resources...")
        # Clear CUDA cache if using CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()
        # Clear MPS cache if using MPS
        elif self.device == "mps":
            torch.mps.empty_cache()

        # Delete temporary directory
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            import shutil

            rospy.loginfo("Removing temporary directory...")
            shutil.rmtree(self.temp_dir)

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def run_camera_node():
    """Run the simulated camera as a ROS node."""
    # Initialize ROS node
    rospy.init_node("simulated_rgbd_camera")

    # Get rate parameter
    rate_hz = get_param("/perception/camera/sensing/frame_rate", 10)
    rospy.loginfo(f"Publishing at {rate_hz} Hz")

    camera = None
    try:
        # Create camera and load configuration
        camera = SimulatedCamera()
        # Set update rate
        rate = rospy.Rate(rate_hz)
        rospy.loginfo("Simulated RGBD camera node is running")

        while not rospy.is_shutdown() and not shutdown_flag.is_set():
            try:
                # Publish camera data
                camera.publish_camera_data()

                # Log occasionally to show activity
                if rospy.get_time() % 5 < 0.1:  # Log every ~5 seconds
                    rospy.loginfo("Publishing camera data...")

                # Sleep to maintain rate
                rate.sleep()
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr(f"Error in camera node: {e}")
                time.sleep(0.1)  # Short sleep on error
    except Exception as e:
        rospy.logerr(f"Error initializing camera: {e}")
    finally:
        if camera:
            rospy.loginfo("Cleaning up camera resources...")
            camera.cleanup()
        rospy.loginfo("Camera node shutting down")


if __name__ == "__main__":
    # Set up exception handling for the entire script
    try:
        run_camera_node()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted by user, shutting down")
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received, shutting down")
    except Exception as e:
        rospy.logerr(f"Unhandled exception: {e}")
    finally:
        cleanup_handler()
        sys.exit(0)
