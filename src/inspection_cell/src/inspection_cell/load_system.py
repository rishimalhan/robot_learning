#!/usr/bin/env python3

# External

import rospy
import os
from tf.transformations import quaternion_from_euler, quaternion_multiply
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
import rospkg
import moveit_msgs.srv
import moveit_msgs.msg
from moveit_msgs.msg import (
    PlanningScene,
    AllowedCollisionEntry,
    ObjectColor,
)
from moveit_msgs.srv import ApplyPlanningScene
from std_msgs.msg import ColorRGBA
import tf2_ros
import geometry_msgs.msg
import ros_numpy
import numpy as np

# Internal

from inspection_cell.utils import (
    resolve_package_path,
    load_yaml_to_params,
    get_param,
)
from inspection_cell.collision_checker import CollisionCheck
from inspection_cell.planner import Planner
from inspection_cell.executor import Executor


class EnvironmentLoader:
    def __init__(self, move_group_name="manipulator", clear_scene=True):
        # Initialize ROS node only if it hasn't been already
        if not rospy.core.is_initialized():
            rospy.init_node("environment_loader", anonymous=True)
        else:
            rospy.loginfo("Using existing ROS node")

        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander(move_group_name)
        self.robot = RobotCommander()
        self.group_name = move_group_name
        self.planner = Planner()
        self.executor = Executor()

        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Create RosPack instance
        self.rospack = rospkg.RosPack()

        # Get the planning frame
        self.planning_frame = self.move_group.get_planning_frame()
        rospy.loginfo(
            f"Planning frame or coordinate frame in which motion planning is performed: {self.planning_frame}"
        )

        # Initialize TCP transform cache
        self.tcp_transform = None

        # Clear the planning scene if requested
        if clear_scene:
            self.clear_scene()
        # Load environment configuration first
        self._load_config()
        # Attach tool if configured
        self._attach_tool()
        # Ensure TCP transform is set up
        tool_config = get_param("/environment/tool", None)
        if tool_config and "tcp" in tool_config:
            self._setup_tcp_frame(tool_config)
        else:
            rospy.logwarn("No TCP configuration found in tool config")
        # Fix any self-collision issues
        self._ensure_acm_applied()
        # Add objects to scene
        self._add_objects_to_scene()

        # Wait for scene to update
        rospy.sleep(1.0)

        # Print current scene objects
        self._print_scene_objects()

        # Print the ACM status for debugging
        self.print_acm_status()

        # Initialize the optimized collision checker AFTER environment is fully loaded
        rospy.loginfo("Initializing collision checker after environment setup...")
        self.collision_checker = CollisionCheck(move_group_name=move_group_name)
        rospy.loginfo("Collision checker initialized")

    def _load_config(self):
        """Load environment configuration from YAML file and store in parameter server."""
        config_path = os.path.join(
            self.rospack.get_path("inspection_cell"), "config", "environment.yaml"
        )
        rospy.loginfo(f"Loading configuration from: {config_path}")

        # Load YAML to parameter server
        self.config = load_yaml_to_params(config_path, "/environment")
        rospy.loginfo("Environment configuration loaded to parameter server")

    def clear_scene(self):
        """Clear all objects from the planning scene."""
        rospy.loginfo("Clearing planning scene...")
        # Get all objects in the scene
        objects = self.scene.get_objects()
        # Remove each object
        for obj in objects:
            self.scene.remove_world_object(obj)
        rospy.loginfo("Planning scene cleared")

    def _print_scene_objects(self):
        """Print all objects in the current planning scene."""
        rospy.loginfo("Current objects in planning scene:")
        for obj in self.scene.get_objects():
            rospy.loginfo(f"  - {obj}")

    def _configure_robot(self):
        """Configure robot parameters from config file."""
        robot_config = get_param("/environment/robot", {})
        self.move_group.set_planning_time(robot_config.get("planning_time", 2.0))
        self.move_group.set_num_planning_attempts(
            robot_config.get("num_planning_attempts", 5)
        )
        self.move_group.set_max_velocity_scaling_factor(
            robot_config.get("max_velocity_scaling_factor", 1.0)
        )
        self.move_group.set_max_acceleration_scaling_factor(
            robot_config.get("max_acceleration_scaling_factor", 1.0)
        )
        rospy.loginfo("Robot parameters configured")

    def _add_objects_to_scene(self):
        """Add objects to the planning scene based on configuration."""
        objects = get_param("/environment/objects", {})
        rospy.loginfo(f"Adding {len(objects)} objects to scene")

        # Create planning scene publisher
        planning_scene_pub = rospy.Publisher(
            "/planning_scene", PlanningScene, queue_size=10
        )
        rospy.sleep(0.5)  # Allow publisher to initialize

        for name, obj in objects.items():
            self._add_single_object(name, obj, planning_scene_pub)

        # Wait for the scene to update
        rospy.sleep(1.0)

    def _add_single_object(self, name, obj, publisher):
        """Add a single object to the planning scene.

        Args:
            name: Name of the object
            obj: Object configuration
            publisher: PlanningScene publisher
        """
        rospy.loginfo(f"Adding object: {name} of type {obj['type']}")

        # Create a planning scene message
        planning_scene = PlanningScene()
        planning_scene.is_diff = True

        # Create collision object
        collision_object = self._create_collision_object(name, obj)

        # Add collision object to planning scene if created successfully
        if collision_object:
            planning_scene.world.collision_objects.append(collision_object)

            # Set color if specified
            self._set_object_color(name, obj, planning_scene)

            # Publish planning scene
            publisher.publish(planning_scene)
            rospy.loginfo(f"Published object: {name} to planning scene")

            # Configure collision checking
            self._configure_object_collision(name, obj)

    def _create_collision_object(self, name, obj):
        """Create a collision object based on the object configuration.

        Args:
            name: Name of the object
            obj: Object configuration

        Returns:
            CollisionObject or None if creation failed
        """
        collision_object = CollisionObject()
        collision_object.header.frame_id = self.planning_frame
        collision_object.id = name
        collision_object.operation = CollisionObject.ADD

        # Handle different object types
        if obj["type"] in ["box", "sphere", "cylinder"]:
            self._add_primitive_to_object(obj, collision_object)
        elif obj["type"] == "mesh":
            success = self._add_mesh_to_object(obj, collision_object, name)
            if not success:
                return None
        else:
            rospy.logerr(f"Unsupported object type: {obj['type']}")
            return None

        return collision_object

    def _add_primitive_to_object(self, obj, collision_object):
        """Add a primitive shape to a collision object.

        Args:
            obj: Object configuration
            collision_object: CollisionObject to add the primitive to
        """
        # Create primitive
        primitive = SolidPrimitive()
        primitive.type = self._get_primitive_type(obj["type"])

        if obj["type"] == "box":
            primitive.dimensions = obj["dimensions"]
            rospy.loginfo(f"  Box dimensions: {obj['dimensions']}")
        elif obj["type"] == "sphere":
            primitive.dimensions = [obj["radius"]]
            rospy.loginfo(f"  Sphere radius: {obj['radius']}")
        elif obj["type"] == "cylinder":
            primitive.dimensions = [obj["height"], obj["radius"]]
            rospy.loginfo(
                f"  Cylinder height: {obj['height']}, radius: {obj['radius']}"
            )

        # Add the primitive to the collision object
        collision_object.primitives.append(primitive)

        # Create and add the pose
        pose = self._create_pose_from_config(obj["pose"])
        collision_object.primitive_poses.append(pose)

        # Log the pose
        euler = obj["pose"]["orientation"]
        rospy.loginfo(
            f"  Pose: position={obj['pose']['position']}, orientation={euler}"
        )

    def _add_mesh_to_object(self, obj, collision_object, name):
        """Add a mesh to a collision object.

        Args:
            obj: Object configuration
            collision_object: CollisionObject to add the mesh to
            name: Name of the object for error reporting

        Returns:
            bool: True if mesh was added successfully, False otherwise
        """
        mesh_path = obj.get("mesh_path")
        if not mesh_path:
            rospy.logerr(f"Mesh path not specified for object {name}")
            return False

        # Resolve package:// prefix
        mesh_path = resolve_package_path(mesh_path)
        rospy.loginfo(f"Loading mesh from: {mesh_path}")

        try:
            import meshio

            # Load the mesh file
            mesh = meshio.read(mesh_path)

            # Create mesh for collision object
            from shape_msgs.msg import Mesh, MeshTriangle

            co_mesh = Mesh()
            scale = obj.get("scale", [1.0, 1.0, 1.0])

            # Add vertices
            for point in mesh.points:
                vertex = Point()
                vertex.x = point[0] * scale[0]
                vertex.y = point[1] * scale[1]
                vertex.z = point[2] * scale[2]
                co_mesh.vertices.append(vertex)

            # Add triangles
            for cell in mesh.cells:
                if cell.type == "triangle":
                    for indices in cell.data:
                        triangle = MeshTriangle()
                        triangle.vertex_indices = [
                            int(indices[0]),
                            int(indices[1]),
                            int(indices[2]),
                        ]
                        co_mesh.triangles.append(triangle)

            if not co_mesh.triangles:
                rospy.logerr(f"No triangles found in mesh {mesh_path}")
                return False

            # Add mesh to collision object
            collision_object.meshes.append(co_mesh)

            # Add mesh pose
            pose = self._create_pose_from_config(obj["pose"])
            collision_object.mesh_poses.append(pose)

            return True

        except Exception as e:
            rospy.logerr(f"Failed to load mesh {mesh_path}: {e}")
            return False

    def _create_pose_from_config(self, pose_config):
        """Create a Pose message from pose configuration.

        Args:
            pose_config: Pose configuration with position and orientation

        Returns:
            Pose: The created pose message
        """
        pose = Pose()
        pose.position = Point(*pose_config["position"])
        euler = pose_config["orientation"]
        q = quaternion_from_euler(*euler)
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pose

    def _set_object_color(self, name, obj, planning_scene):
        """Set the color of an object in the planning scene.

        Args:
            name: Name of the object
            obj: Object configuration
            planning_scene: PlanningScene message to add the color to
        """
        color = obj.get("color")
        alpha = obj.get("alpha", 1.0)
        if color:
            object_color = ObjectColor()
            object_color.id = name
            object_color.color = ColorRGBA(color[0], color[1], color[2], alpha)
            planning_scene.object_colors.append(object_color)

    def _configure_object_collision(self, name, obj):
        """Configure collision checking for an object.

        Args:
            name: Name of the object
            obj: Object configuration
        """
        # Disable collision if needed
        if not obj.get("collision_enabled", True):
            rospy.loginfo(
                f"Object '{name}' has collision_enabled=False, disabling collision checking"
            )
            self._disable_collision_for_object(name)
        else:
            rospy.loginfo(
                f"Object '{name}' has collision_enabled=True, collision checking is active"
            )

    def _disable_collision_for_object(self, object_id):
        """
        Disable collision checking between the specified object and all other objects/links.

        In the AllowedCollisionMatrix:
        - True = collisions are ALLOWED (ignored/disabled)
        - False = collisions are NOT ALLOWED (checked/enabled)

        This method sets entries to TRUE to ALLOW collisions (disable checking).
        """
        rospy.loginfo(
            f"Configuring collision checking for object: {object_id} (setting to IGNORE collisions)"
        )

        # Get the current ACM
        current_acm = self._get_current_acm()

        # Update the ACM for the object
        updated_acm = self._update_acm_for_object(current_acm, object_id)

        # Apply the updated ACM
        success = self._apply_acm(updated_acm)

        rospy.loginfo(
            f"Collision checking disabled for object: {object_id}, success: {success}"
        )

    def _get_current_acm(self):
        """Get the current Allowed Collision Matrix from the planning scene.

        Returns:
            AllowedCollisionMatrix: The current ACM
        """
        # Wait for services
        rospy.wait_for_service("/get_planning_scene")
        get_planning_scene = rospy.ServiceProxy(
            "/get_planning_scene", moveit_msgs.srv.GetPlanningScene
        )

        # Create request for ACM
        request = moveit_msgs.srv.GetPlanningSceneRequest()
        request.components.components = (
            moveit_msgs.msg.PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        )

        # Get the current ACM
        response = get_planning_scene(request)
        return response.scene.allowed_collision_matrix

    def _update_acm_for_object(self, current_acm, object_id):
        """Update the ACM for a specific object.

        Args:
            current_acm: Current AllowedCollisionMatrix
            object_id: ID of the object to update

        Returns:
            AllowedCollisionMatrix: The updated ACM
        """
        # Add object to ACM if not already there
        if object_id not in current_acm.entry_names:
            current_acm.entry_names.append(object_id)

            # Create new entry that allows collision with everything (TRUE = ignore collisions)
            entry = AllowedCollisionEntry()
            entry.enabled = [True] * len(
                current_acm.entry_names
            )  # Set TRUE to ALLOW collisions with everything (disable checking)
            current_acm.entry_values.append(entry)

            # Update all existing entries to allow collision with the new object
            for i in range(
                len(current_acm.entry_values) - 1
            ):  # -1 because we just added one
                current_acm.entry_values[i].enabled.append(True)
        else:
            # Object already exists in ACM - find its index
            idx = current_acm.entry_names.index(object_id)

            # Set all collisions with this object to be ALLOWED (TRUE = ignore collisions)
            for i in range(len(current_acm.entry_values)):
                current_acm.entry_values[i].enabled[
                    idx
                ] = True  # Allow collision with this object
                current_acm.entry_values[idx].enabled[
                    i
                ] = True  # Allow collision from this object

        return current_acm

    def _apply_acm(self, acm):
        """Apply an Allowed Collision Matrix to the planning scene.

        Args:
            acm: AllowedCollisionMatrix to apply

        Returns:
            bool: True if successful, False otherwise
        """
        # Wait for service
        rospy.wait_for_service("/apply_planning_scene")
        apply_planning_scene = rospy.ServiceProxy(
            "/apply_planning_scene", ApplyPlanningScene
        )

        # Create a planning scene with the ACM
        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.allowed_collision_matrix = acm

        # Apply the scene
        response = apply_planning_scene(planning_scene)
        return response.success

    def _get_primitive_type(self, type_str):
        """Convert string type to SolidPrimitive type."""
        type_map = {
            "box": SolidPrimitive.BOX,
            "sphere": SolidPrimitive.SPHERE,
            "cylinder": SolidPrimitive.CYLINDER,
        }
        return type_map.get(type_str, SolidPrimitive.BOX)

    def get_move_group(self):
        """Return the configured move group."""
        return self.move_group

    def get_scene(self):
        """Return the planning scene interface."""
        return self.scene

    def get_robot(self):
        """Return the robot commander object."""
        return self.robot

    def _ensure_acm_applied(self):
        """Ensure the SRDF's allowed collision matrix is correctly applied."""
        rospy.loginfo("Ensuring allowed collision matrix is correctly applied...")

        # Get the current ACM
        current_acm = self._get_current_acm()

        # Apply it to ensure it's correctly set
        success = self._apply_acm(current_acm)

        rospy.loginfo(f"Reapplied ACM from SRDF, success: {success}")
        return success

    def check_collision(self, joint_positions=None, group_name=None):
        """Check if a robot state is in collision using the optimized collision checker.

        Args:
            joint_positions: List of joint positions to check. If None, the current robot state is used.
            group_name: Name of the move group to check. Ignored if using the optimized checker.

        Returns:
            tuple: (is_valid, contacts) where is_valid is a boolean and contacts is a list of collision contacts
        """
        # Use the optimized collision checker
        is_valid = self.collision_checker.check_state_validity(joint_positions)

        # Return the result and contacts in the same format as before
        return is_valid, self.collision_checker.last_contacts

    def print_acm_status(self):
        """Print the current Allowed Collision Matrix status for debugging."""
        rospy.loginfo("Retrieving current Allowed Collision Matrix (ACM) status...")

        # Get the current ACM
        acm = self._get_current_acm()

        # Analyze and print ACM statistics
        self._print_acm_statistics(acm)

        # Print detailed object information
        self._print_acm_object_details(acm)

        return acm

    def _print_acm_statistics(self, acm):
        """Print statistics about the ACM.

        Args:
            acm: The AllowedCollisionMatrix to analyze
        """
        # Calculate statistics
        entry_count = len(acm.entry_names)
        enabled_count = 0
        total_pairs = 0

        for i in range(entry_count):
            for j in range(entry_count):
                total_pairs += 1
                if acm.entry_values[i].enabled[j]:
                    enabled_count += 1

        disabled_count = total_pairs - enabled_count

        # Print summary
        rospy.loginfo(f"ACM contains {entry_count} entries")
        rospy.loginfo(f"Total collision pairs: {total_pairs}")
        rospy.loginfo(
            f"Collision checking ENABLED (FALSE in ACM): {disabled_count} pairs"
        )
        rospy.loginfo(
            f"Collision checking DISABLED (TRUE in ACM): {enabled_count} pairs"
        )

    def _print_acm_object_details(self, acm):
        """Print detailed information about objects in the ACM.

        Args:
            acm: The AllowedCollisionMatrix to analyze
        """
        for name in acm.entry_names:
            idx = acm.entry_names.index(name)
            disabled_with = []

            for i, other_name in enumerate(acm.entry_names):
                if acm.entry_values[idx].enabled[i]:
                    disabled_with.append(other_name)

            if disabled_with:
                rospy.loginfo(
                    f"Object '{name}' has collision DISABLED with {len(disabled_with)} other objects/links"
                )

    def _attach_tool(self):
        """Attach a mesh tool to the robot's tool0 frame if configured."""
        tool_config = get_param("/environment/tool", None)
        if not tool_config:
            rospy.loginfo("No tool configuration found, skipping tool attachment")
            return

        rospy.loginfo("Attaching tool to robot's tool0 frame...")

        # Get the end effector link
        eef_link = self.move_group.get_end_effector_link()
        rospy.loginfo(f"Attaching tool to end effector link: {eef_link}")

        # Get all links in the manipulator group for touch_links
        touch_links = self.robot.get_link_names(group=self.group_name)

        # Create pose for the tool
        pose = PoseStamped()
        pose.header.frame_id = eef_link
        pose.pose.position = Point(*tool_config["pose"]["position"])

        # Convert euler angles to quaternion
        euler = tool_config["pose"]["orientation"]
        q = quaternion_from_euler(*euler)
        pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Resolve the mesh path
        mesh_path = resolve_package_path(tool_config["mesh_path"])

        # Attach the mesh tool
        self.scene.attach_mesh(
            eef_link,
            "tool",
            pose=pose,
            filename=mesh_path,
            size=tool_config["scale"],
            touch_links=touch_links,
        )

        # Wait for attachment with timeout
        timeout = 4.0  # seconds
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the tool is in attached objects
            attached_objects = self.scene.get_attached_objects(["tool"])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the tool is in the scene
            is_known = "tool" in self.scene.get_known_object_names()

            # Test if we are in the expected state
            if is_attached and not is_known:
                rospy.loginfo(f"Successfully attached tool to {eef_link}")
                # Set up TCP frame after tool attachment
                self._setup_tcp_frame(tool_config)
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        rospy.logerr(f"Failed to attach tool to {eef_link} within {timeout} seconds")
        return False

    def _setup_tcp_frame(self, tool_config):
        """Set up the Tool Center Point (TCP) frame for MoveIt.

        Args:
            tool_config: Tool configuration dictionary containing TCP pose
        """
        if "tcp" not in tool_config:
            rospy.logwarn("No TCP configuration found in tool config")
            return

        tcp_config = tool_config["tcp"]
        if "pose" not in tcp_config:
            rospy.logwarn("No TCP pose configuration found")
            return

        # Get the end effector link
        eef_link = self.move_group.get_end_effector_link()

        # Create TCP pose
        tcp_pose = Pose()
        tcp_position = tcp_config["pose"]["position"]
        tcp_pose.position.x = tcp_position[0]
        tcp_pose.position.y = tcp_position[1]
        tcp_pose.position.z = tcp_position[2]

        # Set orientation
        tcp_orientation = tcp_config["pose"]["orientation"]
        q = quaternion_from_euler(*tcp_orientation)
        tcp_pose.orientation.x = q[0]
        tcp_pose.orientation.y = q[1]
        tcp_pose.orientation.z = q[2]
        tcp_pose.orientation.w = q[3]

        # Cache the TCP transform
        self.tcp_transform = ros_numpy.numpify(tcp_pose)

        # Create static transform broadcaster
        static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Create transform message
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = eef_link
        transform.child_frame_id = "tcp"

        # Set translation
        transform.transform.translation.x = tcp_position[0]
        transform.transform.translation.y = tcp_position[1]
        transform.transform.translation.z = tcp_position[2]

        # Set rotation
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]

        # Send the transform
        static_broadcaster.sendTransform(transform)

        rospy.loginfo(
            f"Set TCP pose relative to {eef_link} in MoveIt and published to TF tree"
        )

    def get_end_effector_transform(self, pose):
        """
        Get the end-effector transform for given target.

        Args:
            pose: The pose from target to world or TCP to world
        Returns:
            The transformed pose (same type as input)
        """
        if self.tcp_transform is None:
            rospy.logwarn("TCP transform not set up")
            return pose

        # Convert to Pose if PoseStamped
        if isinstance(pose, PoseStamped):
            pose = pose.pose

        target = ros_numpy.numpify(pose)  # target to world
        # Get end-effector to tcp
        # w_T_e = w_T_t * t_T_e
        return ros_numpy.geometry.numpy_to_pose(
            np.matmul(target, np.linalg.inv(self.tcp_transform))
        )


def main():
    try:
        # Initialize environment loader
        env = EnvironmentLoader()

        # Keep the node running
        rospy.loginfo("Environment loader is running. Press Ctrl+C to exit.")
        from IPython import embed

        embed()

    except ImportError as e:
        rospy.logerr(f"Missing required package: {e}")
        rospy.logerr("Please install required packages with: pip install meshio")
    except Exception as e:
        rospy.logerr(f"Error in environment loader: {e}")
    finally:
        try:
            # Attempt to clear scene before exiting
            if "env" in locals():
                env.clear_scene()
        except:
            pass


if __name__ == "__main__":
    main()
