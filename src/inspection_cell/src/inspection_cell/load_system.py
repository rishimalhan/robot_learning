#!/usr/bin/env python3

# External

import rospy
import os
from tf.transformations import quaternion_from_euler
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
import rospkg
from moveit_msgs.msg import (
    PlanningScene,
    AllowedCollisionEntry,
    ObjectColor,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
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
        # Initialize ROS node if needed
        if not rospy.core.is_initialized():
            rospy.init_node("environment_loader", anonymous=True)

        # Setup basic components
        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander(move_group_name)
        self.robot = RobotCommander()
        self.group_name = move_group_name
        self.planner = Planner()
        self.executor = Executor()
        self.planning_frame = self.move_group.get_planning_frame()
        self.rospack = rospkg.RosPack()

        # Create reusable publishers
        self.planning_scene_pub = rospy.Publisher(
            "/planning_scene", PlanningScene, queue_size=10
        )
        rospy.sleep(0.5)  # Allow publisher to initialize

        # Setup service proxies
        rospy.wait_for_service("/get_planning_scene", timeout=2.0)
        self.get_planning_scene_srv = rospy.ServiceProxy(
            "/get_planning_scene", GetPlanningScene
        )

        rospy.wait_for_service("/apply_planning_scene")
        self.apply_planning_scene_srv = rospy.ServiceProxy(
            "/apply_planning_scene", ApplyPlanningScene
        )

        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize TCP transform cache
        self.tcp_transform = np.eye(4)

        # Clear scene if requested
        if clear_scene:
            self.clear_scene()

        # Setup the environment
        self._load_config()
        self._attach_tool()
        self._add_objects_to_scene()
        self._update_acm()

        # Add a small delay to ensure ACM changes have propagated
        rospy.sleep(1.0)

        # Initialize collision checker
        self.collision_checker = CollisionCheck(move_group_name=move_group_name)

    def _load_config(self):
        """Load environment configuration from YAML file."""
        config_path = os.path.join(
            self.rospack.get_path("inspection_cell"), "config", "environment.yaml"
        )
        self.config = load_yaml_to_params(config_path, "/environment")
        rospy.loginfo("Environment configuration loaded")

    def clear_scene(self):
        """Clear all objects from the planning scene."""
        for obj in self.scene.get_objects():
            self.scene.remove_world_object(obj)
        rospy.loginfo("Planning scene cleared")

    def _add_objects_to_scene(self):
        """Add objects to the planning scene based on configuration."""
        objects = get_param("/environment", {})
        rospy.sleep(0.5)  # Allow publisher to initialize

        # Process each object in the environment config
        for name, obj in objects.items():
            # Skip tool since it's handled separately
            if name == "tool":
                continue

            rospy.loginfo(f"Adding object: {name}")

            # Skip objects that are missing required fields
            if not obj or "type" not in obj:
                rospy.logerr(
                    f"Object '{name}' is missing required 'type' field. Skipping."
                )
                continue

            # Create a planning scene message
            planning_scene = PlanningScene()
            planning_scene.is_diff = True

            # Create and add collision object
            collision_object = self._create_collision_object(name, obj)
            if collision_object:
                planning_scene.world.collision_objects.append(collision_object)

                # Set color if specified
                if "color" in obj:
                    object_color = ObjectColor()
                    object_color.id = name
                    object_color.color = ColorRGBA(
                        obj["color"][0],
                        obj["color"][1],
                        obj["color"][2],
                        obj.get("alpha", 1.0),
                    )
                    planning_scene.object_colors.append(object_color)

                # Publish planning scene
                self.planning_scene_pub.publish(planning_scene)
                rospy.sleep(0.1)  # Give time for update

    def _create_collision_object(self, name, obj):
        """Create a collision object based on configuration."""
        collision_object = CollisionObject()
        collision_object.header.frame_id = self.planning_frame
        collision_object.id = name
        collision_object.operation = CollisionObject.ADD

        # Handle different object types
        if obj["type"] in ["box", "sphere", "cylinder"]:
            # Create primitive
            primitive = SolidPrimitive()

            if obj["type"] == "box":
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = obj["dimensions"]
            elif obj["type"] == "sphere":
                primitive.type = SolidPrimitive.SPHERE
                primitive.dimensions = [obj["radius"]]
            elif obj["type"] == "cylinder":
                primitive.type = SolidPrimitive.CYLINDER
                primitive.dimensions = [obj["height"], obj["radius"]]

            collision_object.primitives.append(primitive)

            # Add pose
            pose = Pose()
            pose.position = Point(*obj["pose"]["position"])
            euler = obj["pose"]["orientation"]
            q = quaternion_from_euler(*euler)
            pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            collision_object.primitive_poses.append(pose)

        elif obj["type"] == "mesh":
            # Handle mesh objects
            try:
                import meshio

                # Resolve mesh path
                mesh_path = resolve_package_path(obj["mesh_path"])
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

                # Add mesh to collision object
                collision_object.meshes.append(co_mesh)

                # Add mesh pose
                pose = Pose()
                pose.position = Point(*obj["pose"]["position"])
                euler = obj["pose"]["orientation"]
                q = quaternion_from_euler(*euler)
                pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                collision_object.mesh_poses.append(pose)

            except Exception as e:
                rospy.logerr(f"Failed to load mesh for {name}: {e}")
                return None

        else:
            rospy.logerr(f"Unsupported object type: {obj['type']}")
            return None

        return collision_object

    def _attach_tool(self):
        """Attach a tool to the robot if configured."""
        tool_config = get_param("/environment/tool", None)
        if not tool_config:
            return

        # Get the end effector link
        eef_link = self.move_group.get_end_effector_link()

        # Get touch links
        touch_links = self.robot.get_link_names(group=self.group_name)

        # Create pose for the tool
        pose = PoseStamped()
        pose.header.frame_id = eef_link
        pose.pose.position = Point(*tool_config["pose"]["position"])

        # Set orientation
        euler = tool_config["pose"]["orientation"]
        q = quaternion_from_euler(*euler)
        pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Resolve mesh path
        mesh_path = resolve_package_path(tool_config["mesh_path"])

        # Get objects that should have collision disabled with tool
        environment_config = get_param("/environment")
        collision_disabled_objects = []
        for obj_name, obj_config in environment_config.items():
            if not obj_config.get("collision_enabled", True):
                collision_disabled_objects.append(obj_name)

        # IMPORTANT: First attach the mesh tool TELLING IT TO IGNORE collisions with specific objects
        self.scene.attach_mesh(
            eef_link,
            "tool",
            pose=pose,
            filename=mesh_path,
            size=tool_config["scale"],
            touch_links=touch_links,
        )

        # Wait for attachment to complete
        rospy.sleep(1.0)

        # Set up TCP frame if configured
        if "tcp" in tool_config:
            self._setup_tcp_frame(tool_config)

    def _setup_tcp_frame(self, tool_config):
        """Set up the Tool Center Point (TCP) frame for MoveIt."""
        if "tcp" not in tool_config or "pose" not in tool_config["tcp"]:
            return

        # Get the end effector link
        eef_link = self.move_group.get_end_effector_link()

        # Create TCP pose
        tcp_pose = Pose()
        tcp_position = tool_config["tcp"]["pose"]["position"]
        tcp_pose.position.x = tcp_position[0]
        tcp_pose.position.y = tcp_position[1]
        tcp_pose.position.z = tcp_position[2]

        # Set orientation
        tcp_orientation = tool_config["tcp"]["pose"]["orientation"]
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
        # Convert to Pose if PoseStamped
        if isinstance(pose, PoseStamped):
            pose = pose.pose

        target = ros_numpy.numpify(pose)  # target to world
        # Get end-effector to tcp
        # w_T_e = w_T_t * t_T_e
        return ros_numpy.geometry.numpy_to_pose(
            np.matmul(target, np.linalg.inv(self.tcp_transform))
        )

    def _update_acm(self):
        """Update the Allowed Collision Matrix."""
        try:
            # Get current ACM
            current_acm = self._get_current_acm()
            if not current_acm:
                rospy.logwarn("No ACM retrieved, skipping ACM update")
                return
            objects = self.scene.get_objects()
            objects.update(self.scene.get_attached_objects())
            environment_config = get_param("/environment")

            # Track objects that should have collision disabled
            collision_disabled_objects = []
            for obj_name, obj_config in environment_config.items():
                if obj_name in objects and not obj_config.get(
                    "collision_enabled", True
                ):
                    collision_disabled_objects.append(obj_name)

            if collision_disabled_objects:
                rospy.loginfo(
                    f"Objects with disabled collision: {', '.join(collision_disabled_objects)}"
                )

            # First ensure all objects are in the ACM
            added_objects = []
            existing_entry_indices = []
            for object_id in objects:
                # Skip if already in ACM
                if object_id in current_acm.entry_names:
                    existing_entry_indices.append(
                        current_acm.entry_names.index(object_id)
                    )
                    continue
                # Add object to entry names
                current_acm.entry_names.append(object_id)
                added_objects.append(object_id)

            # Extend existing entries with default values for new objects
            for i in range(len(current_acm.entry_values)):
                if len(added_objects) > 0:
                    current_acm.entry_values[i].enabled.extend(
                        [False] * len(added_objects)
                    )
                    rospy.loginfo(
                        f"Extended entry at index {i} for object {current_acm.entry_names[i]} to handle {len(added_objects)} new objects"
                    )

            # Create new entries for added objects
            for i, object_id in enumerate(added_objects):
                entry = AllowedCollisionEntry()
                entry.enabled = [False] * len(current_acm.entry_names)
                current_acm.entry_values.append(entry)
                # Log the new entry with proper index
                new_idx = len(current_acm.entry_names) - len(added_objects) + i
                rospy.loginfo(
                    f"Added new entry for {object_id} at index {new_idx} with {len(entry.enabled)} values"
                )

            # Apply the updated ACM with new objects
            self._apply_acm(current_acm)

            # Now disable collisions for objects that need it
            for obj_name in collision_disabled_objects:
                # Disable self-collision for this object
                self._disable_collision_between(obj_name, obj_name)

                # Disable collision with everything else in scene
                for other_obj_name in objects.keys():
                    if other_obj_name != obj_name:
                        self._disable_collision_between(obj_name, other_obj_name)

        except Exception as e:
            rospy.logerr(f"Error updating ACM: {e}")

    def _disable_collision_between(self, name1, name2):
        """Explicitly disable collision between two named objects."""
        try:
            # Get current ACM
            current_acm = self._get_current_acm()
            if not current_acm:
                return False

            # Check if both objects are in the ACM
            if (
                name1 not in current_acm.entry_names
                or name2 not in current_acm.entry_names
            ):
                return False

            # Get indices
            idx1 = current_acm.entry_names.index(name1)
            idx2 = current_acm.entry_names.index(name2)

            # True in ACM means collisions are ALLOWED (ignored/disabled)
            current_acm.entry_values[idx1].enabled[idx2] = True
            if idx1 != idx2:  # Don't set twice for same object
                current_acm.entry_values[idx2].enabled[idx1] = True

            # Apply the change
            return self._apply_acm(current_acm)
        except Exception as e:
            rospy.logerr(f"Error disabling collision between {name1} and {name2}: {e}")
            return False

    def _get_current_acm(self):
        """Get the current Allowed Collision Matrix."""
        try:
            # Create proper PlanningSceneComponents message
            components = PlanningSceneComponents()
            components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX

            # Call with the proper message
            response = self.get_planning_scene_srv(components)
            return response.scene.allowed_collision_matrix
        except Exception as e:
            rospy.logerr(f"Failed to get planning scene: {e}")
            return None

    def _apply_acm(self, acm):
        """Apply an Allowed Collision Matrix to the planning scene."""
        try:
            # Create a planning scene with the ACM
            planning_scene = PlanningScene()
            planning_scene.is_diff = True
            planning_scene.allowed_collision_matrix = acm

            # Apply the scene
            response = self.apply_planning_scene_srv(planning_scene)
            return response.success
        except Exception as e:
            rospy.logerr(f"Failed to apply ACM: {e}")
            return False

    def check_collision(self, joint_positions=None):
        """Check if a robot state is in collision."""
        return self.collision_checker.check_collision(joint_positions)

    def get_move_group(self):
        """Return the configured move group."""
        return self.move_group

    def get_scene(self):
        """Return the planning scene interface."""
        return self.scene

    def get_robot(self):
        """Return the robot commander object."""
        return self.robot


def main():
    try:
        # Assume environment parameters are already loaded to the ROS parameter server
        # Initialize environment loader
        env = EnvironmentLoader()

        # Keep the node running
        rospy.loginfo("Environment loader running. Press Ctrl+C to exit.")
        from IPython import embed

        embed()

    except Exception as e:
        rospy.logerr(f"Error in environment loader: {e}")
    finally:
        # Attempt to clear scene before exiting
        if "env" in locals():
            env.clear_scene()


if __name__ == "__main__":
    main()
