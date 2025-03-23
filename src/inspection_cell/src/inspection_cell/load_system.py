#!/usr/bin/env python3

# External

import rospy
import yaml
import os
from tf.transformations import quaternion_from_euler
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
import rospkg
from moveit_msgs.msg import PlanningScene, AllowedCollisionMatrix, AllowedCollisionEntry, ObjectColor
from moveit_msgs.srv import ApplyPlanningScene
from std_msgs.msg import ColorRGBA

# Internal

from inspection_cell.utils import resolve_package_path

class EnvironmentLoader:
    def __init__(self, move_group_name="manipulator", clear_scene=True):
        rospy.init_node('environment_loader', anonymous=True)
        
        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander(move_group_name)
        
        # Create RosPack instance
        self.rospack = rospkg.RosPack()
        
        # Get the planning frame
        self.planning_frame = self.move_group.get_planning_frame()
        rospy.loginfo(f"Planning frame: {self.planning_frame}")
        
        # Clear the planning scene if requested
        if clear_scene:
            self.clear_scene()
        
        # Load environment configuration using rospkg
        config_path = os.path.join(self.rospack.get_path('inspection_cell'), 'config', 'environment.yaml')
        rospy.loginfo(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Configure robot parameters
        self._configure_robot()
        
        # Add objects to scene
        self._add_objects_to_scene()
        
        # Wait for scene to update
        rospy.sleep(1.0)
        
        # Print current scene objects
        self._print_scene_objects()

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
        robot_config = self.config['robot']
        self.move_group.set_planning_time(robot_config['planning_time'])
        self.move_group.set_num_planning_attempts(robot_config['num_planning_attempts'])
        self.move_group.set_max_velocity_scaling_factor(robot_config['max_velocity_scaling_factor'])
        self.move_group.set_max_acceleration_scaling_factor(robot_config['max_acceleration_scaling_factor'])
        rospy.loginfo("Robot parameters configured")

    @staticmethod
    def _disable_collision_for_object(object_id):
        rospy.wait_for_service('/apply_planning_scene')
        apply_planning_scene = rospy.ServiceProxy('/apply_planning_scene', ApplyPlanningScene)

        # Create PlanningScene message
        planning_scene = PlanningScene()
        planning_scene.is_diff = True

        # Create entry for AllowedCollisionMatrix
        acm = AllowedCollisionMatrix()
        acm.entry_names.append(object_id)

        entry = AllowedCollisionEntry()
        entry.enabled = [True]  # Allow collision with everything

        acm.entry_values.append(entry)
        planning_scene.allowed_collision_matrix = acm

        # Apply the scene
        response = apply_planning_scene(planning_scene)
        rospy.loginfo(f"Collision disabled for object: {object_id}, success: {response.success}")

    def _add_objects_to_scene(self):
        """Add objects to the planning scene based on configuration."""
        rospy.loginfo(f"Adding {len(self.config['objects'])} objects to scene")
        
        # Create planning scene publisher
        planning_scene_pub = rospy.Publisher('/planning_scene', PlanningScene, queue_size=10)
        rospy.sleep(0.5)  # Allow publisher to initialize
        
        for name, obj in self.config['objects'].items():
            rospy.loginfo(f"Adding object: {name} of type {obj['type']}")
            
            # Create a planning scene message
            planning_scene = PlanningScene()
            planning_scene.is_diff = True
            
            # Create collision object
            collision_object = CollisionObject()
            collision_object.header.frame_id = self.planning_frame
            collision_object.id = name
            collision_object.operation = CollisionObject.ADD
            
            # Create primitive or mesh based on type
            if obj['type'] in ['box', 'sphere', 'cylinder']:
                # Create primitive
                primitive = SolidPrimitive()
                primitive.type = self._get_primitive_type(obj['type'])
                
                if obj['type'] == 'box':
                    primitive.dimensions = obj['dimensions']
                    rospy.loginfo(f"  Box dimensions: {obj['dimensions']}")
                elif obj['type'] == 'sphere':
                    primitive.dimensions = [obj['radius']]
                    rospy.loginfo(f"  Sphere radius: {obj['radius']}")
                elif obj['type'] == 'cylinder':
                    primitive.dimensions = [obj['height'], obj['radius']]
                    rospy.loginfo(f"  Cylinder height: {obj['height']}, radius: {obj['radius']}")
                
                # Add the primitive to the collision object
                collision_object.primitives.append(primitive)
                
                # Create and add the pose for this primitive
                pose = Pose()
                pose.position = Point(*obj['pose']['position'])
                euler = obj['pose']['orientation']
                q = quaternion_from_euler(*euler)
                pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                collision_object.primitive_poses.append(pose)
                rospy.loginfo(f"  Pose: position={obj['pose']['position']}, orientation={euler}")
                
            elif obj['type'] == 'mesh':
                mesh_path = obj.get('mesh_path')
                if not mesh_path:
                    rospy.logerr(f"Mesh path not specified for object {name}")
                    continue
                # Remove package:// prefix if present
                mesh_path = resolve_package_path(mesh_path)
                rospy.loginfo(f"Loading mesh from: {mesh_path}")
                
                try:
                    import meshio
                    # Load the mesh file
                    mesh = meshio.read(mesh_path)
                    
                    # Create mesh for collision object
                    from shape_msgs.msg import Mesh, MeshTriangle
                    co_mesh = Mesh()
                    scale = obj.get('scale', [1.0, 1.0, 1.0])
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
                                triangle.vertex_indices = [int(indices[0]), int(indices[1]), int(indices[2])]
                                co_mesh.triangles.append(triangle)
                    
                    if not co_mesh.triangles:
                        rospy.logerr(f"No triangles found in mesh {mesh_path}")
                        continue
                    
                    # Add mesh to collision object
                    collision_object.meshes.append(co_mesh)
                    
                    # Add mesh pose
                    pose = Pose()
                    pose.position = Point(*obj['pose']['position'])
                    euler = obj['pose']['orientation']
                    q = quaternion_from_euler(*euler)
                    pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                    collision_object.mesh_poses.append(pose)
                    
                except Exception as e:
                    rospy.logerr(f"Failed to load mesh {mesh_path}: {e}")
                    continue
            
            # Add collision object to planning scene
            planning_scene.world.collision_objects.append(collision_object)
            
            # Set color if specified
            color = obj.get('color')
            alpha = obj.get('alpha', 1.0)
            if color:
                object_color = ObjectColor()
                object_color.id = name
                object_color.color = ColorRGBA(color[0], color[1], color[2], alpha)
                planning_scene.object_colors.append(object_color)
            
            # Publish planning scene
            planning_scene_pub.publish(planning_scene)
            rospy.loginfo(f"Published object: {name} to planning scene")
            
            # Disable collision if needed
            if not obj.get('collision_enabled', True):
                self._disable_collision_for_object(name)
        
        # Wait for the scene to update
        rospy.sleep(1.0)

    def _get_primitive_type(self, type_str):
        """Convert string type to SolidPrimitive type."""
        type_map = {
            'box': SolidPrimitive.BOX,
            'sphere': SolidPrimitive.SPHERE,
            'cylinder': SolidPrimitive.CYLINDER
        }
        return type_map.get(type_str, SolidPrimitive.BOX)

    def get_move_group(self):
        """Return the configured move group."""
        return self.move_group

    def get_scene(self):
        """Return the planning scene interface."""
        return self.scene

def main():
    try:
        # Initialize environment loader
        env_loader = EnvironmentLoader()
        
        # Keep the node running
        rospy.loginfo("Environment loader is running. Press Ctrl+C to exit.")
        rospy.spin()
        
    except ImportError as e:
        rospy.logerr(f"Missing required package: {e}")
        rospy.logerr("Please install required packages with: pip install meshio")
    except Exception as e:
        rospy.logerr(f"Error in environment loader: {e}")
    finally:
        try:
            # Attempt to clear scene before exiting
            if 'env_loader' in locals():
                env_loader.clear_scene()
        except:
            pass

if __name__ == '__main__':
    main()
