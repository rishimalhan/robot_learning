#!/usr/bin/env python3
"""
MuJoCo viewer for the inspection sensor scene.
Loads the same dynamic scene from MoveIt environment parameters.
"""

import os
import shutil
import tempfile
import time

import mujoco
import mujoco.viewer
import numpy as np
import rospy
from tf.transformations import quaternion_from_euler

from core.utils import convert_stl_to_obj
from neural_engine.scene_assets import get_part_spec


def load_scene_from_environment():
    """Load MuJoCo scene from the configured physical asset."""
    spec = get_part_spec()
    mesh_path = spec["mesh_path"]
    pose = spec.get("pose")
    position = pose.get("position")
    orientation = pose.get("orientation")
    scale = spec.get("scale")

    temp_dir = tempfile.mkdtemp(prefix="inspection_viewer_")
    obj_path = convert_stl_to_obj(mesh_path, temp_dir)
    quat = quaternion_from_euler(*orientation)

    xml_string = f"""
<mujoco model="inspection_sensor_scene">
    <option timestep="0.01" gravity="0 0 -9.81"/>
    <visual>
        <global offwidth="1024" offheight="1024"/>
    </visual>
    <asset>
        <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.3 0.3 0.3" rgb2="0.4 0.4 0.4"/>
        <material name="floor_mat" texture="floor_tex" texrepeat="5 5" texuniform="true" reflectance="0.2"/>
        <material name="fixture_mat" rgba="0.2 0.2 0.2 1"/>
        <material name="workpiece_mat" rgba="0.8 0.2 0.2 1"/>
        <mesh name="workpiece_mesh" file="{obj_path}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
    </asset>
    <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" material="floor_mat"/>
        <body name="sensor_mount" pos="0 0 3.0">
            <geom name="mast" type="cylinder" size="0.02 0.1" rgba="0.5 0.5 0.5 1" pos="0 0 -0.1"/>
            <body name="sensor_head" pos="0 0 0">
                <geom name="housing" type="box" size="0.1 0.15 0.06" material="fixture_mat"/>
                <geom name="lens" type="cylinder" size="0.04 0.02" pos="0 0 -0.06" rgba="0 0 0 1" quat="0.707 0.707 0 0"/>
                <camera name="inspection_sensor" pos="0 0 0" quat="1 0 0 0" fovy="60" resolution="1024 1024"/>
            </body>
        </body>
        <body name="workpiece" pos="{position[0]} {position[1]} {position[2]}" quat="{quat[3]} {quat[0]} {quat[1]} {quat[2]}">
            <geom name="workpiece_geom" type="mesh" mesh="workpiece_mesh" material="workpiece_mat"/>
        </body>
        <light name="key_light" pos="0 0 2.5" dir="0 0 -1" diffuse="1 1 1"/>
    </worldbody>
</mujoco>
"""

    return xml_string, temp_dir


def main():
    rospy.init_node("mujoco_viewer", anonymous=True)

    rospy.loginfo("Loading scene from MoveIt environment...")
    xml_string, temp_dir = load_scene_from_environment()

    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    rospy.loginfo("Launching MuJoCo viewer window...")
    try:
        with mujoco.viewer.launch(model, data) as viewer:
            last_time = time.time()
            while viewer.is_running() and not rospy.is_shutdown():
                current_time = time.time()
                if current_time - last_time < model.opt.timestep:
                    time.sleep(model.opt.timestep - (current_time - last_time))
                mujoco.mj_step(model, data)
                viewer.sync()
                last_time = current_time
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    rospy.loginfo("MuJoCo viewer closed.")


if __name__ == "__main__":
    main()
