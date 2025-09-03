import os
import time
import numpy as np

import mujoco
import mujoco.viewer

from dm_control import mjcf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--robot', default='fr3', choices=['fr3'])

args = parser.parse_args()

if args.robot == 'fr3':
    from robot_descriptions import fr3_mj_description as robot_mj_description
else:
    raise ValueError(f"Unknown robot: {args.robot}")

# ###############################################################################
# robot arm and hand
# ###############################################################################

robot_mjcf = mjcf.from_path(robot_mj_description.MJCF_PATH)

joint_names = [joint.name for joint in robot_mjcf.find_all('joint')]
print("Joint names:", joint_names)

del robot_mjcf.actuator
motor_class = robot_mjcf.default.add('default', dclass='motor')
motor_class.motor.set_attributes(ctrlrange='-1000 1000')
for joint_name in joint_names:
    robot_mjcf.actuator.add('motor', dclass='motor', name=joint_name+'_motor', joint=joint_name)
    
for joint_name in joint_names:
    robot_mjcf.sensor.add('jointpos', name=joint_name+'_pos', joint=joint_name)
    
for joint_name in joint_names:
    robot_mjcf.sensor.add('jointvel', name=joint_name+'_vel', joint=joint_name)

# ###############################################################################
# mujoco scene
# ###############################################################################

scene = mjcf.RootElement()
scene.asset.add('texture', type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="32", height="512")
scene.asset.add('texture', type="2d", name="groundplane", builtin="checker", mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
scene.asset.add('material', name="groundplane", texture="groundplane", texuniform="true", texrepeat="5 5", reflectance="0.2")

scene.worldbody.add('light', pos="0 0 1")
scene.worldbody.add('light', pos="0 -0.2 1", dir="0 0.2 -0.8", directional="true")
scene.worldbody.add('geom', name='floor', size="0 0 0.05", type='plane', material='groundplane')

scene.statistic.center = "0.2 0 0.4"
scene.statistic.extent = "0.8"

# attach robot to the scene
fixed_base_site = scene.worldbody.add('site', name='fixed_base', pos="0 0 0", euler="0 0 0")
fixed_base_site.attach(robot_mjcf)

# ###############################################################################
# output
# ###############################################################################

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'resources', args.robot+"_mj_description")
output_file = args.robot + "_scene.xml"
if os.path.exists(output_dir):
    if os.name == 'posix':
        os.system(f"rm -rf {output_dir}")
    else:
        os.system(f"rmdir /s /q {output_dir}")
mjcf.export_with_assets(scene, out_dir=output_dir, out_file_name=output_file)
print(f"Output to {os.path.join(output_dir, output_file)}")

