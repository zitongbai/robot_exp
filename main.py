import time
import os
import numpy as np
import argparse
import enum
import mujoco as mj
import mujoco.viewer

import pinocchio as pin

from robot import Robot
from utils.math import lemniscate_trajectory


class SimStage(enum.Enum):
    INIT = 1
    TRAJECTORY = 2
    FINAL = 3


def print_sim_info(model, data):
    # print joint names
    print("-"*50)
    print("Simulation Info: ")
    joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    print("Joint names:", joint_names)
    print("Simulation in dt =", model.opt.timestep)
    print("-"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', default='fr3', choices=['fr3'])
    parser.add_argument("-d", "--duration", default=5, type=float, help="Duration to run the simulation (in seconds)")
    parser.add_argument("-r", "--record", action="store_true", help="Record simulation data")

    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.robot == 'fr3':
        xml_path = os.path.join(current_dir, 'resources/fr3_mj_description/fr3_scene.xml')
    else:
        raise ValueError(f"Unknown robot: {args.robot}")

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    
    # -----------------------------------------------------------
    print_sim_info(model, data)
    # -----------------------------------------------------------
    
    pin_model = pin.buildModelFromMJCF(xml_path)
    # -----------------------------------------------------------
    # test pinocchio
    print("Create pinocchio model: ", pin_model.name)
    pin_data = pin_model.createData()
    test_q = pin.randomConfiguration(pin_model)
    pin.forwardKinematics(pin_model, pin_data, test_q)
    pin.updateFramePlacements(pin_model, pin_data)
    print("-"*50)
    for name, oMi in zip(pin_model.names, pin_data.oMi):
        print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))
    print("-"*50)
    for frame, oMf in zip(pin_model.frames, pin_data.oMf):
        print("{:<24} : {: .2f} {: .2f} {: .2f}".format(frame.name, *oMf.translation.T.flat))
    print("-"*50)
    # -----------------------------------------------------------
    
    robot = Robot(model, data, pin_model)
    
    # init data
    # init_target_pos, _ = lemniscate_trajectory(0, a=0.2, offset=np.array([0.4,0,0.6]))
    # init_q, init_dq = robot.inverse_kinematics(
    #     target_pos=init_target_pos,
    #     target_ori=np.array([
    #                 [1, 0, 0], 
    #                 [0, -1, 0], 
    #                 [0, 0, -1]
    #             ])
    # )
    init_q = np.array([0.0, 0.4, 0.0, -1.4, 0.0, 1.68, 0.0])
    init_dq = np.zeros(7)
    
    data.qpos[:robot.joint_num] = init_q.copy()
    data.qvel[:robot.joint_num] = init_dq.copy()
    print("Initial joint positions:", data.qpos[:robot.joint_num])
    print("Initial joint velocities:", data.qvel[:robot.joint_num])

    stage = SimStage.INIT

    record_qpos = []
    record_qvel = []
    record_ee_pos = []
    

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < args.duration:
            step_start = time.time()
            
            if stage == SimStage.INIT:
                q_des = init_q
                dq_des = init_dq
                
                tau = robot.ctrl_fb_lin(
                    q_des=q_des,
                    dp_des=dq_des,
                    Kp=np.array([200, 200, 200, 200, 50, 50, 50]),
                    # Kd=np.array([10, 10, 10, 10, 10, 10, 10])
                )
                
                eps = 1e-2
                if np.linalg.norm(data.qpos[:robot.joint_num] - q_des) < eps and np.linalg.norm(data.qvel[:robot.joint_num] - dq_des) < eps:
                    print("Switch to TRAJECTORY stage")
                    stage = SimStage.TRAJECTORY
                    traj_start = time.time()
                
            elif stage == SimStage.TRAJECTORY:
                t = time.time() - traj_start
                target_pos, target_vel, _ = lemniscate_trajectory(t, a=0.2, offset=np.array([0.4,0,0.6]))
                q_des, dq_des = robot.inverse_kinematics(
                    target_pos=target_pos,
                    target_ori=np.array([
                        [1, 0, 0], 
                        [0, -1, 0], 
                        [0, 0, -1]
                    ]), 
                    target_lin_vel=target_vel,
                    target_ang_vel=np.zeros(3),
                )
            
                tau = robot.ctrl_fb_lin(
                    q_des=q_des,
                    dp_des=dq_des,
                    Kp=np.array([200, 200, 200, 200, 50, 50, 50]),
                    # Kd=np.array([10, 10, 10, 10, 10, 10, 10])
                )
            
            if args.record:
                record_qpos.append(data.qpos[:robot.joint_num].copy())
                record_qvel.append(data.qvel[:robot.joint_num].copy())
                ee_pos = robot.forward_kinematics(data.qpos[:robot.joint_num]).translation
                record_ee_pos.append(ee_pos)

            robot.apply_torque(tau)
            mj.mj_step(model, data)
            
            # data.qpos[:robot.joint_num] = q_des
            # mj.mj_forward(model, data)
            
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        print(q_des)


    if args.record:
        record_qpos = np.array(record_qpos)
        record_qvel = np.array(record_qvel)
        record_ee_pos = np.array(record_ee_pos)

        # plot the end effector position
        import matplotlib.pyplot as plt
        plt.plot(record_ee_pos[:, 0], record_ee_pos[:, 1], label="End Effector Position")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.show()