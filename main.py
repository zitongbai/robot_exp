import time
import os
import numpy as np
import argparse
import enum
import mujoco as mj
import mujoco.viewer

import pinocchio as pin

import matplotlib.pyplot as plt

from robot import Robot
from utils.math import lemniscate_trajectory
from control_law import AdaptiveControlLaw

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
    parser.add_argument("-t", "--type", default="sin", choices=["sin", "eight"], help="Type of trajectory")

    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.robot == 'fr3':
        xml_path = os.path.join(current_dir, 'resources/fr3_mj_description/fr3_scene.xml')
    else:
        raise ValueError(f"Unknown robot: {args.robot}")

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    model.opt.timestep = 0.001

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
    
    init_q = np.array([0.0, 0.4, 0.0, -1.4, 0.0, 1.68, 0.0])
    init_dq = np.zeros(7)
    
    data.qpos[:robot.joint_num] = init_q.copy()
    data.qvel[:robot.joint_num] = init_dq.copy()
    print("Initial joint positions:", data.qpos[:robot.joint_num])
    print("Initial joint velocities:", data.qvel[:robot.joint_num])
    
    if args.type == "sin":
        init_q_des = np.array([0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.0])
        init_dq_des = np.zeros(robot.joint_num)
    elif args.type == "eight":
        offset = np.array([0.4, 0, 0.6])
        target_pos, target_vel, _ = lemniscate_trajectory(0, a=0.2, offset=offset)
        init_q_des, init_dq_des = robot.inverse_kinematics(
            target_pos=target_pos,
            target_ori=np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ]),
            target_lin_vel=target_vel,
            target_ang_vel=np.zeros(3),
        )
    else:
        raise ValueError("Unknown trajectory type")

    stage = SimStage.INIT

    record_qpos = []
    record_qvel = []
    record_ee_pos = []
    
    record_qpos_des = []
    record_qvel_des = []
    
    record_idx = 0
    
    # adap_ctrl = AdaptiveControlLaw(
    #     dof_num=robot.joint_num,
    #     dt = model.opt.timestep,
    #     Lambda=np.diag([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]),
    #     K=np.diag([10,40,10,40,5,5,3.5]) * 0.4,
    # )
    
    adap_ctrl = AdaptiveControlLaw(
        dof_num=robot.joint_num,
        dt = model.opt.timestep,
        Lambda=np.diag([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),  # 增加Lambda值
        K=np.diag([5,80,5,30,1.2,3,0.5])*0.8,  # 增加K值
        eps_s=0.01,  # 减小eps_s
        gammas=[0.01, 0.01, 0.01],  # 增加自适应增益
        kappas=[0.0001, 0.0001, 0.0001],  # 增加泄漏项
    )
    
    dq_des_last = np.zeros(robot.joint_num)
    first = True

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        sim_time = 0
        dt = model.opt.timestep
        while viewer.is_running() and time.time() - start < args.duration:
            step_start = time.time()

            if stage == SimStage.INIT:
                q_des = init_q_des
                dq_des = init_dq_des
                ddq_des = np.zeros(robot.joint_num)  # 添加初始化
                
                tau = robot.ctrl_fb_lin(
                    q_des=init_q_des,
                    dp_des=init_dq_des,
                    Kp=np.array([200, 200, 200, 200, 200, 200, 200]),
                    # Kd=np.array([10, 10, 10, 10, 10, 10, 10])
                )
                
                eps = 1
                if np.linalg.norm(data.qpos[:robot.joint_num] - q_des) < eps:
                    print("Switch to TRAJECTORY stage")
                    stage = SimStage.TRAJECTORY
                    traj_start = sim_time
                    
                    if args.record:
                        # save the record_idx
                        traj_start_idx = record_idx + 1
                
            elif stage == SimStage.TRAJECTORY:

                t = sim_time - traj_start
                if args.type == "sin":
                    amplitude = 0.2
                    frequency = 0.5
                    
                    q_des = init_q_des + amplitude * np.sin(frequency * t * np.ones_like(q_des)) 
                    dq_des =  amplitude * frequency * np.cos(frequency * t * np.ones_like(dq_des))  # 修复：正确的一阶导数
                    ddq_des = -amplitude * frequency**2 * np.sin(frequency * t * np.ones_like(ddq_des))  # 修复：正确的二阶导数

                elif args.type == "eight":

                    target_pos, target_vel, _ = lemniscate_trajectory(t, a=0.2, offset=np.array([0.4,0,0.6]))
                    
                    if first:
                        ddq_des = np.zeros(robot.joint_num)
                        first = False
                    else:
                        ddq_des = (dq_des - dq_des_last) / dt
                    dq_des_last = dq_des

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
                    Kp=np.array([200, 200, 200, 200, 200, 200, 200]),
                    # Kd=np.array([10, 10, 10, 10, 10, 10, 10])
                )
                
                # scale = 0.0
                # g = pin.rnea(pin_model, pin_data, robot.dof_pos, np.zeros(robot.joint_num), np.zeros(robot.joint_num))
                
                # tau = adap_ctrl.compute(
                #     q=robot.dof_pos,
                #     dq=robot.dof_vel,
                #     q_des=q_des,
                #     dq_des=dq_des,
                #     ddq_des=ddq_des
                # ) + scale * g
                
                # tau = tau_1 + tau_2
            
            if args.record:
                record_qpos.append(data.qpos[:robot.joint_num].copy())
                record_qvel.append(data.qvel[:robot.joint_num].copy())
                ee_pos = robot.forward_kinematics(data.qpos[:robot.joint_num]).translation
                record_ee_pos.append(ee_pos)
                record_qpos_des.append(q_des.copy())
                record_qvel_des.append(dq_des.copy())
                
                record_idx += 1

            robot.apply_torque(tau)
            mj.mj_step(model, data)
            
            # data.qpos[:robot.joint_num] = q_des
            # mj.mj_forward(model, data)
            
            viewer.sync()
            
            sim_time += dt
            print(f"\rTime: {sim_time:.2f}s / {args.duration}s", end="")
            
            # time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)


    if args.record:
        record_qpos = np.array(record_qpos)
        record_qvel = np.array(record_qvel)
        record_ee_pos = np.array(record_ee_pos)
        record_qpos_des = np.array(record_qpos_des)
        record_qvel_des = np.array(record_qvel_des)

        # plot the qpos and its desired value, all in one figure
        plt.figure()
        for i in range(robot.joint_num):
            plt.subplot(robot.joint_num, 1, i + 1)
            plt.plot(record_qpos[:, i], label="Actual")
            plt.plot(record_qpos_des[:, i], label="Desired")
            if 'traj_start_idx' in locals():
                plt.axvline(x=traj_start_idx, color='red', linestyle='--', alpha=0.7, label='Trajectory Start')
            plt.ylabel(f"Joint {i + 1} Position")
            plt.legend()
        plt.xlabel("Time Step")
        plt.show()
