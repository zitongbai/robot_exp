import numpy as np

import mujoco as mj
import pinocchio as pin


class Robot:
    def __init__(self, mj_model, mj_data, pin_model:pin.Model, base_link_name: str = "fr3/base"):
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.pin_model: pin.Model = pin_model
        self.pin_data: pin.Data = pin_model.createData()
        
        self.base_link_name = base_link_name
        self.base_link_id = self.pin_model.getFrameId(base_link_name)
        print("Base link ID:", self.base_link_id)

        self.joint_names = [mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(mj_model.njnt)]
        self.joint_num = len(self.joint_names)
        
        self._init_buffer()
    
    def _init_buffer(self):
        self.q: np.ndarray = self.mj_data.sensordata[:self.joint_num]
        self.dq: np.ndarray = self.mj_data.sensordata[self.joint_num:2*self.joint_num]

        # self.q: np.ndarray = self.mj_data.qpos[:self.joint_num]
        # self.dq: np.ndarray = self.mj_data.qvel[:self.joint_num]
        
        self.tau: np.ndarray = np.zeros(self.joint_num)
        
    @property
    def dof_pos(self):
        return self.q

    @property
    def dof_vel(self):
        return self.dq

    @property
    def dof_tau(self):
        return self.tau
    
    def apply_torque(self, tau: np.ndarray):
        assert tau.shape == (self.joint_num,), f"Torque vector shape mismatch: {tau.shape} != {self.joint_num}"
        self.mj_data.ctrl[:self.joint_num] = tau

    def inverse_kinematics(self, 
                           target_pos: np.ndarray, 
                           target_ori: np.ndarray = np.eye(3), 
                           target_lin_vel: np.ndarray = np.zeros(3),
                           target_ang_vel: np.ndarray = np.zeros(3),
                           q0: np.ndarray = None,
                           eps = 1e-4, 
                           max_iterations = 1000, 
                           dt = 1e-1,
                           damp = 1e-12
                        ):
        ee_id = self.joint_num
        
        bMdes = pin.SE3(target_ori, target_pos) # base to desired
        
        if q0 is None:
            q = np.array([0.0, 0.4, 0.0, -1.4, 0.0, 1.68, 0.0])
        else:
            q = q0.copy()
            
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        oMb = self.pin_data.oMf[self.base_link_id] # world to base link
        oMdes = oMb * bMdes # world to desired

        i = 0
        while True:
            pin.forwardKinematics(self.pin_model, self.pin_data, q)
            iMd = self.pin_data.oMi[ee_id].actInv(oMdes) # in joint frame
            # pin.updateFramePlacements(self.pin_model, self.pin_data)
            # iMd = self.pin_data.oMf[-1].actInv(oMdes) # we assume the end effector is the last frame
            err = pin.log(iMd).vector
            if np.linalg.norm(err) < eps:
                success = True
                break
            if i >= max_iterations:
                success = False
                break
            J = pin.computeJointJacobian(self.pin_model, self.pin_data, q, ee_id) # in joint frame
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.pin_model, q, dt * v)
            i += 1
        
        if success:
            J = pin.computeJointJacobian(self.pin_model, self.pin_data, q, ee_id) # in joint frame
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            iMb = self.pin_data.oMi[ee_id].actInv(oMb)
            lin_vel = iMb * target_lin_vel
            ang_vel = iMb * target_ang_vel
            v_ee = np.hstack((lin_vel, ang_vel))
            dq = np.linalg.pinv(J) @ v_ee

            return q, dq
        else:
            raise RuntimeError("IK did not converge")
        
    def forward_kinematics(self, q: np.ndarray) -> pin.SE3:
        ee_id = self.joint_num
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        oMb = self.pin_data.oMf[self.base_link_id]
        oMi = self.pin_data.oMi[ee_id]
        bMi = oMb.actInv(oMi)
        return bMi

    def ctrl_fb_lin(self, q_des:np.ndarray, dp_des:np.ndarray, Kp:np.ndarray, Kd:np.ndarray = None):
        # mass matrix
        M = pin.crba(self.pin_model, self.pin_data, self.q)
        # dynamic drift -- Coriolis, centrifugal, gravity
        b = pin.rnea(self.pin_model, self.pin_data, self.q, self.dq, np.zeros_like(self.dq))
        # Gain matrix
        Kp = np.diag(Kp)
        if Kd is None:
            Kd = np.sqrt(Kp)*2
        else:
            Kd = np.diag(Kd)
        pd_law = Kp @ (q_des - self.q) + Kd @ (dp_des - self.dq)
        tau = M @ pd_law + b
        return tau
    
    
    
