import numpy as np
from utils.math import commutation_matrix, vech_expansion_matrix

class RbfNetworks:
    def __init__(self, 
                 network_num, 
                 input_num, 
                 output_num,
                 Gamma: np.ndarray,
                 reg_leak: float,
                 width: float,
                 init_adaptive_W: np.ndarray,
                 center_range: tuple[float, float] = (-np.pi, np.pi)
                ):
        assert Gamma.shape == (network_num, network_num)
        assert init_adaptive_W.shape == (network_num, output_num)
        assert center_range[0] < center_range[1], "Center range is invalid."
        
        self.net_num = network_num
        self.input_num = input_num
        self.output_num = output_num

        self.Gamma = Gamma
        self.reg_leak = reg_leak
        self.width = width

        self.centers = np.random.uniform(low=center_range[0], high=center_range[1], size=(network_num, input_num))

        self.W = init_adaptive_W.copy()
        
        
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward 
        Phi: shape (num,)
        Psi: shape (num, input_num)
        """
        assert x.shape == (self.input_num,)

        diff = x - self.centers # (num, input_num)
        Phi = np.exp(-np.sum(diff ** 2, axis=1) / (2 * self.width ** 2)) # (num,)
        # Ψᵢⱼ = ∂Φᵢ/∂xⱼ = -((xⱼ - cᵢⱼ)/σ²) · Φᵢ
        Psi = - diff / (self.width ** 2) * Phi[:, None] # (num, input_num)
        return Phi, Psi

class AdaptiveControlLaw:
    def __init__(self, 
                 dof_num, 
                 dt: float,
                 Lambda: np.ndarray,
                 K: np.ndarray,
                 eps_s = 0.1,
                 gammas: list[float] = [0.01, 0.01, 0.01], 
                 kappas: list[float] = [0.0001, 0.0001, 0.0001],
                 rho0s: list[float] = [0.0, 0.0, 0.0],
                 ):
        
        self.dof_num = dof_num
        self.dt = dt
        
        assert Lambda.shape == (dof_num, dof_num)
        assert K.shape == (dof_num, dof_num)
        self.Lambda = Lambda
        self.K = K
        
        self.eps_s = eps_s
        
        assert len(gammas) == 3
        assert len(kappas) == 3
        assert len(rho0s) == 3
        self.gammas = np.array(gammas)
        self.kappas = np.array(kappas)

        self.rho = np.array(rho0s)

        # ------------------------------------------------
        # set RBF networks

        # D network
        num_d = 30
        num_d_output = dof_num * (dof_num + 1) // 2
        self.d_net = RbfNetworks(
            network_num=num_d, 
            input_num=self.dof_num,
            output_num=num_d_output,
            Gamma=np.eye(num_d)*1.0,
            reg_leak=0.01,
            width=3.0,
            init_adaptive_W=0.1 * np.random.randn(num_d, num_d_output),
        )
        
        # g network
        num_g = 80
        self.g_net = RbfNetworks(
            network_num=num_g,
            input_num=dof_num,
            output_num=dof_num,
            Gamma=np.eye(num_g)*10.0,
            reg_leak=0.01,
            width=15.0,
            init_adaptive_W=np.zeros((num_g, dof_num)),
            # center_range=(-0.5, 0.5)
        )
        
        # f network
        num_f = 30
        self.f_net = RbfNetworks(
            network_num=num_f,
            input_num=dof_num,
            output_num=dof_num,
            Gamma=np.eye(num_f)*5.0,
            reg_leak=0.01,
            width=3.0,
            init_adaptive_W=np.zeros((num_f, dof_num)),
        )

        # ------------------------------------------------
        
        # get Upsilon matrix
        Ut = commutation_matrix(dof_num)
        Uh = vech_expansion_matrix(dof_num)
        self.Upsilon = (np.eye(self.dof_num**2) + Ut) @ Uh # (dof_num**2, num_d_output)
        
        # with np.printoptions(threshold=np.inf, linewidth=np.inf):
        #     print("Upsilon: \n", self.Upsilon)


    def compute(self, q:np.ndarray, dq:np.ndarray, q_des:np.ndarray, dq_des:np.ndarray, ddq_des:np.ndarray):
        # errors
        e = q - q_des
        s = dq - dq_des + self.Lambda @ e
        v = dq - s
        vdot = ddq_des - self.Lambda @ (dq - dq_des)
        
        s_norm = np.linalg.norm(s)
        v_norm = np.linalg.norm(v)
        vdot_norm = np.linalg.norm(vdot)
        dq_norm = np.linalg.norm(dq)
        
        # get the features of rbf network
        phi_d, psi_d = self.d_net.forward(q) # (num_d, ), (num_d, input_num)
        phi_g, _ = self.g_net.forward(q) # (num_g, 1)
        phi_f, _ = self.f_net.forward(dq) # (num_f, 1)

        # update estimate from networks for M, g, f
        M_hat_vec = self.Upsilon @ (self.d_net.W.T @ phi_d)
        M_hat = M_hat_vec.reshape(self.dof_num, self.dof_num, order='F')
        g_hat = self.g_net.W.T @ phi_g
        f_hat = self.f_net.W.T @ phi_f
        
        # C_hat acting on v
        UWPsi = self.Upsilon @ (self.d_net.W.T @ psi_d) # (dof_num**2, input_num)
        term1 = 0.5 * (np.kron(dq[None, :], np.eye(self.dof_num)) @ UWPsi) @ v
        term2 = 0.5 * (np.kron(v [None, :], np.eye(self.dof_num)) @ UWPsi) @ dq
        term3 = 0.5 * (psi_d.T @ self.d_net.W @ self.Upsilon.T) @ np.kron(v, dq)
        C_hat_v = term1 + term2 - term3

        # Robust sliding term 
        phi_s = s / np.sqrt(s_norm**2 + self.eps_s**2)
        rho_gain = self.rho[0] + self.rho[1] * vdot_norm + self.rho[2] * dq_norm * v_norm

        # ------------------------------------
        # update adaptive law
        temp_1 = phi_d[:, None] @ (np.kron(vdot, s)[None, :] @ self.Upsilon) # (num_d, num_d_output)
        temp_2 = 0.5 * (psi_d @ v)[:, None] @ (np.kron(dq, s)[None, :] @ self.Upsilon) # (num_d, num_d_output)
        temp_3 = 0.5 * (psi_d @ dq)[:, None] @(np.kron(v, s)[None, :] @ self.Upsilon) # (num_d, num_d_output)
        temp_4 = - 0.5 * (psi_d @ s)[:, None] @(np.kron(v, dq)[None, :] @ self.Upsilon)
        W_d_dot = - self.d_net.Gamma @ (temp_1 + temp_2 + temp_3 + temp_4 + self.d_net.reg_leak * self.d_net.W )
        
        W_g_dot = - self.g_net.Gamma @ (phi_g[:, None] @ s[None, :] + self.g_net.reg_leak * self.g_net.W)
        W_f_dot = - self.f_net.Gamma @ (phi_f[:, None] @ s[None, :] + self.f_net.reg_leak * self.f_net.W)
        
        rho_0_dot = self.gammas[0] * ((s_norm - self.eps_s) - self.kappas[0] * self.rho[0])
        rho_1_dot = self.gammas[1] * ((s_norm - self.eps_s) * vdot_norm - self.kappas[1] * self.rho[1])
        rho_2_dot = self.gammas[2] * ((s_norm - self.eps_s) * dq_norm * v_norm - self.kappas[2] * self.rho[2])
        
        # update using forward euler
        self.d_net.W += W_d_dot * self.dt
        self.g_net.W += W_g_dot * self.dt
        self.f_net.W += W_f_dot * self.dt

        self.rho[0] += rho_0_dot * self.dt
        self.rho[1] += rho_1_dot * self.dt
        self.rho[2] += rho_2_dot * self.dt

        # ------------------------------------
        
        # print((M_hat @ vdot).shape)
        # print(C_hat_v.shape)
        # print(g_hat.shape)
        # print(f_hat.shape)
        # print((self.K @ s).shape)
        # print(( rho_gain * phi_s).shape)
        
        tau = M_hat @ vdot + C_hat_v + g_hat + f_hat - self.K @ s - rho_gain * phi_s
        
        return tau

    def reset(self):
        # reset adaptive laws
        self.d_net.W = np.random.randn(*self.d_net.W.shape) * 0.1
        self.g_net.W = np.zeros_like(self.g_net.W)
        self.f_net.W = np.zeros_like(self.f_net.W)

        self.rho = np.zeros_like(self.rho)
