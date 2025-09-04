import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lemniscate_trajectory(t, a=0.2, offset:np.ndarray=np.array([0.5,0,0.6])):
    """Given t, returns the (x, y, z) position on a lemniscate trajectory.
    ref: https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli

    Returns:
        pos (3,), vel (3,), acc (3,)
    """
    assert offset.shape == (3,)
    s = np.sin(t)
    c = np.cos(t)

    D = 1 + s**2

    x = a * s * c / D + offset[0]
    y = a * c / D + offset[1]
    z = offset[2]

    # intermediate numerators for velocities
    Nx = c**2 - s**2 - s**4 - s**2 * c**2
    Ny = -s - s**3 - 2 * s * c**2

    vx = a * Nx / (D**2)
    vy = a * Ny / (D**2)
    vz = 0.0

    # derivatives needed for accelerations
    # D' = 2 s c
    Dp = 2 * s * c
    # Nx' = -6 s c  (derived)
    Nx_p = -6 * s * c
    # Ny' = -3 c^3 (derived)
    Ny_p = -3 * c**3

    ax = a * (Nx_p * D - 2 * Nx * Dp) / (D**3)
    ay = a * (Ny_p * D - 2 * Ny * Dp) / (D**3)
    az = 0.0

    return np.array([x, y, z]), np.array([vx, vy, vz]), np.array([ax, ay, az])


def commutation_matrix(n):
    """生成 n×n 的 commutation matrix K"""
    K = np.zeros((n**2, n**2))
    for i in range(n):
        for j in range(n):
            row = i * n + j
            col = j * n + i
            K[row, col] = 1
    return K

def vech_expansion_matrix(n):
    """vech(D) -> vec(D) 的映射"""
    m = n * (n + 1) // 2
    Uh = np.zeros((n**2, m))
    k = 0
    for i in range(n):
        for j in range(i + 1):
            Uh[i * n + j, k] = 1
            k += 1
    return Uh


if __name__ == "__main__":
    # 生成轨迹数据
    t_dense = np.linspace(0, 2*np.pi, 1000)
    positions = []
    velocities = []
    
    for t in t_dense:
        pos, vel, acc = lemniscate_trajectory(t)
        positions.append(pos)
        velocities.append(vel)
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 5))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    
    # 标记起始点
    start_pos = positions[0]
    ax1.scatter(start_pos[0], start_pos[1], start_pos[2], 
               color='green', s=100, marker='o', label='Start Point')
    
    # 降采样绘制速度箭头
    downsample_factor = 50  # 每50个点取一个
    arrow_scale = 0.1  # 箭头大小缩放因子
    
    for i in range(0, len(positions), downsample_factor):
        pos = positions[i]
        vel = velocities[i]
        ax1.quiver(pos[0], pos[1], pos[2], 
                   vel[0], vel[1], vel[2], 
                   length=arrow_scale, 
                   color='red', 
                   arrow_length_ratio=0.3)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Lemniscate Trajectory and Velocity Vectors')
    ax1.legend()
    ax1.grid(True)
    
    # 2D xy平面投影
    ax2 = fig.add_subplot(122)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory Projection')

    # 在2D图中标记起始点
    ax2.scatter(start_pos[0], start_pos[1], 
               color='green', s=100, marker='o', label='Start Point')

    # 在xy平面绘制速度箭头
    for i in range(0, len(positions), downsample_factor):
        pos = positions[i]
        vel = velocities[i]
        ax2.arrow(pos[0], pos[1], 
                  vel[0] * arrow_scale, vel[1] * arrow_scale,
                  head_width=0.01, head_length=0.01, 
                  fc='red', ec='red', alpha=0.7)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane Projection and Velocity Vectors')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # 打印一些统计信息
    print(f"Number of trajectory points: {len(positions)}")
    print(f"Velocity range: {np.min(np.linalg.norm(velocities, axis=1)):.3f} - {np.max(np.linalg.norm(velocities, axis=1)):.3f}")