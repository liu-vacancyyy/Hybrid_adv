import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 生成100帧的数据（模拟飞机的XYZ位置和姿态角度）
def generate_data(num_frames=100):
    # 随机生成飞机的位置（在x、y、z方向上的随机值）
    positions = np.c_[np.linspace(0, 10, num_frames),
                      np.sin(np.linspace(0, 10, num_frames)),
                      np.cos(np.linspace(0, 10, num_frames))]
    
    # 随机生成每个位置的pitch, roll, yaw角度（单位为度）
    pitch = np.linspace(0, 30, num_frames)
    roll = np.linspace(0, 45, num_frames)
    yaw = np.linspace(0, 90, num_frames)
    
    return positions, pitch, roll, yaw

# 定义函数，输入固定翼飞机的xyz位置和pitch, roll, yaw
def plot_trajectory_and_orientation(positions, pitch, roll, yaw, intervall=20):
    # 将角度转换为弧度
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)
    
    # 计算飞机姿态的旋转矩阵
    def rotation_matrix(pitch, roll, yaw):
        # 计算飞机的姿态旋转矩阵
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        # 总旋转矩阵是三个旋转矩阵的乘积
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R
    
    # 设置绘图窗口
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制飞机的轨迹（位置）
    positions = np.array(positions)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Trajectory', color='b')
    
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置图形标题
    ax.set_title('3D Trajectory and Orientation of Aircraft')
    
    # 显示图例
    ax.legend()

    # 初始化飞机朝向的箭头
    quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='r', length=0.5, normalize=True)
    
    # 更新函数
    def update(frame):
        # 获取当前位置和姿态角度
        pos = positions[frame]
        p, r, y = pitch[frame], roll[frame], yaw[frame]
        
        # 获取旋转矩阵
        R = rotation_matrix(p, r, y)
        
        # 飞机的朝向向量（假设飞机朝向是沿着x轴）
        direction = np.array([1, 0, 0])  # 飞机朝向沿着x轴
        
        # 旋转朝向向量
        rotated_direction = np.dot(R, direction)
        
        # 更新飞机朝向箭头
        quiver.remove()  # 删除之前的箭头
        quiver = ax.quiver(pos[0], pos[1], pos[2], rotated_direction[0], rotated_direction[1], rotated_direction[2], length=0.5, color='r', normalize=True)
        
        return quiver

    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(positions), interval=intervall)
    
    # 显示动画
    plt.show()

# 生成100帧的模拟数据
positions, pitch, roll, yaw = generate_data(num_frames=100)

# 绘制轨迹和朝向
plot_trajectory_and_orientation(positions, pitch, roll, yaw, intervall=50)  # 每50毫秒更新一次
