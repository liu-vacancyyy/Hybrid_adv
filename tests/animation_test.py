import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # 用于三维绘图

# 创建三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 3D坐标轴
xdata, ydata, zdata = [], [], []
u_data, v_data, w_data = [], [], []  # 箭头的方向向量（u, v, w）

# 初始化箭头
arrow = ax.quiver([], [], [], [], [], [], color='r', length=0.1)

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)  # 设置Z轴范围
    return arrow,

def update(frame):
    # 更新数据
    xdata.append(frame)
    ydata.append(np.sin(frame))
    zdata.append(np.cos(frame))
    
    # 计算箭头的方向：这里使用单位向量来控制箭头方向
    u = np.cos(frame)  # u方向
    v = np.sin(frame)  # v方向
    w = np.cos(frame)  # w方向
    
    u_data.append(u)
    v_data.append(v)
    w_data.append(w)
    
    # 更新箭头的位置和方向
    arrow.set_offsets(np.c_[xdata, ydata])  # 更新箭头的起点
    arrow.set_verts([np.c_[u_data, v_data, w_data]])  # 更新箭头的方向
    return arrow,

# 创建动画
ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)

plt.show()


