import numpy as np

class HumanLikeSignalGenerator:
    """
    生成仿人类遥控信号的类
    - 输出: vx, vy, vz, yaw
    - 信号平滑，可随机变化
    """
    def __init__(self, dt=0.02, sigma_vel=1.0, sigma_yaw=0.1, theta=0.15, max_vel=5.0, max_yaw_rate=np.pi/3):
        """
        dt: 仿真步长
        sigma_vel: 速度噪声幅度
        sigma_yaw: yaw噪声幅度
        theta: OU process 回归系数
        max_vel: 速度限制 (m/s)
        max_yaw_rate: 最大航向变化率 (rad/s)
        """
        self.dt = dt
        self.theta = theta
        self.sigma_vel = sigma_vel
        self.sigma_yaw = sigma_yaw
        self.max_vel = max_vel
        self.max_yaw_rate = max_yaw_rate

        # 初始化状态
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw = 0.0

    def reset(self):
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw = 0.0
        return np.array([self.vx, self.vy, self.vz, self.yaw], dtype=np.float32)

    def step(self):
        """
        更新信号
        """
        # Ornstein-Uhlenbeck 随机过程生成平滑随机速度
        self.vx += self.theta * (-self.vx) * self.dt + self.sigma_vel * np.sqrt(self.dt) * np.random.randn()
        self.vy += self.theta * (-self.vy) * self.dt + self.sigma_vel * np.sqrt(self.dt) * np.random.randn()
        self.vz += self.theta * (-self.vz) * self.dt + self.sigma_vel * np.sqrt(self.dt) * np.random.randn()

        # 限幅
        self.vx = np.clip(self.vx, -self.max_vel, self.max_vel)
        self.vy = np.clip(self.vy, -self.max_vel, self.max_vel)
        self.vz = np.clip(self.vz, -self.max_vel, self.max_vel)

        # yaw 信号平滑变化
        yaw_rate = self.theta * (-self.yaw) + self.sigma_yaw * np.random.randn()
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        self.yaw += yaw_rate * self.dt

        return np.array([self.vx, self.vy, self.vz, self.yaw], dtype=np.float32)

# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    signal_gen = HumanLikeSignalGenerator(dt=0.02, sigma_vel=0.5, sigma_yaw=0.05)
    signal = []
    signal_gen.reset()
    for _ in range(500):  # 10秒信号
        s = signal_gen.step()
        signal.append(s)

    signal = np.array(signal)
    plt.figure(figsize=(10,6))
    plt.plot(signal[:,0], label='vx')
    plt.plot(signal[:,1], label='vy')
    plt.plot(signal[:,2], label='vz')
    plt.plot(signal[:,3], label='yaw')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('仿人类遥控信号示例')
    plt.legend()
    plt.grid(True)
    plt.show()