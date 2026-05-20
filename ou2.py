import numpy as np

class HumanLikeSignalGenerator:
    """
    生成仿人类遥控信号的类
    - 输出: vx, vy, vz, yaw
    - 使用 OU 过程，信号平滑，可随机变化
    - 支持趋势 trend 模拟手持续推杆
    """

    def __init__(self, dt=0.02, sigma_vel=0.5, sigma_yaw=0.05,
                 theta=0.15, max_vel=5.0, max_yaw=np.pi,
                 trend_vel=0.0, trend_yaw=0.0, mu_yaw=0.0):
        """
        dt: 仿真步长
        sigma_vel: 速度噪声幅度
        sigma_yaw: yaw 噪声幅度
        theta: OU process 回归系数
        max_vel: 最大速度 (m/s)
        max_yaw: 最大 yaw (rad)
        trend_vel: 每步速度增加量，用于模拟手持续推杆
        trend_yaw: 每步 yaw 增量，用于模拟手持续旋转
        mu_yaw: yaw OU 长期均值
        """
        self.dt = dt
        self.theta = theta
        self.sigma_vel = sigma_vel
        self.sigma_yaw = sigma_yaw
        self.max_vel = max_vel
        self.max_yaw = max_yaw
        self.trend_vel = trend_vel
        self.trend_yaw = trend_yaw
        self.mu_yaw = mu_yaw

        # 初始化状态
        self.reset()

    def reset(self):
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw = 0.0
        return np.array([self.vx, self.vy, self.vz, self.yaw], dtype=np.float32)

    def step(self):
        """更新信号"""
        # vx, vy, vz OU + trend
        self.vx += self.trend_vel + self.theta * (-self.vx) * self.dt + self.sigma_vel * np.sqrt(self.dt) * np.random.randn()
        self.vy += self.trend_vel + self.theta * (-self.vy) * self.dt + self.sigma_vel * np.sqrt(self.dt) * np.random.randn()
        self.vz += self.trend_vel + self.theta * (-self.vz) * self.dt + self.sigma_vel * np.sqrt(self.dt) * np.random.randn()

        # 限幅
        self.vx = np.clip(self.vx, -self.max_vel, self.max_vel)
        self.vy = np.clip(self.vy, -self.max_vel, self.max_vel)
        self.vz = np.clip(self.vz, -self.max_vel, self.max_vel)

        # yaw OU + trend
        self.yaw += self.trend_yaw + self.theta * (self.mu_yaw - self.yaw) * self.dt + self.sigma_yaw * np.sqrt(self.dt) * np.random.randn()
        self.yaw = np.clip(self.yaw, -self.max_yaw, self.max_yaw)

        return np.array([self.vx, self.vy, self.vz, self.yaw], dtype=np.float32)


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 创建信号生成器
    signal_gen = HumanLikeSignalGenerator(
        dt=0.02,
        sigma_vel=0.3,
        sigma_yaw=0.3,
        theta=0.2,
        max_vel=5.0,
        max_yaw=np.pi,
        trend_vel=0.0,   # 模拟手持续往上推
        trend_yaw=0.0,  # 模拟手持续旋转
        mu_yaw=0.0
    )

    # 生成信号
    signal = []
    signal_gen.reset()
    for _ in range(500):  # 10秒信号
        s = signal_gen.step()
        signal.append(s)

    signal = np.array(signal)

    # 绘图
    plt.figure(figsize=(10,6))
    plt.plot(signal[:,0], label='vx')
    plt.plot(signal[:,1], label='vy')
    plt.plot(signal[:,2], label='vz')
    plt.plot(signal[:,3], label='yaw')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('仿人类遥控信号示例(OU + trend)')
    plt.legend()
    plt.grid(True)
    plt.show()