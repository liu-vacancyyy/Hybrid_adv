"""
Dryden 湍流模型生成器

使用 Dryden 谱模型生成大气湍流，该模型在航空航天仿真中广泛应用。
输出体坐标系（body-frame）下的风扰动 (u, v, w)。

参考文献：
    MIL-STD-1797B (2011): 轻度湍流
    https://en.wikipedia.org/wiki/Dryden_gust_model
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os


class DrydenTurbulenceModel:
    """
    Dryden 湍流模型，用于低空飞行 (h < 1000 ft / ~300m)。

    该模型通过低通滤波白噪声生成相关的风扰动 (u, v, w)。
    参数随高度和参考空速变化。
    """

    def __init__(self, altitude_m=10.0, airspeed_ms=5.0, dt=0.02, duration_s=30.0):
        """
        初始化 Dryden 湍流生成器。

        参数：
            altitude_m (float): 飞机高度（米）。影响湍流强度。
                               典型范围：5-500 m（低空模型）。
            airspeed_ms (float): 参考空速（m/s）。影响强度和频率。
                                典型范围：1-20 m/s。
            dt (float): 仿真时间步长（默认 0.02s = 50 Hz）。
            duration_s (float): 总仿真时长（秒）。

        属性：
            sigma_u, sigma_v, sigma_w (float): 各轴向风速均方根 (m/s)。
            L_u, L_v, L_w (float): 各轴向长度尺度 (m)。
            H_u, H_v, H_w: 传递函数分子/分母。
            z_u, z_v, z_w: 数字滤波器状态。
        """
        self.altitude = float(altitude_m)
        self.airspeed = float(airspeed_ms)
        self.dt = float(dt)
        self.duration = float(duration_s)
        self.num_steps = int(self.duration / self.dt)

        # --- Dryden 参数（低空、轻度湍流） ---
        # 参考：MIL-STD-1797B
        # 强度随高度增加（高达 ~300m），之后饱和
        self._compute_scales()
        self._design_filters()
        self._init_states()

    def _compute_scales(self):
        """根据高度计算 Dryden 强度和长度尺度。"""
        # 将高度限制在有效范围内（模型设计用于 h < 300m）
        h = np.clip(self.altitude, 10.0, 300.0)

        # 均方根强度 (m/s) — 随高度增加
        # 轻度湍流强度取自 MIL-STD-1797B
        self.sigma_u = 0.1 * h ** 0.2  # 纵向 (m/s)
        self.sigma_w = 0.1 * h ** 0.2  # 竖直 (m/s)
        self.sigma_v = self.sigma_u     # 横向 ≈ 纵向

        # 长度尺度 (m) — 相关距离
        # 典型值：200-500m 低空
        self.L_u = 200.0 + 0.5 * h     # 纵向长度尺度
        self.L_v = self.L_u             # 横向 ≈ 纵向
        self.L_w = 50.0 + 0.2 * h      # 竖直（通常更短）

    def _design_filters(self):
        """设计严执于推读的传递函数中步转事算法（MIL-STD-1797B二阶）。"""
        # --- 纵向滤波器 H_u(s) ---
        # 标准 Dryden: H_u(s) = sigma_u * sqrt(2*L_u/pi) / (1 + L_u/V * s)
        # 但我们使用二阶模型以获得更好的谱匹配
        V = max(self.airspeed, 0.1)  # 不会除以零
        
        # 二阶收整的传递函数 (Dryden 标准)
        # 分子：正比于 sigma * sqrt(2*L/pi)
        sqrt_2L_pi_u = np.sqrt(2 * self.L_u / np.pi)
        sqrt_2L_pi_w = np.sqrt(2 * self.L_w / np.pi)

        # 极点位于 1/(L/V)
        pole_u = V / self.L_u
        pole_w = V / self.L_w

        # 推导时间数值分子
        num_u = self.sigma_u * sqrt_2L_pi_u * np.array([pole_u])
        num_w = self.sigma_w * sqrt_2L_pi_w * np.array([pole_w])
        
        # 推导时间数值分母: (s + pole)
        den_u = np.array([1.0, pole_u])
        den_w = np.array([1.0, pole_w])

        # 使用双线性转换传化为离散时间 (s = 2/dt * (z-1)/(z+1))
        sys_u_d = signal.cont2discrete((num_u, den_u), self.dt, method='bilinear')
        sys_w_d = signal.cont2discrete((num_w, den_w), self.dt, method='bilinear')

        # 处理旧每非最新的 scipy API
        if isinstance(sys_u_d, tuple):
            self.num_u = sys_u_d[0].flatten() if hasattr(sys_u_d[0], 'flatten') else np.asarray(sys_u_d[0]).flatten()
            self.den_u = sys_u_d[1].flatten() if hasattr(sys_u_d[1], 'flatten') else np.asarray(sys_u_d[1]).flatten()
            self.num_w = sys_w_d[0].flatten() if hasattr(sys_w_d[0], 'flatten') else np.asarray(sys_w_d[0]).flatten()
            self.den_w = sys_w_d[1].flatten() if hasattr(sys_w_d[1], 'flatten') else np.asarray(sys_w_d[1]).flatten()
        else:
            self.num_u = np.asarray(sys_u_d.num).flatten()
            self.den_u = np.asarray(sys_u_d.den).flatten()
            self.num_w = np.asarray(sys_w_d.num).flatten()
            self.den_w = np.asarray(sys_w_d.den).flatten()

        # 横向 (v) 通道：简化，通常仅是一阶
        pole_v = V / self.L_v
        num_v = self.sigma_v * np.sqrt(2 * self.L_v / np.pi) * np.array([pole_v])
        den_v = np.array([1.0, pole_v])
        sys_v_d = signal.cont2discrete((num_v, den_v), self.dt, method='bilinear')
        
        if isinstance(sys_v_d, tuple):
            self.num_v = sys_v_d[0].flatten() if hasattr(sys_v_d[0], 'flatten') else np.asarray(sys_v_d[0]).flatten()
            self.den_v = sys_v_d[1].flatten() if hasattr(sys_v_d[1], 'flatten') else np.asarray(sys_v_d[1]).flatten()
        else:
            self.num_v = np.asarray(sys_v_d.num).flatten()
            self.den_v = np.asarray(sys_v_d.den).flatten()

    def _init_states(self):
        """初始化滤波器状态缓冲区。"""
        # 对于一阶创二阶滤波器，最多需要 1 个爾州変量
        max_order = max(len(self.num_u), len(self.num_v), len(self.num_w))
        self.z_u = np.zeros(max_order - 1)
        self.z_v = np.zeros(max_order - 1)
        self.z_w = np.zeros(max_order - 1)

    def _apply_filter(self, white_noise, num, den, state):
        """示例配会使用第二阶滤波法将 IIR 滤波器应用于白噪声。"""
        output = np.zeros_like(white_noise)
        # 规一化分母
        a = den / den[0]
        b = num / den[0]
        
        for i, sample in enumerate(white_noise):
            # 䒢斯函数反馈
            if len(state) > 0:
                y = sample - np.dot(a[1:], state)
            else:
                y = sample
            
            # 前馈
            if len(state) > 0:
                output[i] = b[0] * y + np.dot(b[1:], state)
                # 更新缺子状态
                new_state = np.concatenate([[y], state[:-1]])
                state[:] = new_state
            else:
                output[i] = b[0] * y
        
        return output

    def generate(self, seed=None):
        """
        生成湍流时间序列。

        参数：
            seed (int, optional): 随机数种子，与可重现性。

        返回：
            dict: 时间序列，键为 't', 'u', 'v', 'w' (m/s)。
        """
        if seed is not None:
            np.random.seed(seed)

        # 白噪声输入（零均值，命副方差 1）
        wn_u = np.random.randn(self.num_steps)
        wn_v = np.random.randn(self.num_steps)
        wn_w = np.random.randn(self.num_steps)

        # 应用滤波器
        u_wind = self._apply_filter(wn_u, self.num_u, self.den_u, self.z_u.copy())
        v_wind = self._apply_filter(wn_v, self.num_v, self.den_v, self.z_v.copy())
        w_wind = self._apply_filter(wn_w, self.num_w, self.den_w, self.z_w.copy())

        t = np.arange(self.num_steps) * self.dt

        return {
            't': t,
            'u': u_wind,
            'v': v_wind,
            'w': w_wind,
        }

    def get_stats(self, data):
        """计算並显示风素统计。"""
        u, v, w = data['u'], data['v'], data['w']
        rms_u = np.sqrt(np.mean(u ** 2))
        rms_v = np.sqrt(np.mean(v ** 2))
        rms_w = np.sqrt(np.mean(w ** 2))
        windspeed_total = np.sqrt(u ** 2 + v ** 2 + w ** 2)

        return {
            'rms_u': rms_u,
            'rms_v': rms_v,
            'rms_w': rms_w,
            'total_rms': np.sqrt(np.mean(windspeed_total ** 2)),
            'max_u': np.max(np.abs(u)),
            'max_v': np.max(np.abs(v)),
            'max_w': np.max(np.abs(w)),
        }


def plot_turbulence(data, stats, altitude_m=10.0, airspeed_ms=5.0, out_png=None):
    """
    创建全面的湍流可视化。

    参数：
        data (dict): DrydenTurbulenceModel.generate() 的输出。
        stats (dict): DrydenTurbulenceModel.get_stats() 的输出。
        altitude_m (float): 高度，用于澎示。
        airspeed_ms (float): 空速，用于澎示。
        out_png (str, optional): 保存路径。若为 None，则显示嘯形。
    """
    t = data['t']
    u, v, w = data['u'], data['v'], data['w']

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # --- 第 1 列：时间序列 ---
    ax = axes[0, 0]
    ax.plot(t, u, 'b-', lw=1.5, label='u (纵向)')
    ax.axhline(stats['rms_u'], color='b', ls='--', alpha=0.5, label=f"RMS={stats['rms_u']:.3f} m/s")
    ax.axhline(-stats['rms_u'], color='b', ls='--', alpha=0.5)
    ax.set_title('纵向风 (u)')
    ax.set_ylabel('u (m/s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, v, 'g-', lw=1.5, label='v (横向)')
    ax.axhline(stats['rms_v'], color='g', ls='--', alpha=0.5, label=f"RMS={stats['rms_v']:.3f} m/s")
    ax.axhline(-stats['rms_v'], color='g', ls='--', alpha=0.5)
    ax.set_title('横向风 (v)')
    ax.set_ylabel('v (m/s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(t, w, 'r-', lw=1.5, label='w (竖直)')
    ax.axhline(stats['rms_w'], color='r', ls='--', alpha=0.5, label=f"RMS={stats['rms_w']:.3f} m/s")
    ax.axhline(-stats['rms_w'], color='r', ls='--', alpha=0.5)
    ax.set_title('竖直风 (w)')
    ax.set_ylabel('w (m/s)')
    ax.set_xlabel('时间 (s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 第 2 列：谱特性/统计 ---
    # 自相关函数
    ax = axes[0, 1]
    max_lag = min(500, len(u) // 2)
    acf_full = np.correlate(u - u.mean(), u - u.mean(), mode='full') / (np.var(u) * len(u))
    center_idx = len(acf_full) // 2
    acf_u = acf_full[center_idx:center_idx + max_lag]
    lags = np.arange(len(acf_u)) * (t[1] - t[0])
    ax.plot(lags, acf_u, 'b-', lw=1.5, label='u 自相关')
    ax.set_title('自相关函数')
    ax.set_ylabel('ACF')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 水平风向继繻、横横交叉风 (u 对横 u vs v)
    ax = axes[1, 1]
    ax.scatter(u, v, c=t, cmap='viridis', s=10, alpha=0.6)
    ax.set_title('水平风向继繻图 (u vs v)')
    ax.set_xlabel('u (m/s)')
    ax.set_ylabel('v (m/s)')
    ax.axhline(0, color='k', ls='-', alpha=0.2, lw=0.5)
    ax.axvline(0, color='k', ls='-', alpha=0.2, lw=0.5)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label('时间 (s)')

    # 3D 风帖量浅度
    ax = axes[2, 1]
    wind_mag = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    ax.plot(t, wind_mag, 'k-', lw=1.5, label='总风速')
    ax.axhline(stats['total_rms'], color='k', ls='--', alpha=0.5, label=f"RMS={stats['total_rms']:.3f} m/s")
    ax.set_title('总风速大小')
    ax.set_ylabel('|W| (m/s)')
    ax.set_xlabel('时间 (s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Dryden 湍流模型: h={altitude_m}m, V={airspeed_ms}m/s, dt={t[1]-t[0]:.3f}s, \u603b时间={t[-1]:.1f}s',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches='tight')
        print(f"[嘯形] 已保存: {out_png}")
    else:
        plt.show()

    return fig


def main():
    """生成并绘制 Dryden 湍流示例。"""
    import sys
    
    ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    out_dir = os.path.join(ROOT, 'renders', 'result')
    os.makedirs(out_dir, exist_ok=True)

    # 示例 1: 低高度、低空速（悬停状态类似）
    print("\n" + "="*70)
    print("情况 1: 低高度 (10m)、低空速 (2 m/s) [悬停状态类似]")
    print("="*70)
    gen1 = DrydenTurbulenceModel(altitude_m=10.0, airspeed_ms=2.0, dt=0.02, duration_s=30.0)
    data1 = gen1.generate(seed=42)
    stats1 = gen1.get_stats(data1)
    print(f"  u RMS: {stats1['rms_u']:.4f} m/s (max: {stats1['max_u']:.4f})")
    print(f"  v RMS: {stats1['rms_v']:.4f} m/s (max: {stats1['max_v']:.4f})")
    print(f"  w RMS: {stats1['rms_w']:.4f} m/s (max: {stats1['max_w']:.4f})")
    print(f"  Total RMS: {stats1['total_rms']:.4f} m/s")
    plot_turbulence(
        data1, stats1, altitude_m=10.0, airspeed_ms=2.0,
        out_png=os.path.join(out_dir, 'dryden_case1_hover.png')
    )

    # Example 2: Medium altitude and airspeed
    print("\n" + "="*70)
    print("Case 2: Medium altitude (50m), medium airspeed (8 m/s)")
    print("="*70)
    gen2 = DrydenTurbulenceModel(altitude_m=50.0, airspeed_ms=8.0, dt=0.02, duration_s=30.0)
    data2 = gen2.generate(seed=43)
    stats2 = gen2.get_stats(data2)
    print(f"  u RMS: {stats2['rms_u']:.4f} m/s (max: {stats2['max_u']:.4f})")
    print(f"  v RMS: {stats2['rms_v']:.4f} m/s (max: {stats2['max_v']:.4f})")
    print(f"  w RMS: {stats2['rms_w']:.4f} m/s (max: {stats2['max_w']:.4f})")
    print(f"  Total RMS: {stats2['total_rms']:.4f} m/s")
    plot_turbulence(
        data2, stats2, altitude_m=50.0, airspeed_ms=8.0,
        out_png=os.path.join(out_dir, 'dryden_case2_medium.png')
    )

    # Example 3: Higher altitude (turbulence grows)
    print("\n" + "="*70)
    print("Case 3: High altitude (200m), high airspeed (15 m/s)")
    print("="*70)
    gen3 = DrydenTurbulenceModel(altitude_m=200.0, airspeed_ms=15.0, dt=0.02, duration_s=30.0)
    data3 = gen3.generate(seed=44)
    stats3 = gen3.get_stats(data3)
    print(f"  u RMS: {stats3['rms_u']:.4f} m/s (max: {stats3['max_u']:.4f})")
    print(f"  v RMS: {stats3['rms_v']:.4f} m/s (max: {stats3['max_v']:.4f})")
    print(f"  w RMS: {stats3['rms_w']:.4f} m/s (max: {stats3['max_w']:.4f})")
    print(f"  Total RMS: {stats3['total_rms']:.4f} m/s")
    plot_turbulence(
        data3, stats3, altitude_m=200.0, airspeed_ms=15.0,
        out_png=os.path.join(out_dir, 'dryden_case3_high.png')
    )

    print("\n" + "="*70)
    print("All plots saved to:", out_dir)
    print("="*70)


if __name__ == '__main__':
    main()
