"""
PID controller evaluation for Simulink angle tracking task.

Uses the same ControlEnv as PPO training, but replaces the neural network
with a classic PID controller.

PID parameters: Kp=5, Ki=0.08, Kd=-2
Control target: theta → target_theta
"""
import os
import sys
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.control_env import ControlEnv


def _t2n(x):
    return x.detach().cpu().numpy()


class PIDController:
    """Discrete PID controller for Simulink theta tracking."""

    def __init__(self, Kp, Ki, Kd, dt, output_limit=0.35,
                 integral_limit=0.08, back_calc_coeff=0.15):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limit = output_limit
        self.integral_limit = integral_limit      # 积分器饱和限制 (-0.08, 0.08)
        self.back_calc_coeff = back_calc_coeff    # 反算系数 Kb

        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0

    def compute(self, error, q):
        """
        Compute PI-qD output from error = target - actual.
        D term uses pitch rate q directly: D = Kd * q (Kd=-2).
        Anti-windup: back-calculation method.
        Returns action in [-output_limit, output_limit].
        """
        # Proportional
        P = self.Kp * error

        # Integral (before back-calculation correction)
        I = self.Ki * self.integral

        # Derivative: use measured pitch rate q instead of numerical diff
        # q = d(theta)/dt, so this is equivalent to D-on-measurement
        D = self.Kd * q

        # Unsaturated output
        output_unsat = P + I + D

        # Saturated output
        output_sat = np.clip(output_unsat, -self.output_limit, self.output_limit)

        # Back-calculation anti-windup:
        # Correct integral based on saturation error: Kb * (u_sat - u_unsat)
        sat_error = output_sat - output_unsat
        self.integral += (error + self.back_calc_coeff * sat_error) * self.dt

        # Clamp integrator to hard limits (-0.08, 0.08)
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        return output_sat


def evaluate_step_response_metrics(theta_seg, target_seg, dt):
    """计算单个episode的超调量、上升时间、调节时间、稳态误差。"""
    theta_seg = np.asarray(theta_seg)
    target_seg = np.asarray(target_seg)
    if theta_seg.size < 3:
        return {
            'overshoot_percent': np.nan,
            'rise_time_s': np.nan,
            'settling_time_s': np.nan,
            'steady_state_error': np.nan,
        }

    y0 = float(theta_seg[0])
    yf = float(np.mean(target_seg))
    amp = yf - y0
    abs_amp = abs(amp)

    # 稳态误差
    tail_len = max(5, int(0.1 * theta_seg.size))
    y_ss = float(np.mean(theta_seg[-tail_len:]))
    steady_state_error = abs(y_ss - yf)

    if abs_amp < 1e-8:
        return {
            'overshoot_percent': 0.0,
            'rise_time_s': np.nan,
            'settling_time_s': 0.0,
            'steady_state_error': steady_state_error,
        }

    # 超调量(%)
    if amp > 0:
        peak = float(np.max(theta_seg))
        overshoot_percent = max(0.0, (peak - yf) / abs_amp * 100.0)
    else:
        valley = float(np.min(theta_seg))
        overshoot_percent = max(0.0, (yf - valley) / abs_amp * 100.0)

    # 上升时间(10% -> 90%)
    y10 = y0 + 0.1 * amp
    y90 = y0 + 0.9 * amp
    idx10, idx90 = None, None
    if amp > 0:
        idx10_arr = np.where(theta_seg >= y10)[0]
        idx90_arr = np.where(theta_seg >= y90)[0]
    else:
        idx10_arr = np.where(theta_seg <= y10)[0]
        idx90_arr = np.where(theta_seg <= y90)[0]
    if idx10_arr.size > 0:
        idx10 = int(idx10_arr[0])
    if idx90_arr.size > 0:
        idx90 = int(idx90_arr[0])
    rise_time_s = (idx90 - idx10) * dt if (idx10 is not None and idx90 is not None and idx90 >= idx10) else np.nan

    # 调节时间(2%误差带)
    band = 0.02 * abs_amp
    settling_idx = None
    err = np.abs(theta_seg - yf)
    for i in range(theta_seg.size):
        if np.all(err[i:] <= band):
            settling_idx = i
            break
    settling_time_s = settling_idx * dt if settling_idx is not None else np.nan

    return {
        'overshoot_percent': overshoot_percent,
        'rise_time_s': rise_time_s,
        'settling_time_s': settling_time_s,
        'steady_state_error': steady_state_error,
    }


def plot_pid_results(dt, appc_buf, tet_buf, target_theta_buf, save_dir='./result'):
    """绘制PID评估结果曲线（前5个episode）"""
    os.makedirs(save_dir, exist_ok=True)
    steps = np.arange(len(tet_buf))
    time_axis = steps * dt

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle('Simulink PID Evaluation (Kp=5, Ki=0.08, Kd=-2)', fontsize=16, fontweight='bold')

    # theta vs target
    ax = axes[0]
    ax.plot(time_axis, tet_buf, color='r', linewidth=1.2, label='actual θ')
    ax.plot(time_axis, target_theta_buf, color='b', linewidth=1.2, linestyle='--', label='target θ')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('θ (rad)')
    ax.set_title('Pitch Angle θ Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # control
    ax = axes[1]
    ax.plot(time_axis, appc_buf, color='g', linewidth=1.0, label='appc')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('appc')
    ax.set_title('Control Command (appc)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'simulink_pid_result.png')
    plt.savefig(save_path, dpi=150)
    print(f"曲线图已保存至: {save_path}")
    plt.show()


# ===================== 主程序 =====================

# --- PID 参数 ---
Kp, Ki, Kd = 5.0, 0.08, -2.0
NUM_EPISODES = 100

device = "cuda:0"
config = "simulink"
model_name = "Simulink"

env = ControlEnv(num_envs=1, config=config, model=model_name, random_seed=42, device=device)
dt = env.model.dt  # 0.005
max_steps = int(getattr(env.config, 'max_steps', 1000))

pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt)

print(f"PID控制器: Kp={Kp}, Ki={Ki}, Kd={Kd}")
print(f"环境: dt={dt}, max_steps={max_steps}")
print(f"评估episodes数: {NUM_EPISODES}")
print("=" * 60)

# --- 数据记录 ---
tet_buf = []
target_theta_buf = []
appc_buf = []

episode_metrics = []
start = time.time()

for ep in range(NUM_EPISODES):
    # Reset env (force reset by setting done flags)
    env.is_done[:] = True
    env.bad_done[:] = False
    env.exceed_time_limit[:] = False
    obs = env.reset()

    pid.reset()

    # Record initial state
    theta0 = float(_t2n(env.model.s[:, 3]).item())
    target = float(_t2n(env.task.target_theta).item())
    step_mag = abs(target - theta0)

    ep_tet = [theta0]
    ep_target = [target]
    ep_appc = [0.0]

    for step in range(max_steps):
        # Read current theta and target
        theta = float(_t2n(env.model.s[:, 3]).item())
        target = float(_t2n(env.task.target_theta).item())

        # PI-qD: error = target - actual, q = pitch rate
        error = target - theta
        q = float(_t2n(env.model.s[:, 2]).item())
        action_val = pid.compute(error, q)

        # PID outputs in [-0.35, 0.35]; convert to action domain [-1, 1] for env.step
        action = torch.tensor([[action_val / 0.35]], dtype=torch.float32, device=device)

        obs, rewards, dones, bad_dones, exceed_time_limits, infos = env.step(action)

        # Record
        theta_new = float(_t2n(env.model.s[:, 3]).item())
        control_new = float(_t2n(env.model.u[:, 0]).item())
        ep_tet.append(theta_new)
        ep_target.append(target)
        ep_appc.append(control_new)

        # Check for early termination (bad_done = diverged)
        if int(_t2n(bad_dones).item()):
            break

    # Compute metrics for this episode
    m = evaluate_step_response_metrics(
        theta_seg=np.array(ep_tet),
        target_seg=np.array(ep_target),
        dt=dt,
    )
    m['target'] = float(np.mean(ep_target))
    m['step_mag'] = step_mag
    m['length'] = len(ep_tet)
    m['terminated'] = int(_t2n(bad_dones).item()) if step < max_steps - 1 else 0
    episode_metrics.append(m)

    # Append to global buffers (for plotting first 10 episodes)
    if ep < 10:
        tet_buf.extend(ep_tet)
        target_theta_buf.extend(ep_target)
        appc_buf.extend(ep_appc)

    # Print progress
    os_str = f'{m["overshoot_percent"]:.2f}%' if not np.isnan(m['overshoot_percent']) else 'N/A'
    st_str = f'{m["settling_time_s"]:.4f}s' if not np.isnan(m['settling_time_s']) else 'N/A'
    print(f"[Ep {ep+1:>3}/{NUM_EPISODES}] target={m['target']:.4f} "
          f"step={step_mag:.4f} OS={os_str} settle={st_str} "
          f"SSE={m['steady_state_error']:.6f}")

end = time.time()

# ===================== 汇总输出 =====================
print('\n' + '=' * 85)
print(f'PID控制器: Kp={Kp}, Ki={Ki}, Kd={Kd}')
print(f'总耗时: {end - start:.2f}s')
print(f'总episodes: {NUM_EPISODES}')
terminated_count = sum(m['terminated'] for m in episode_metrics)
print(f'发散终止次数: {terminated_count}/{NUM_EPISODES}')
print()

# --- 每个 episode 的性能指标表 ---
print(f'{"Ep":>3} {"Target":>8} {"Step":>6} {"Len":>5} {"Overshoot%":>11} {"Rise(s)":>9} {"Settle(s)":>10} {"SSE(rad)":>12} {"Term":>5}')
print('-' * 85)
for i, m in enumerate(episode_metrics):
    tgt_str = f'{m["target"]:.4f}'
    stp_str = f'{m["step_mag"]:.4f}'
    os_str = f'{m["overshoot_percent"]:.4f}' if not np.isnan(m['overshoot_percent']) else 'N/A'
    rt_str = f'{m["rise_time_s"]:.4f}' if not np.isnan(m['rise_time_s']) else 'N/A'
    st_str = f'{m["settling_time_s"]:.4f}' if not np.isnan(m['settling_time_s']) else 'N/A'
    sse_str = f'{m["steady_state_error"]:.6f}'
    term_str = 'YES' if m['terminated'] else ''
    print(f'{i+1:>3} {tgt_str:>8} {stp_str:>6} {m["length"]:>5} {os_str:>11} {rt_str:>9} {st_str:>10} {sse_str:>12} {term_str:>5}')

# --- 平均指标 ---
avg_overshoot = np.nanmean([m['overshoot_percent'] for m in episode_metrics])
avg_rise_time = np.nanmean([m['rise_time_s'] for m in episode_metrics])
avg_settling_time = np.nanmean([m['settling_time_s'] for m in episode_metrics])
avg_ss_error = np.nanmean([m['steady_state_error'] for m in episode_metrics])

print('-' * 85)
print(f'{"AVG":>3} {"":>8} {"":>6} {"":>5} {avg_overshoot:>11.4f} {avg_rise_time:>9.4f} {avg_settling_time:>10.4f} {avg_ss_error:>12.6f}')
print('=' * 85)

# --- 保存数据 ---
result_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'result')
os.makedirs(result_dir, exist_ok=True)

np.save(os.path.join(result_dir, 'pid_avg_overshoot_percent.npy'), np.array([avg_overshoot]))
np.save(os.path.join(result_dir, 'pid_avg_rise_time_s.npy'), np.array([avg_rise_time]))
np.save(os.path.join(result_dir, 'pid_avg_settling_time_s.npy'), np.array([avg_settling_time]))
np.save(os.path.join(result_dir, 'pid_avg_steady_state_error.npy'), np.array([avg_ss_error]))
print(f"\n指标数据已保存至: {result_dir}")

# --- 绘制前10个episode的曲线 ---
if tet_buf:
    plot_pid_results(
        dt=dt,
        appc_buf=np.array(appc_buf),
        tet_buf=np.array(tet_buf),
        target_theta_buf=np.array(target_theta_buf),
        save_dir=result_dir,
    )
