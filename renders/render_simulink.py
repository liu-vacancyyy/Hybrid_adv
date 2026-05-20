import os
import sys
import glob
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.control_env import ControlEnv
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.DEBUG)

CURRENT_WORK_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '64 64'
        self.act_hidden_size = '64 64'
        self.activation_id = 1
        self.use_feature_normalization = True
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 64
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cuda:0'))
        self.use_prior = False


def _t2n(x):
    return x.detach().cpu().numpy()


def find_latest_simulink_run(runs_dir):
    """自动查找最新的simulink训练目录"""
    pattern = os.path.join(runs_dir, "*simulink*Simulink*ppo*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"在 {runs_dir} 中未找到simulink的训练目录")
    return dirs[-1]


def plot_simulink_results(dt, appc_buf, tet_buf, target_theta_buf, u_buf, w_buf, q_buf, save_dir='./result'):
    """绘制simulink评估结果曲线"""
    os.makedirs(save_dir, exist_ok=True)
    steps = np.arange(len(tet_buf))
    time_axis = steps * dt  # 转换为秒

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Simulink PPO Evaluation Results', fontsize=16, fontweight='bold')

    # ---- 1. theta (tet) vs target_theta ----
    ax = axes[0, 0]
    ax.plot(time_axis, tet_buf, color='r', linewidth=1.2, label='actual θ (tet)')
    ax.plot(time_axis, target_theta_buf, color='b', linewidth=1.2, linestyle='--', label='target θ')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('θ (rad)')
    ax.set_title('Pitch Angle θ Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 2. tracking error ----
    ax = axes[0, 1]
    error = tet_buf - target_theta_buf
    ax.plot(time_axis, error, color='purple', linewidth=1.0, label='θ error')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (rad)')
    ax.set_title('Tracking Error (θ - target θ)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 3. appc (control command) ----
    ax = axes[1, 0]
    ax.plot(time_axis, appc_buf, color='g', linewidth=1.0, label='appc')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('appc')
    ax.set_title('Control Command (appc)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 4. longitudinal velocity u ----
    ax = axes[1, 1]
    ax.plot(time_axis, u_buf, color='orange', linewidth=1.0, label='u')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('u (m/s)')
    ax.set_title('Longitudinal Velocity (u)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 5. vertical velocity w ----
    ax = axes[2, 0]
    ax.plot(time_axis, w_buf, color='brown', linewidth=1.0, label='w')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('w (m/s)')
    ax.set_title('Vertical Velocity (w)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 6. pitch rate q ----
    ax = axes[2, 1]
    ax.plot(time_axis, q_buf, color='teal', linewidth=1.0, label='q')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('q (rad/s)')
    ax.set_title('Pitch Rate (q)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'simulink_result.png')
    plt.savefig(save_path, dpi=150)
    print(f"曲线图已保存至: {save_path}")
    plt.show()


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

    # 稳态误差：最后10%样本(至少5个点)的均值误差
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

    # 调节时间(2%误差带，且之后始终在带内)
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


# ===================== 主程序 =====================

# --- 自动查找最新simulink运行目录的episode_80 ---
runs_dir = os.path.join(CURRENT_WORK_PATH, "scripts", "runs")
latest_run = find_latest_simulink_run(runs_dir)
latest_run = "/home/a/demo/NeuralPlane_stable_V2/scripts/runs/2026-04-17_12-13-24_Control_simulink_Simulink_ppo_v1"
ego_run_dir = os.path.join(latest_run, "episode_83")
print(f"加载模型: {ego_run_dir}")
assert os.path.exists(os.path.join(ego_run_dir, "actor_latest.ckpt")), \
    f"未找到checkpoint: {ego_run_dir}/actor_latest.ckpt"

device = "cuda:0"
config = "simulink"
model_name = "Simulink"

env = ControlEnv(num_envs=1, config=config, model=model_name, random_seed=5, device=device)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device(device))
ego_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + "/actor_latest.ckpt", map_location=device))

print("Start render")
ego_obs = env.reset()

# --- 初始状态记录 ---
# 状态量: [u, w, q, theta]
state = env.model.get_state()
u_buf = np.array([np.mean(_t2n(state[:, 0]))])
w_buf = np.array([np.mean(_t2n(state[:, 1]))])
q_buf = np.array([np.mean(_t2n(state[:, 2]))])
tet_buf = np.array([np.mean(_t2n(state[:, 3]))])  # theta (tet)

# 控制量: appc
control = env.model.get_control()
appc_buf = np.array([np.mean(_t2n(control[:, 0]))])

# 目标角度
target_theta_buf = np.array([np.mean(_t2n(env.task.target_theta))])

# 奖励
reward_buf = np.array([0.0])

counts = 0
episode_rewards = 0
ego_rnn_states = torch.zeros((1, 1, args.recurrent_hidden_size), device=torch.device(device))
masks = torch.ones((1, 1), device=torch.device(device))
start = time.time()
unreach_target = 0
reset_target = 0
episode_end_indices = []
episode_metrics = []

while True:
    with torch.no_grad():
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)

    ego_obs, rewards, dones, bad_dones, exceed_time_limits, infos = env.step(ego_actions, render=True, count=counts)

    unreach_target += int(_t2n(bad_dones))
    reset_target += int(_t2n(dones))
    reset_flag = int(_t2n(dones)) or int(_t2n(bad_dones)) or int(_t2n(exceed_time_limits))

    # --- 记录状态量 ---
    state = env.model.get_state()
    u_buf = np.hstack((u_buf, np.mean(_t2n(state[:, 0]))))
    w_buf = np.hstack((w_buf, np.mean(_t2n(state[:, 1]))))
    q_buf = np.hstack((q_buf, np.mean(_t2n(state[:, 2]))))
    tet_buf = np.hstack((tet_buf, np.mean(_t2n(state[:, 3]))))

    # --- 记录控制量 appc ---
    control = env.model.get_control()
    appc_buf = np.hstack((appc_buf, np.mean(_t2n(control[:, 0]))))

    # --- 记录目标角度 ---
    target_theta_buf = np.hstack((target_theta_buf, np.mean(_t2n(env.task.target_theta))))

    # --- 记录奖励 ---
    reward_buf = np.hstack((reward_buf, np.mean(_t2n(rewards))))

    counts += 1
    episode_rewards += _t2n(rewards)
    print(f"step: {counts}, reward: {_t2n(rewards).item():.4f}, "
          f"tet: {np.mean(_t2n(state[:, 3])):.4f}, "
          f"target: {np.mean(_t2n(env.task.target_theta)):.4f}, "
          f"appc: {np.mean(_t2n(control[:, 0])):.4f}")

    if reset_flag:
        episode_end_indices.append(len(tet_buf) - 1)
        # --- 计算该episode的控制性能指标 ---
        ep_start = episode_end_indices[-2] + 1 if len(episode_end_indices) >= 2 else 0
        ep_end = episode_end_indices[-1]
        if ep_end - ep_start + 1 >= 3:
            m = evaluate_step_response_metrics(
                theta_seg=tet_buf[ep_start:ep_end + 1],
                target_seg=target_theta_buf[ep_start:ep_end + 1],
                dt=env.model.dt,
            )
            ep_type = 'done' if int(_t2n(dones)) else ('bad_done' if int(_t2n(bad_dones)) else 'timeout')
            m['type'] = ep_type
            m['start'] = ep_start
            m['end'] = ep_end
            m['target'] = float(np.mean(target_theta_buf[ep_start:ep_end + 1]))
            episode_metrics.append(m)

    if counts >= 10000:
        break

end = time.time()
print('=' * 60)
print(f'总耗时: {end - start:.2f}s')
print(f'总步数: {counts}')
print(f'累计奖励: {episode_rewards.item():.4f}')
print(f'成功到达目标次数: {reset_target}')
print(f'未到达目标次数: {unreach_target}')
if (reset_target + unreach_target) > 0:
    print(f'成功率: {reset_target / (reset_target + unreach_target):.4f}')
    print(f'平均episode奖励: {episode_rewards.item() / (unreach_target + reset_target):.4f}')

# --- 评估指标汇总：平均超调量、上升时间、调节时间、稳态误差 ---
# 处理最后一段（如果没有以reset结尾）
if len(episode_end_indices) == 0 or episode_end_indices[-1] != len(tet_buf) - 1:
    last_start = (episode_end_indices[-1] + 1) if episode_end_indices else 0
    last_end = len(tet_buf) - 1
    if last_end - last_start + 1 >= 3:
        m = evaluate_step_response_metrics(
            theta_seg=tet_buf[last_start:last_end + 1],
            target_seg=target_theta_buf[last_start:last_end + 1],
            dt=env.model.dt,
        )
        m['type'] = '未完成'
        m['start'] = last_start
        m['end'] = last_end
        m['target'] = float(np.mean(target_theta_buf[last_start:last_end + 1]))
        episode_metrics.append(m)

# --- 输出每个 episode 的性能指标 ---
if episode_metrics:
    print('\n' + '=' * 70)
    print(f'{"Ep":>3} {"Type":>10} {"Steps":>12} {"Len":>5} {"Target":>8} {"Overshoot%":>11} {"Rise(s)":>9} {"Settle(s)":>10} {"SSE(rad)":>12}')
    print('-' * 78)
    for i, m in enumerate(episode_metrics):
        ep_len = m['end'] - m['start'] + 1
        tgt_str = f'{m["target"]:.4f}'
        os_str = f'{m["overshoot_percent"]:.4f}'
        rt_str = f'{m["rise_time_s"]:.4f}' if not np.isnan(m['rise_time_s']) else 'N/A'
        st_str = f'{m["settling_time_s"]:.4f}' if not np.isnan(m['settling_time_s']) else 'N/A'
        sse_str = f'{m["steady_state_error"]:.6f}'
        print(f'{i+1:>3} {m["type"]:>10} {m["start"]:>5}-{m["end"]:<5} {ep_len:>5} {tgt_str:>8} {os_str:>11} {rt_str:>9} {st_str:>10} {sse_str:>12}')

    avg_overshoot = np.nanmean([m['overshoot_percent'] for m in episode_metrics])
    avg_rise_time = np.nanmean([m['rise_time_s'] for m in episode_metrics])
    avg_settling_time = np.nanmean([m['settling_time_s'] for m in episode_metrics])
    avg_ss_error = np.nanmean([m['steady_state_error'] for m in episode_metrics])

    print('-' * 78)
    print(f'{"AVG":>3} {"":>10} {"":>12} {"":>5} {"":>8} {avg_overshoot:>11.4f} {avg_rise_time:>9.4f} {avg_settling_time:>10.4f} {avg_ss_error:>12.6f}')
    print('=' * 78)

# --- 保存数据 ---
result_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'result')
os.makedirs(result_dir, exist_ok=True)

np.save(os.path.join(result_dir, 'simulink_appc.npy'), appc_buf)
np.save(os.path.join(result_dir, 'simulink_tet.npy'), tet_buf)
np.save(os.path.join(result_dir, 'simulink_target_theta.npy'), target_theta_buf)
np.save(os.path.join(result_dir, 'simulink_u.npy'), u_buf)
np.save(os.path.join(result_dir, 'simulink_w.npy'), w_buf)
np.save(os.path.join(result_dir, 'simulink_q.npy'), q_buf)
np.save(os.path.join(result_dir, 'simulink_reward.npy'), reward_buf)
np.save(os.path.join(result_dir, 'simulink_avg_overshoot_percent.npy'), np.array([avg_overshoot]))
np.save(os.path.join(result_dir, 'simulink_avg_rise_time_s.npy'), np.array([avg_rise_time]))
np.save(os.path.join(result_dir, 'simulink_avg_settling_time_s.npy'), np.array([avg_settling_time]))
np.save(os.path.join(result_dir, 'simulink_avg_steady_state_error.npy'), np.array([avg_ss_error]))
print(f"数据已保存至: {result_dir}")

# --- 绘制曲线 ---
plot_simulink_results(
    dt=env.model.dt,
    appc_buf=appc_buf,
    tet_buf=tet_buf,
    target_theta_buf=target_theta_buf,
    u_buf=u_buf,
    w_buf=w_buf,
    q_buf=q_buf,
    save_dir=result_dir,
)
