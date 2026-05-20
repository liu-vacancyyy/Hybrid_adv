"""
PPO vs PID comparison for Simulink angle tracking — NEW task (θ₀=0, target∈[-2,+2]).

Three figures displayed one by one:
  Fig 1: SSE vs target value (N_TOTAL episodes, STEPS_PER_EP steps each)
  Fig 2: θ tracking curves  (first N_PLOT episodes)
  Fig 3: Control output      (first N_PLOT episodes)
"""
import os
import sys
import random as rnd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.control_env import ControlEnv
from algorithms.ppo.ppo_actor import PPOActor

# ===================== Helpers =====================

def _t2n(x):
    return x.detach().cpu().numpy()

def reseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rnd.seed(seed)

class Args:
    def __init__(self):
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


class PIDController:
    """PI-qD controller: D term uses measured pitch rate q."""

    def __init__(self, Kp, Ki, Kd, dt, output_limit=0.35,
                 integral_limit=0.08, back_calc_coeff=0.15):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.back_calc_coeff = back_calc_coeff
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def compute(self, error, q):
        P = self.Kp * error
        I = self.Ki * self.integral
        D = self.Kd * q
        output_unsat = P + I + D
        output_sat = np.clip(output_unsat, -self.output_limit, self.output_limit)
        sat_error = output_sat - output_unsat
        self.integral += (error + self.back_calc_coeff * sat_error) * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        return output_sat


def compute_sse(theta_arr, target):
    """Steady-state error: mean of last 10% of trajectory vs target."""
    tail_len = max(5, int(0.1 * len(theta_arr)))
    return abs(float(np.mean(theta_arr[-tail_len:])) - target)


def override_target(env, target):
    """Override env target after reset, keeping initial state from env.reset()."""
    initial_theta = env.model.s[:, 3].clone()
    env.task.target_theta[:] = target
    env.task.initial_theta[:] = initial_theta
    env.task.step_magnitude[:] = torch.clamp(
        torch.abs(env.task.target_theta - env.task.initial_theta), min=0.1)
    env.task.last_delta_theta[:] = initial_theta - target
    env.task.integral_error[:] = 0.0
    env.task.settle_count[:] = 0.0
    obs = env.obs()
    return obs


# ===================== Configuration =====================

SEED = 42
N_TOTAL = 50         # total episodes for SSE comparison
N_PLOT = 10          # first N episodes for tracking / control plots
STEPS_PER_EP = 5000  # steps per episode (25s at dt=0.005)

device = "cuda:0"
config = "simulink"
model_name = "Simulink"

# PID parameters (PI-qD: D = Kd * q)
Kp, Ki, Kd = 5.0, 0.08, -2.0

# PPO model path — update to latest checkpoint
PPO_RUN_DIR = "/home/a/demo/NeuralPlane_stable_V2/scripts/runs/2026-04-17_10-16-11_Control_simulink_Simulink_ppo_v1"
PPO_EPISODE_DIR = os.path.join(PPO_RUN_DIR, "episode_25")

result_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'result')
os.makedirs(result_dir, exist_ok=True)

# Pre-generate targets so PPO and PID see the exact same targets
# Uniform in [-2, +2], same as env task config
target_rng = np.random.RandomState(SEED)
def _gen_target_no_deadzone(rng, low=-2.0, high=2.0, deadzone=0.03):
    while True:
        t = rng.random() * (high - low) + low
        if abs(t) >= deadzone:
            return t

episode_targets = [_gen_target_no_deadzone(target_rng) for _ in range(N_TOTAL)]

# ===================== Load PPO model =====================
env_ppo = ControlEnv(num_envs=1, config=config, model=model_name,
                     random_seed=SEED, device=device)
dt = env_ppo.model.dt

args = Args()
ego_policy = PPOActor(args, env_ppo.observation_space, env_ppo.action_space,
                      device=torch.device(device))
ego_policy.eval()
ego_policy.load_state_dict(
    torch.load(PPO_EPISODE_DIR + "/actor_latest.ckpt", map_location=device))

# ===================== Run PPO =====================
print("=" * 60)
print(f"Running PPO controller: {N_TOTAL} episodes × {STEPS_PER_EP} steps")
print("=" * 60)

reseed(SEED)
ppo_data = []

for ep in range(N_TOTAL):
    env_ppo.is_done[:] = True
    env_ppo.bad_done[:] = False
    env_ppo.exceed_time_limit[:] = False
    obs = env_ppo.reset()

    # Override target to pre-generated value
    target = episode_targets[ep]
    obs = override_target(env_ppo, target)

    rnn = torch.zeros(1, 1, args.recurrent_hidden_size, device=torch.device(device))
    masks = torch.zeros(1, 1, device=torch.device(device))

    ep_theta = [float(_t2n(env_ppo.model.s[:, 3]).item())]
    ep_appc = [float(_t2n(env_ppo.model.u[:, 0]).item())]
    bad = False

    for step in range(STEPS_PER_EP):
        with torch.no_grad():
            actions, _, rnn = ego_policy(obs, rnn, masks, deterministic=True)

        obs, rew, done, bad_done, exceed, info = env_ppo.step(actions)
        masks = torch.ones(1, 1, device=torch.device(device))

        ep_theta.append(float(_t2n(env_ppo.model.s[:, 3]).item()))
        ep_appc.append(float(_t2n(env_ppo.model.u[:, 0]).item()))

        if int(_t2n(bad_done).item()):
            bad = True
            break

        env_ppo.is_done[:] = False
        env_ppo.bad_done[:] = False
        env_ppo.exceed_time_limit[:] = False

    sse = compute_sse(ep_theta, target)
    ppo_data.append(dict(theta=np.array(ep_theta), appc=np.array(ep_appc),
                         target=target, sse=sse, bad=bad))
    tag = " [BAD]" if bad else ""
    print(f"  PPO Ep {ep+1:>3}/{N_TOTAL}: target={target:+.4f} SSE={sse:.6f}{tag}")

del env_ppo

# ===================== Run PID =====================
print("\n" + "=" * 60)
print(f"Running PID controller (Kp={Kp}, Ki={Ki}, Kd={Kd}): "
      f"{N_TOTAL} episodes × {STEPS_PER_EP} steps")
print("=" * 60)

env_pid = ControlEnv(num_envs=1, config=config, model=model_name,
                     random_seed=SEED, device=device)

pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt)
pid_data = []

for ep in range(N_TOTAL):
    env_pid.is_done[:] = True
    env_pid.bad_done[:] = False
    env_pid.exceed_time_limit[:] = False
    obs = env_pid.reset()
    pid.reset()

    target = episode_targets[ep]
    obs = override_target(env_pid, target)

    ep_theta = [float(_t2n(env_pid.model.s[:, 3]).item())]
    ep_appc = [float(_t2n(env_pid.model.u[:, 0]).item())]
    bad = False

    for step in range(STEPS_PER_EP):
        theta = float(_t2n(env_pid.model.s[:, 3]).item())
        q = float(_t2n(env_pid.model.s[:, 2]).item())
        error = target - theta
        action_val = pid.compute(error, q)

        action = torch.tensor([[action_val / 0.35]], dtype=torch.float32, device=device)
        obs, rew, done, bad_done, exceed, info = env_pid.step(action)

        ep_theta.append(float(_t2n(env_pid.model.s[:, 3]).item()))
        ep_appc.append(float(_t2n(env_pid.model.u[:, 0]).item()))

        if int(_t2n(bad_done).item()):
            bad = True
            break

        env_pid.is_done[:] = False
        env_pid.bad_done[:] = False
        env_pid.exceed_time_limit[:] = False

    sse = compute_sse(ep_theta, target)
    pid_data.append(dict(theta=np.array(ep_theta), appc=np.array(ep_appc),
                         target=target, sse=sse, bad=bad))
    tag = " [BAD]" if bad else ""
    print(f"  PID Ep {ep+1:>3}/{N_TOTAL}: target={target:+.4f} SSE={sse:.6f}{tag}")

del env_pid

# ===================== Summary =====================
ppo_mean_sse = np.mean([d['sse'] for d in ppo_data])
pid_mean_sse = np.mean([d['sse'] for d in pid_data])
ppo_bad = sum(d['bad'] for d in ppo_data)
pid_bad = sum(d['bad'] for d in pid_data)
print(f"\nPPO: mean SSE = {ppo_mean_sse:.6f}, diverged = {ppo_bad}/{N_TOTAL}")
print(f"PID: mean SSE = {pid_mean_sse:.6f}, diverged = {pid_bad}/{N_TOTAL}")

# ===================== Figure 1: SSE vs Target =====================
print("\n绘制 Figure 1: 稳态误差 vs 目标值 ...")

targets = np.array([d['target'] for d in ppo_data])
ppo_sse = np.array([d['sse'] for d in ppo_data])
pid_sse = np.array([d['sse'] for d in pid_data])
sort_idx = np.argsort(targets)

fig1, ax1 = plt.subplots(figsize=(14, 5))

ax1.scatter(targets[sort_idx], ppo_sse[sort_idx],
            color='tab:red', s=20, zorder=3, alpha=0.5)
ax1.scatter(targets[sort_idx], pid_sse[sort_idx],
            color='tab:blue', s=20, marker='s', zorder=3, alpha=0.5)

x_sorted = targets[sort_idx]
x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
try:
    spl_ppo = make_interp_spline(x_sorted, ppo_sse[sort_idx], k=3)
    spl_pid = make_interp_spline(x_sorted, pid_sse[sort_idx], k=3)
    ax1.plot(x_smooth, spl_ppo(x_smooth), color='tab:red', linewidth=1.5, label='PPO')
    ax1.plot(x_smooth, spl_pid(x_smooth), color='tab:blue', linewidth=1.5, label='PID')
except Exception:
    ax1.plot(x_sorted, ppo_sse[sort_idx], color='tab:red', linewidth=1.0, label='PPO')
    ax1.plot(x_sorted, pid_sse[sort_idx], color='tab:blue', linewidth=1.0, label='PID')

ax1.set_xlabel('Target θ (rad)', fontsize=12)
ax1.set_ylabel('Steady-State Error (rad)', fontsize=12)
ax1.set_title(f'PPO vs PID — Steady-State Error (new task: θ₀=0, target∈[-2,+2])',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(os.path.join(result_dir, 'compare_v2_sse.png'), dpi=150)
print(f"  已保存: {result_dir}/compare_v2_sse.png")
plt.show()

# ===================== Figure 2: θ Tracking (N_PLOT episodes) =====================
print(f"\n绘制 Figure 2: θ 跟踪曲线 (前{N_PLOT}个episode) ...")

fig2, axes2 = plt.subplots(2, 5, figsize=(22, 8), sharex=True)
fig2.suptitle(f'PPO vs PID — θ Tracking (new task, {N_PLOT} episodes)',
              fontsize=15, fontweight='bold')

for i in range(N_PLOT):
    ax = axes2[i // 5, i % 5]
    t_ppo = np.arange(len(ppo_data[i]['theta'])) * dt
    t_pid = np.arange(len(pid_data[i]['theta'])) * dt
    tgt = ppo_data[i]['target']

    ax.axhline(y=tgt, color='gray', linestyle='--', linewidth=0.8, label='target')
    ax.plot(t_ppo, ppo_data[i]['theta'], color='tab:red', linewidth=1.0, label='PPO')
    ax.plot(t_pid, pid_data[i]['theta'], color='tab:blue', linewidth=1.0, label='PID')
    ax.set_title(f'Ep {i+1}  tgt={tgt:+.3f}', fontsize=10)
    ax.set_ylabel('θ (rad)', fontsize=9)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc='best')
    if i >= 5:
        ax.set_xlabel('Time (s)', fontsize=9)

plt.tight_layout()
fig2.savefig(os.path.join(result_dir, 'compare_v2_tracking.png'), dpi=150)
print(f"  已保存: {result_dir}/compare_v2_tracking.png")
plt.show()

# ===================== Figure 3: Control Output (N_PLOT episodes) =====================
print(f"\n绘制 Figure 3: 控制输出曲线 (前{N_PLOT}个episode) ...")

fig3, axes3 = plt.subplots(2, 5, figsize=(22, 8), sharex=True)
fig3.suptitle(f'PPO vs PID — Control Output (new task, {N_PLOT} episodes)',
              fontsize=15, fontweight='bold')

for i in range(N_PLOT):
    ax = axes3[i // 5, i % 5]
    t_ppo = np.arange(len(ppo_data[i]['appc'])) * dt
    t_pid = np.arange(len(pid_data[i]['appc'])) * dt

    ax.plot(t_ppo[1:], ppo_data[i]['appc'][1:], color='tab:red', linewidth=0.8, label='PPO', alpha=0.85)
    ax.plot(t_pid[1:], pid_data[i]['appc'][1:], color='tab:blue', linewidth=0.8, label='PID', alpha=0.85)
    ax.set_title(f'Ep {i+1}  tgt={ppo_data[i]["target"]:+.3f}', fontsize=10)
    ax.set_ylabel('appc', fontsize=9)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc='best')
    if i >= 5:
        ax.set_xlabel('Time (s)', fontsize=9)

plt.tight_layout()
fig3.savefig(os.path.join(result_dir, 'compare_v2_control.png'), dpi=150)
print(f"  已保存: {result_dir}/compare_v2_control.png")
plt.show()

print("\n全部完成！")
