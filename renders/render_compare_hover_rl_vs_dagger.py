"""Compare hover performance: PPO(RL) policy vs DAgger student policy.

This script evaluates two *network* policies on the same Hybrid hover task:
1) PPO actor checkpoint trained by pure RL (scripts/train_hover.sh)
2) DAgger student checkpoint (algorithms/dagger)

For each rollout seed, the two policies are run in separate env instances with
identical scenario/model/seed settings, then key trajectories and metrics are
compared.

Usage:
    python renders/render_compare_hover_rl_vs_dagger.py \
        --rl-ckpt scripts/runs/2026-05-08_21-08-37_Control_hover_HYBRID_ppo_v1/episode_559/actor_latest.ckpt \
        --dagger-ckpt algorithms/dagger/checkpoints/20260429_230311/dagger_iter1789.pt

For the PID baseline comparison entrypoint, run:
    python renders/render_compare_hover_rl_vs_dagger_vs_pid.py
"""
import os
import sys
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib
os.environ.setdefault('DISPLAY', ':1')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

from envs.control_env import ControlEnv                                  # noqa: E402
from envs.utils.utils import _t2n                                        # noqa: E402
from algorithms.ppo.ppo_actor import PPOActor                            # noqa: E402
from algorithms.dagger.policy import PPOActorStudent, default_actor_args # noqa: E402
from algorithms.pid.hover_pid import HoverPIDController                  # noqa: E402


@dataclass
class PPOActorArgs:
    gain: float = 0.01
    hidden_size: str = '128 128'
    act_hidden_size: str = '128 128'
    activation_id: int = 1
    use_feature_normalization: bool = True
    use_recurrent_policy: bool = True
    recurrent_hidden_size: int = 128
    recurrent_hidden_layers: int = 1
    use_prior: bool = False


class RLPolicyRunner:
    """Inference wrapper for PPO actor checkpoint."""

    def __init__(self, actor, recurrent_layers, recurrent_hidden, device):
        self.actor = actor
        self.recurrent_layers = recurrent_layers
        self.recurrent_hidden = recurrent_hidden
        self.device = device
        self.rnn_states = None
        self.masks = None

    def reset(self, n_envs):
        self.rnn_states = torch.zeros(
            (n_envs, self.recurrent_layers, self.recurrent_hidden),
            dtype=torch.float32,
            device=self.device,
        )
        self.masks = torch.ones((n_envs, 1), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs):
        actions, _, self.rnn_states = self.actor(
            obs, self.rnn_states, self.masks, deterministic=True)
        return actions

    def on_step_done(self, reset_mask):
        self.masks = (~reset_mask).float().unsqueeze(-1)


class DaggerPolicyRunner:
    """Inference wrapper for DAgger student checkpoint."""

    def __init__(self, actor):
        self.actor = actor

    def reset(self, n_envs):
        self.actor.reset_rollout_state(n_envs)

    @torch.no_grad()
    def act(self, obs):
        return self.actor.act(obs)

    def on_step_done(self, reset_mask):
        self.actor.set_done_mask(reset_mask)


class NoisyModelView:
    """Inject the same hover sensor noise for PID feedback."""

    def __init__(self, model, task, dt=0.02):
        self.model = model
        self.pos_std = float(task.sensor_pos_std)
        self.vel_std = float(task.sensor_vel_std)
        self.att_std = float(task.sensor_att_std)
        self.omega_std = float(task.sensor_omega_std)
        self.enabled = bool(task.enable_sensor_noise)
        self.dt = float(dt)

    @staticmethod
    def _wrap_pi(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def _noise(self, x, std):
        if not self.enabled or std == 0.0:
            return x
        return x + torch.randn_like(x) * std

    def get_position(self):
        n, e, a = self.model.get_position()
        return self._noise(n, self.pos_std), self._noise(e, self.pos_std), self._noise(a, self.pos_std)

    def get_posture(self):
        r, p, y = self.model.get_posture()
        if self.enabled:
            r = self._wrap_pi(r + torch.randn_like(r) * self.att_std)
            p = self._wrap_pi(p + torch.randn_like(p) * self.att_std)
            y = self._wrap_pi(y + torch.randn_like(y) * self.att_std)
        return r, p, y

    def get_ground_speed(self):
        vx, vy = self.model.get_ground_speed()
        return self._noise(vx, self.vel_std), self._noise(vy, self.vel_std)

    def get_climb_rate(self):
        return self._noise(self.model.get_climb_rate(), self.vel_std)

    def get_angular_velocity(self):
        P, Q, R = self.model.get_angular_velocity()
        return self._noise(P, self.omega_std), self._noise(Q, self.omega_std), self._noise(R, self.omega_std)


class PIDPolicyRunner:
    """Inference wrapper for cascade PID hover controller."""

    def __init__(self, device, use_noisy=True, mass=1.779, gravity=9.807):
        self.device = device
        self.use_noisy = use_noisy
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.pid = None
        self.env = None
        self.view = None

    def bind_env(self, env):
        self.env = env
        self.view = NoisyModelView(env.model, env.task, dt=env.model.dt) if self.use_noisy else env.model
        self.pid = HoverPIDController(
            n=env.n,
            device=self.device,
            dt=env.model.dt,
            mass=self.mass,
            gravity=self.gravity,
            max_thrust_per_motor=env.model.max_F,
        )

    def reset(self, n_envs):
        _ = n_envs
        self.pid.reset()
        self.pid.set_targets(
            target_altitude=self.env.task.target_altitude.clone(),
            target_heading=self.env.task.target_heading.clone(),
            target_npos=self.env.task.target_npos.clone(),
            target_epos=self.env.task.target_epos.clone(),
        )

    def act(self, obs):
        _ = obs
        return self.pid.compute_action(self.view)

    def on_step_done(self, reset_mask):
        if torch.any(reset_mask):
            self.pid.reset(mask=reset_mask)
            self.pid.set_targets(
                target_altitude=self.env.task.target_altitude.clone(),
                target_heading=self.env.task.target_heading.clone(),
                target_npos=self.env.task.target_npos.clone(),
                target_epos=self.env.task.target_epos.clone(),
            )


def _new_buf():
    return dict(
        t=[], npos=[], epos=[], alt=[],
        roll=[], pitch=[], yaw=[],
        vx=[], vy=[], vz=[],
        reward=[],
        target_npos=[], target_epos=[], target_alt=[], target_heading=[],
        act=[],   # final applied motor forces from env.model.u, shape: (T, action_dim)
    )


def _push(buf, env):
    npos, epos, alt = env.model.get_position()
    roll, pitch, yaw = env.model.get_posture()
    vx, vy = env.model.get_ground_speed()
    vz = env.model.get_climb_rate()

    buf['npos'].append(_t2n(npos).mean())
    buf['epos'].append(_t2n(epos).mean())
    buf['alt'].append(_t2n(alt).mean())
    buf['roll'].append(_t2n(roll).mean())
    buf['pitch'].append(_t2n(pitch).mean())
    buf['yaw'].append(_t2n(yaw).mean())
    buf['vx'].append(_t2n(vx).mean())
    buf['vy'].append(_t2n(vy).mean())
    buf['vz'].append(_t2n(vz).mean())

    buf['target_npos'].append(_t2n(env.task.target_npos).mean())
    buf['target_epos'].append(_t2n(env.task.target_epos).mean())
    buf['target_alt'].append(_t2n(env.task.target_altitude).mean())
    buf['target_heading'].append(_t2n(env.task.target_heading).mean())


def _position_err_3d(buf):
    n = np.asarray(buf['npos'])
    e = np.asarray(buf['epos'])
    a = np.asarray(buf['alt'])
    tn = np.asarray(buf['target_npos'])
    te = np.asarray(buf['target_epos'])
    ta = np.asarray(buf['target_alt'])
    return np.sqrt((n - tn) ** 2 + (e - te) ** 2 + (a - ta) ** 2)


def _finalize_hover_action(action):
    """Match hover training: force head motor to zero thrust before env.step.

    HybridModel maps normalized action -1 -> 0 N, 0 -> 3.5 N, 1 -> 7 N.
    """
    final_action = action.clone()
    if final_action.shape[-1] > 0:
        final_action[:, 0] = -1.0
    return final_action


def run_rollout(policy_runner, args, device, seed, label):
    env = ControlEnv(num_envs=1, config='hover', model='HYBRID',
                     random_seed=seed, device=device)
    if hasattr(policy_runner, 'bind_env'):
        policy_runner.bind_env(env)
    obs = env.reset()
    policy_runner.reset(env.n)

    buf = _new_buf()
    _push(buf, env)
    buf['t'].append(0.0)
    buf['reward'].append(0.0)

    bad_done_total = 0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        action = _finalize_hover_action(policy_runner.act(obs))
        obs, reward, done, bad_done, exceed, _ = env.step(action)
        reset_mask = done | bad_done | exceed
        policy_runner.on_step_done(reset_mask)

        buf['t'].append(step * env.model.dt)
        buf['reward'].append(_t2n(reward).mean())
        # Record the final motor forces actually applied by HybridModel.update,
        # after normalized-action mapping, clamping, and motor rate limiting.
        buf['act'].append(_t2n(env.model.get_control()).squeeze(0).tolist())
        _push(buf, env)
        bad_done_total += int(_t2n(bad_done).sum())

    total_reward = float(np.sum(buf['reward']))
    err = _position_err_3d(buf)
    rms_err = float(np.sqrt(np.mean(err ** 2)))
    mean_step_reward = float(np.mean(buf['reward']))

    print(f"[{label}] seed={seed}  sim={args.steps*env.model.dt:.1f}s  "
          f"wall={time.time()-t0:.2f}s  total_reward={total_reward:.1f}  "
          f"rms_3d_err={rms_err:.3f}  bad_done={bad_done_total}")

    metrics = {
        'total_reward': total_reward,
        'mean_step_reward': mean_step_reward,
        'rms_3d_err': rms_err,
        'bad_done': float(bad_done_total),
    }
    return buf, metrics


def _build_rl_runner(env, args, device):
    ppo_args = PPOActorArgs(
        gain=args.rl_gain,
        hidden_size=args.rl_hidden_size,
        act_hidden_size=args.rl_act_hidden_size,
        activation_id=args.rl_activation_id,
        use_feature_normalization=not args.rl_disable_feature_norm,
        use_recurrent_policy=not args.rl_disable_recurrent,
        recurrent_hidden_size=args.rl_recurrent_hidden_size,
        recurrent_hidden_layers=args.rl_recurrent_hidden_layers,
        use_prior=False,
    )

    actor = PPOActor(ppo_args, env.observation_space, env.action_space, device=device)
    state = torch.load(args.rl_ckpt, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    actor.load_state_dict(state)
    actor.eval()

    print(f"[rl] loaded: {args.rl_ckpt}")
    print("[rl] net: hidden='{}' act_hidden='{}' act={} recur={} "
          "rnn={}x{}".format(
              ppo_args.hidden_size,
              ppo_args.act_hidden_size,
              ppo_args.activation_id,
              ppo_args.use_recurrent_policy,
              ppo_args.recurrent_hidden_size,
              ppo_args.recurrent_hidden_layers,
          ))

    return RLPolicyRunner(
        actor=actor,
        recurrent_layers=ppo_args.recurrent_hidden_layers,
        recurrent_hidden=ppo_args.recurrent_hidden_size,
        device=device,
    )


def _build_dagger_runner(env, args, device):
    ckpt = torch.load(args.dagger_ckpt, map_location=device)
    actor_args = default_actor_args(**ckpt.get('args', {}))
    actor = PPOActorStudent(env.observation_space, env.action_space,
                            args=actor_args, device=device)
    actor.load_state_dict(ckpt['policy'])
    actor.eval()

    print(f"[dagger] loaded: {args.dagger_ckpt}")
    print("[dagger] net: hidden='{}' act_hidden='{}' act={} recur={} "
          "rnn={}x{}".format(
              actor_args.hidden_size,
              actor_args.act_hidden_size,
              actor_args.activation_id,
              actor_args.use_recurrent_policy,
              actor_args.recurrent_hidden_size,
              actor_args.recurrent_hidden_layers,
          ))

    return DaggerPolicyRunner(actor)


def plot_compare(buf_rl, buf_dagger, out_png, buf_pid=None):
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    series = [
        ('RL', buf_rl, 'tab:green'),
        ('DAgger', buf_dagger, 'tab:blue'),
    ]
    if buf_pid is not None:
        series.append(('PID', buf_pid, 'tab:orange'))
    c_t = 'k'

    ax = axes[0, 0]
    for name, buf, color in series:
        ax.plot(np.asarray(buf['t']), np.asarray(buf['alt']), color, label=name)
    ax.plot(np.asarray(buf_dagger['t']), np.asarray(buf_dagger['target_alt']), c_t, ls='--', lw=1,
            label='target')
    ax.set_title('Altitude (m)')
    ax.legend()
    ax.grid()

    ax = axes[0, 1]
    for name, buf, color in series:
        t = np.asarray(buf['t'])
        ax.plot(t, np.asarray(buf['npos']), color, label=f'{name} npos')
        ax.plot(t, np.asarray(buf['epos']), color, ls=':', label=f'{name} epos')
    ax.plot(np.asarray(buf_dagger['t']), np.asarray(buf_dagger['target_npos']), c_t, ls='--', lw=1)
    ax.plot(np.asarray(buf_dagger['t']), np.asarray(buf_dagger['target_epos']), c_t, ls='--', lw=1)
    ax.set_title('Horizontal N/E (m)')
    ax.legend(fontsize=8)
    ax.grid()

    ax = axes[0, 2]
    for name, buf, color in series:
        ax.plot(np.asarray(buf['t']), np.degrees(np.asarray(buf['yaw'])), color, label=name)
    ax.plot(np.asarray(buf_dagger['t']), np.degrees(np.asarray(buf_dagger['target_heading'])), c_t, ls='--', lw=1,
            label='target')
    ax.set_title('Yaw (deg)')
    ax.legend()
    ax.grid()

    ax = axes[1, 0]
    for name, buf, color in series:
        t = np.asarray(buf['t'])
        ax.plot(t, np.asarray(buf['vx']), color, label=f'{name} vx')
        ax.plot(t, np.asarray(buf['vy']), color, ls=':', label=f'{name} vy')
    ax.set_title('Ground speed vx/vy (m/s)')
    ax.legend(fontsize=8)
    ax.grid()

    ax = axes[1, 1]
    for name, buf, color in series:
        ax.plot(np.asarray(buf['t']), np.asarray(buf['vz']), color, label=name)
    ax.set_title('Climb rate vz (m/s)')
    ax.legend()
    ax.grid()

    ax = axes[1, 2]
    for name, buf, color in series:
        ax.plot(np.asarray(buf['t']), np.asarray(buf['reward']), color, label=name)
    ax.set_title('Per-step reward')
    ax.legend()
    ax.grid()

    ax = axes[2, 0]
    for name, buf, color in series:
        err = _position_err_3d(buf)
        ax.plot(np.asarray(buf['t']), err, color,
                label=f"{name} (RMS={np.sqrt(np.mean(err**2)):.2f})")
    ax.set_title('3-D position error (m)')
    ax.legend()
    ax.grid()

    ax = axes[2, 1]
    for name, buf, color in series:
        ax.plot(np.asarray(buf['t']), np.cumsum(np.asarray(buf['reward'])), color, label=name)
    ax.set_title('Cumulative reward')
    ax.legend()
    ax.grid()

    ax = axes[2, 2]
    for name, buf, color in series:
        t = np.asarray(buf['t'])
        ax.plot(t, np.degrees(np.asarray(buf['roll'])), color, label=f'{name} roll')
        ax.plot(t, np.degrees(np.asarray(buf['pitch'])), color, ls=':', label=f'{name} pitch')
    ax.set_title('Roll/Pitch (deg)')
    ax.legend(fontsize=8)
    ax.grid()

    # ---- Row 3: final applied motor force curves ------------------------
    # act arrays store env.model.u after HybridModel.update.
    ax = axes[3, 0]
    for name, buf, color in series:
        act = np.asarray(buf['act']) if buf['act'] else np.zeros((0, 5))
        ta = np.asarray(buf['t'])[1:len(act) + 1]
        if len(act):
            ax.plot(ta, act[:, 0], color, label=name)
    ax.set_title('Final F_head (N)')
    ax.legend(fontsize=8); ax.grid()

    ax = axes[3, 1]
    for name, buf, color in series:
        act = np.asarray(buf['act']) if buf['act'] else np.zeros((0, 5))
        ta = np.asarray(buf['t'])[1:len(act) + 1]
        if len(act) and act.shape[1] > 1:
            ax.plot(ta, act[:, 1:].mean(axis=1), color, label=name)
    ax.set_title('Final F_1-4 mean (N)')
    ax.legend(fontsize=8); ax.grid()

    ax = axes[3, 2]
    for name, buf, color in series:
        act = np.asarray(buf['act']) if buf['act'] else np.zeros((0, 5))
        ta = np.asarray(buf['t'])[1:len(act) + 1]
        if len(act) and act.shape[1] > 1:
            spread = act[:, 1:].max(axis=1) - act[:, 1:].min(axis=1)
            ax.plot(ta, spread, color, label=name)
    ax.set_title('Final F_1-4 spread (N)')
    ax.legend(fontsize=8); ax.grid()

    title = 'Hover task (Hybrid) - PPO(RL) vs DAgger'
    if buf_pid is not None:
        title += ' vs PID'
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"[plot] saved: {out_png}")


def _summarize(name, rows):
    def _ms(k):
        v = np.asarray([r[k] for r in rows], dtype=np.float64)
        return float(v.mean()), float(v.std())

    tr_m, tr_s = _ms('total_reward')
    sr_m, sr_s = _ms('mean_step_reward')
    er_m, er_s = _ms('rms_3d_err')
    bd_m, bd_s = _ms('bad_done')
    print(f"[{name}] total_reward={tr_m:.1f}±{tr_s:.1f}  "
          f"mean_step_reward={sr_m:.3f}±{sr_s:.3f}  "
          f"rms_3d_err={er_m:.3f}±{er_s:.3f}  "
          f"bad_done={bd_m:.2f}±{bd_s:.2f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rl-ckpt', type=str,
                   default=os.path.join(ROOT, 'scripts', 'runs',
                                        '2026-05-11_17-58-18_Control_hover_HYBRID_ppo_v1',
                                        'episode_220', 'actor_latest.ckpt'))
    p.add_argument('--dagger-ckpt', type=str,
                   default=os.path.join(ROOT, 'algorithms', 'dagger', 'checkpoints',
                                        '20260429_230311', 'dagger_iter1789.pt'))
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--num-rollouts', type=int, default=4,
                   help='number of sequential seeds to aggregate: seed+i')
    p.add_argument('--no-show', action='store_true')
    p.add_argument('--with-pid', action='store_true',
                   help='also run and plot the PID baseline')
    p.add_argument('--pid-clean', action='store_true',
                   help='when --with-pid is set, PID reads clean model states instead of noisy measurements')

    # RL architecture (must match training if you changed defaults)
    p.add_argument('--rl-hidden-size', type=str, default='128 128')
    p.add_argument('--rl-act-hidden-size', type=str, default='128 128')
    p.add_argument('--rl-activation-id', type=int, default=1)
    p.add_argument('--rl-gain', type=float, default=0.01)
    p.add_argument('--rl-disable-feature-norm', action='store_true')
    p.add_argument('--rl-disable-recurrent', action='store_true')
    p.add_argument('--rl-recurrent-hidden-size', type=int, default=128)
    p.add_argument('--rl-recurrent-hidden-layers', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build a probe env once for network construction.
    probe_env = ControlEnv(num_envs=1, config='hover', model='HYBRID',
                           random_seed=args.seed, device=device)
    rl_runner = _build_rl_runner(probe_env, args, device)
    dagger_runner = _build_dagger_runner(probe_env, args, device)
    pid_runner = (PIDPolicyRunner(device=device, use_noisy=(not args.pid_clean))
                  if args.with_pid else None)

    rl_metrics_all = []
    dg_metrics_all = []
    pid_metrics_all = [] if args.with_pid else None
    first_rl_buf = None
    first_dg_buf = None
    first_pid_buf = None

    print('=' * 70)
    title = 'Hover performance comparison: PPO(RL) vs DAgger'
    if args.with_pid:
        title += ' vs PID'
    print(title)
    print('=' * 70)

    for i in range(args.num_rollouts):
        seed_i = args.seed + i
        print('-' * 70)
        print(f'rollout seed = {seed_i}')

        buf_rl, met_rl = run_rollout(rl_runner, args, device, seed_i, label='RL')
        buf_dg, met_dg = run_rollout(dagger_runner, args, device, seed_i, label='DAgger')
        if args.with_pid:
            buf_pid, met_pid = run_rollout(pid_runner, args, device, seed_i,
                                           label='PID-clean' if args.pid_clean else 'PID')

        rl_metrics_all.append(met_rl)
        dg_metrics_all.append(met_dg)
        if args.with_pid:
            pid_metrics_all.append(met_pid)

        if i == 0:
            first_rl_buf = buf_rl
            first_dg_buf = buf_dg
            if args.with_pid:
                first_pid_buf = buf_pid

    print('=' * 70)
    print('Aggregate over seeds')
    _summarize('RL', rl_metrics_all)
    _summarize('DAgger', dg_metrics_all)
    if args.with_pid:
        _summarize('PID-clean' if args.pid_clean else 'PID', pid_metrics_all)

    result_dir = os.path.join(ROOT, 'renders', 'result')
    os.makedirs(result_dir, exist_ok=True)
    out_name = 'hover_compare_rl_vs_dagger_vs_pid.png' if args.with_pid else 'hover_compare_rl_vs_dagger.png'
    out_png = os.path.join(result_dir, out_name)
    plot_compare(first_rl_buf, first_dg_buf, out_png, first_pid_buf)

    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
