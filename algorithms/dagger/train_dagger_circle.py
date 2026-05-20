"""
Train a DAgger policy on the circle task.

The student policy runs on the noisy/randomized circle environment. The expert
is a clean PID controller: it reads ground-truth model state directly and
generates labels without using the noisy observation.

Usage:
    cd NeuralPlane_stable_V2
    python algorithms/dagger/train_dagger_circle.py \
        --num-envs 128 --iters 100 --rollout-steps 256 --device cuda:0
"""
import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'envs'))

from algorithms.dagger.dagger_trainer import DAggerTrainer          # noqa: E402
from algorithms.dagger.policy import PPOActorStudent, default_actor_args  # noqa: E402
from algorithms.pid.circle_pid import CirclePIDController           # noqa: E402
from envs.control_env import ControlEnv                             # noqa: E402


class CirclePIDExpert:
    """Circle PID expert with optional head-motor tangent-speed feedforward."""

    def __init__(self, env, use_head=True, head_max_force=0.8):
        self.env = env
        self.use_head = bool(use_head)
        self.head_max_force = float(head_max_force)
        self.pid = CirclePIDController(
            n=env.n,
            device=env.device,
            dt=env.model.dt,
            mass=1.779,
            gravity=9.807,
            max_thrust_per_motor=env.model.max_F,
            use_head_motor=use_head,
            head_max_force=head_max_force,
        )

    def reset(self, mask=None):
        self.pid.reset(mask=mask)

    def set_targets(self, target_altitude, target_heading,
                    target_npos=None, target_epos=None):
        task = self.env.task
        if hasattr(self.pid, 'set_circle_targets'):
            self.pid.set_circle_targets(task)
        else:
            self.pid.set_targets(
                target_altitude=target_altitude,
                target_heading=target_heading,
                target_npos=target_npos,
                target_epos=target_epos,
            )

    def compute_action(self, model):
        return self.pid.compute_action(model)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs', type=int, default=128)
    p.add_argument('--iters', type=int, default=10000)
    p.add_argument('--rollout-steps', type=int, default=1000)
    p.add_argument('--max-blocks', type=int, default=10)
    p.add_argument('--mini-batches', type=int, default=16)
    p.add_argument('--train-epochs', type=int, default=4)
    p.add_argument('--data-chunk-length', type=int, default=8)
    p.add_argument('--max-chunks-per-minibatch', type=int, default=2048)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--beta0', type=float, default=1.0)
    p.add_argument('--beta-decay', type=float, default=0.99)
    p.add_argument('--beta-min', type=float, default=0.01)

    p.add_argument('--circle-radius', type=float, default=10.0)
    p.add_argument('--circle-period', type=float, default=-40.0,
                   help='Seconds per circle. Negative keeps initial tangent aligned with initial heading.')
    p.add_argument('--circle-offset-left', type=float, default=10.0)
    p.add_argument('--student-noise-scale', type=float, default=0.02,
                   help='Gaussian observation noise seen by the student.')
    p.add_argument('--init-yaw-range', type=float, default=0.25)
    p.add_argument('--init-attitude-range', type=float, default=0.05)
    p.add_argument('--init-vel-range', type=float, default=0.2)
    p.add_argument('--init-omega-range', type=float, default=0.02)
    p.add_argument('--no-head-expert', dest='with_head_expert',
                   action='store_false',
                   help='Disable head motor feedforward in the PID expert.')
    p.set_defaults(with_head_expert=True)
    p.add_argument('--head-max-force', type=float, default=0.8)

    p.add_argument('--hidden-size', type=str, default='128 128')
    p.add_argument('--act-hidden-size', type=str, default='128 128')
    p.add_argument('--activation-id', type=int, default=1)
    p.add_argument('--use-feature-normalization', action='store_true', default=True)
    p.add_argument('--no-feature-normalization',
                   dest='use_feature_normalization', action='store_false')
    p.add_argument('--use-recurrent-policy', action='store_true', default=True)
    p.add_argument('--no-recurrent-policy',
                   dest='use_recurrent_policy', action='store_false')
    p.add_argument('--recurrent-hidden-size', type=int, default=128)
    p.add_argument('--recurrent-hidden-layers', type=int, default=1)
    p.add_argument('--gain', type=float, default=0.01)

    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--verbose-env', action='store_true',
                   help='do not suppress verbose prints from env.step')
    p.add_argument('--reset-rollout-each-iter',
                   dest='continue_rollout',
                   action='store_false',
                   help='Reset policy/PID rollout state at every DAgger iteration.')
    p.set_defaults(continue_rollout=True)
    p.add_argument('--ckpt-dir', type=str,
                   default=os.path.join(ROOT, 'algorithms', 'dagger', 'checkpoints_circle'))
    p.add_argument('--log-dir', type=str,
                   default=os.path.join(ROOT, 'algorithms', 'dagger', 'runs_circle'))
    return p.parse_args()


def configure_circle_env(env, args):
    """Apply circle DAgger overrides after ControlEnv builds the task."""
    env.config.noise_scale = float(args.student_noise_scale)
    env.config.circle_radius = float(args.circle_radius)
    env.config.circle_period = float(args.circle_period)
    env.config.circle_offset_left = float(args.circle_offset_left)
    env.config.init_yaw_range = float(args.init_yaw_range)
    env.config.init_roll_range = float(args.init_attitude_range)
    env.config.init_pitch_range = float(args.init_attitude_range)
    env.config.init_vel_range = float(args.init_vel_range)
    env.config.init_omega_range = float(args.init_omega_range)
    env.model.init_yaw_range = float(args.init_yaw_range)
    env.model.init_roll_range = float(args.init_attitude_range)
    env.model.init_pitch_range = float(args.init_attitude_range)
    env.model.init_vel_range = float(args.init_vel_range)
    env.model.init_omega_range = float(args.init_omega_range)

    env.task.noise_scale = float(args.student_noise_scale)
    env.task.radius = float(args.circle_radius)
    env.task.offset_left = float(args.circle_offset_left)
    env.task.omega = 0.0 if args.circle_period == 0.0 else 2.0 * math.pi / float(args.circle_period)


def main():
    args = parse_args()
    if not args.use_recurrent_policy:
        raise SystemExit('Circle DAgger requires GRU; do not pass --no-recurrent-policy.')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.ckpt_dir = os.path.join(args.ckpt_dir, run_tag)
    if args.log_dir:
        args.log_dir = os.path.join(args.log_dir, run_tag)

    env = ControlEnv(num_envs=args.num_envs,
                     config='circle',
                     model='HYBRID',
                     random_seed=args.seed,
                     device=device)
    configure_circle_env(env, args)
    print(
        f"[env] n={env.n} obs_dim={env.num_observation} act_dim={env.num_actions} "
        f"dt={env.model.dt} noise={env.task.noise_scale} "
        f"period={args.circle_period}s radius={args.circle_radius}m "
        f"continue_rollout={args.continue_rollout}"
    )

    expert = CirclePIDExpert(env,
                             use_head=args.with_head_expert,
                             head_max_force=args.head_max_force)
    print(f"[expert] CirclePIDExpert head_motor={args.with_head_expert}")

    actor_args = default_actor_args(
        hidden_size=args.hidden_size,
        act_hidden_size=args.act_hidden_size,
        activation_id=args.activation_id,
        use_feature_normalization=args.use_feature_normalization,
        use_recurrent_policy=args.use_recurrent_policy,
        recurrent_hidden_size=args.recurrent_hidden_size,
        recurrent_hidden_layers=args.recurrent_hidden_layers,
        gain=args.gain,
        use_prior=False,
    )
    policy = PPOActorStudent(env.observation_space, env.action_space,
                             args=actor_args, device=device)
    n_params = sum(param.numel() for param in policy.parameters())
    print(
        f"[policy] PPOActorStudent hidden='{args.hidden_size}' "
        f"recurrent={args.use_recurrent_policy} params={n_params}"
    )

    trainer = DAggerTrainer(
        env=env,
        expert=expert,
        policy=policy,
        rollout_steps=args.rollout_steps,
        max_blocks=args.max_blocks,
        lr=args.lr,
        mini_batches=args.mini_batches,
        train_epochs=args.train_epochs,
        data_chunk_length=args.data_chunk_length,
        max_chunks_per_minibatch=args.max_chunks_per_minibatch,
        max_grad_norm=args.max_grad_norm,
        beta0=args.beta0,
        beta_decay=args.beta_decay,
        beta_min=args.beta_min,
        device=device,
        suppress_env_stdout=(not args.verbose_env),
        continue_across_iters=args.continue_rollout,
        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir if args.log_dir else None,
    )

    history = trainer.fit(n_iters=args.iters)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    np.save(os.path.join(args.ckpt_dir, 'dagger_history.npy'),
            np.array(history, dtype=object), allow_pickle=True)
    print(f"[done] checkpoints + history -> {args.ckpt_dir}")


if __name__ == '__main__':
    main()
