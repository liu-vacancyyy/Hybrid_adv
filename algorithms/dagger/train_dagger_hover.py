"""
Train a DAgger policy on the hover task using the LearningToFly-style
PID cascade as the expert.

Usage:
    cd NeuralPlane_stable_V2
    python algorithms/dagger/train_dagger_hover.py \
        --num-envs 64 --iters 30 --rollout-steps 256

Outputs:
    algorithms/dagger/checkpoints/dagger_latest.pt
    algorithms/dagger/checkpoints/dagger_history.npy
"""
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'envs'))

from envs.control_env import ControlEnv                              # noqa: E402
from algorithms.pid.hover_pid import HoverPIDController              # noqa: E402
from algorithms.dagger.policy import PPOActorStudent, default_actor_args  # noqa: E402
from algorithms.dagger.dagger_trainer import DAggerTrainer           # noqa: E402

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs',          type=int,   default=128)
    p.add_argument('--iters',             type=int,   default=10000)
    p.add_argument('--rollout-steps',     type=int,   default=256)
    p.add_argument('--max-blocks',        type=int,   default=3,
                   help='How many recent rollout blocks to keep (DAgger aggregation).')
    p.add_argument('--mini-batches',      type=int,   default=16)
    p.add_argument('--train-epochs',      type=int,   default=4)
    p.add_argument('--data-chunk-length', type=int,   default=8,
                   help='BPTT chunk length used when --use-recurrent-policy.')
    p.add_argument('--max-chunks-per-minibatch', type=int, default=2048,
                   help='Hard cap on chunks per recurrent minibatch (BPTT memory bound).')
    p.add_argument('--lr',                type=float, default=3e-4)
    p.add_argument('--max-grad-norm',     type=float, default=1.0)
    p.add_argument('--beta0',             type=float, default=1.0)
    p.add_argument('--beta-decay',        type=float, default=0.94,
                   help='Multiplicative decay per iter. 0.94 -> beta~0.05 at iter 50 (for 100 iters).')
    p.add_argument('--beta-min',          type=float, default=0.0)
    # --- PPO-aligned student architecture (matches algorithms.ppo.ppo_actor.PPOActor) ---
    p.add_argument('--hidden-size',         type=str, default='128 128')
    p.add_argument('--act-hidden-size',     type=str, default='128 128')
    p.add_argument('--activation-id',       type=int, default=1,
                   help='0=Tanh, 1=ReLU, 2=LeakyReLU, 3=ELU')
    p.add_argument('--use-feature-normalization', action='store_true', default=True)
    p.add_argument('--no-feature-normalization',  dest='use_feature_normalization',
                   action='store_false')
    # GRU is enabled by default; pass --no-recurrent-policy to disable.
    p.add_argument('--use-recurrent-policy', action='store_true', default=True)
    p.add_argument('--no-recurrent-policy',  dest='use_recurrent_policy',
                   action='store_false')
    p.add_argument('--recurrent-hidden-size',   type=int, default=128)
    p.add_argument('--recurrent-hidden-layers', type=int, default=1)
    p.add_argument('--gain',                    type=float, default=0.01)
    p.add_argument('--seed',           type=int,   default=0)
    p.add_argument('--device',         type=str,   default='cuda:0')
    p.add_argument('--ckpt-dir',       type=str,
                   default=os.path.join(ROOT, 'algorithms', 'dagger', 'checkpoints'))
    p.add_argument('--log-dir',        type=str,
                   default=os.path.join(ROOT, 'algorithms', 'dagger', 'runs'),
                   help='TensorBoard log directory. Set to empty string to disable.')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Append a timestamped sub-folder so every run is isolated
    _run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.ckpt_dir = os.path.join(args.ckpt_dir, _run_tag)
    if args.log_dir:
        args.log_dir = os.path.join(args.log_dir, _run_tag)

    # --- env -------------------------------------------------------------
    env = ControlEnv(num_envs=args.num_envs,
                     config='hover',
                     model='HYBRID',
                     random_seed=args.seed,
                     device=device)
    obs_dim = env.num_observation
    act_dim = env.num_actions
    print(f"[env]    n={env.n}  obs_dim={obs_dim}  act_dim={act_dim}  dt={env.model.dt}")

    # --- expert ----------------------------------------------------------
    expert = HoverPIDController(n=env.n,
                                device=device,
                                dt=env.model.dt,
                                mass=1.779,
                                gravity=9.807,
                                max_thrust_per_motor=env.model.max_F)
    print(f"[expert] HoverPIDController  throttle_hover={expert.throttle_hover:.3f}")

    # --- student (PPO-compatible architecture) -----------------------
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
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[policy] PPOActorStudent  hidden='{args.hidden_size}'  "
          f"act_hidden='{args.act_hidden_size}'  recurrent={args.use_recurrent_policy}  "
          f"params={n_params}")

    # --- DAgger ----------------------------------------------------------
    trainer = DAggerTrainer(
        env=env, expert=expert, policy=policy,
        rollout_steps=args.rollout_steps,
        max_blocks=args.max_blocks,
        lr=args.lr,
        mini_batches=args.mini_batches,
        train_epochs=args.train_epochs,
        data_chunk_length=args.data_chunk_length,
        max_chunks_per_minibatch=args.max_chunks_per_minibatch,
        max_grad_norm=args.max_grad_norm,
        beta0=args.beta0, beta_decay=args.beta_decay, beta_min=args.beta_min,
        device=device,
        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir if args.log_dir else None,
    )

    history = trainer.fit(n_iters=args.iters)

    # --- save ------------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    np.save(os.path.join(args.ckpt_dir, 'dagger_history.npy'),
            np.array(history, dtype=object), allow_pickle=True)
    print(f"[done] checkpoints + history -> {args.ckpt_dir}")


if __name__ == '__main__':
    main()
