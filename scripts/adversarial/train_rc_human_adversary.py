#!/usr/bin/env python
"""Train a PPO adversary against a fixed rc_human policy.

The adversary perturbs command, observation, and wind spaces within the ranges
already used by the rc_human task/domain randomization.  The victim actor is
frozen; only the adversary is optimized.  This is intended as a feasibility
probe before embedding adversarial samples back into policy training.
"""
import argparse
import datetime
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import gym
import numpy as np
import torch
import torch.utils.tensorboard as tb

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from algorithms.adversarial.rc_human_adv_env import RCHumanAdversarialEnv  # noqa: E402
from algorithms.ppo.ppo_actor import PPOActor                              # noqa: E402
from algorithms.ppo.ppo_policy import PPOPolicy                            # noqa: E402
from algorithms.ppo.ppo_trainer import PPOTrainer                          # noqa: E402
from algorithms.utils.buffer import ReplayBuffer                           # noqa: E402
from envs.control_env import ControlEnv                                     # noqa: E402


DEFAULT_VICTIM = (
    ROOT / "scripts" / "runs"
    / "2026-05-19_22-58-25_Control_rc_human_HYBRID_NEW_ppo_rc_human_rl_gru_wind_first2"
    / "episode_740" / "actor_latest.ckpt"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--victim-ckpt", type=str, default=str(DEFAULT_VICTIM))
    p.add_argument("--scenario-name", type=str, default="rc_human")
    p.add_argument("--model-name", type=str, default="HYBRID_NEW")
    p.add_argument("--experiment-name", type=str, default="rc_human_adv_cmd_obs_wind")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cuda", action="store_true", default=True)
    p.add_argument("--n-rollout-threads", type=int, default=256)
    p.add_argument("--num-env-steps", type=float, default=2.0e7)
    p.add_argument("--max-iterations", type=int, default=1000,
                   help="Hard cap on PPO update iterations. Set <= 0 to disable.")
    p.add_argument("--buffer-size", type=int, default=256)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--run-dir", type=str, default="")

    # PPO for adversary.
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ppo-epoch", type=int, default=8)
    p.add_argument("--clip-param", type=float, default=0.2)
    p.add_argument("--num-mini-batch", type=int, default=8)
    p.add_argument("--value-loss-coef", type=float, default=1.0)
    p.add_argument("--entropy-coef", type=float, default=2e-3)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--hidden-size", type=str, default="128 128")
    p.add_argument("--act-hidden-size", type=str, default="128 128")
    p.add_argument("--activation-id", type=int, default=1)
    p.add_argument("--gain", type=float, default=0.01)
    p.add_argument("--no-feature-normalization",
                   dest="use_feature_normalization", action="store_false")
    p.set_defaults(use_feature_normalization=True)
    p.add_argument("--no-recurrent-policy",
                   dest="use_recurrent_policy", action="store_false")
    p.set_defaults(use_recurrent_policy=True)
    p.add_argument("--recurrent-hidden-size", type=int, default=128)
    p.add_argument("--recurrent-hidden-layers", type=int, default=1)
    p.add_argument("--data-chunk-length", type=int, default=8)

    # Victim architecture.
    p.add_argument("--victim-hidden-size", type=str, default="128 128")
    p.add_argument("--victim-act-hidden-size", type=str, default="128 128")
    p.add_argument("--victim-recurrent-hidden-size", type=int, default=128)
    p.add_argument("--victim-recurrent-hidden-layers", type=int, default=1)
    p.add_argument("--victim-deterministic", action="store_true", default=True)

    # Attack bounds, expressed as fractions of existing task randomization/noise ranges.
    # Defaults are intentionally mild; increase these only after the adversary
    # learns non-saturating risk cases.
    p.add_argument("--adv-command-frac", type=float, default=0.18)
    p.add_argument("--adv-obs-frac", type=float, default=0.6)
    p.add_argument("--adv-wind-frac", type=float, default=0.5)
    p.add_argument("--adv-use-random-command", action="store_true", default=False,
                   help="If set, do not attack command space; use rc_human's original command generator.")
    p.add_argument("--adv-command-random-base", action="store_true", default=False,
                   help="When command attack is enabled, add command perturbations on top of "
                        "rc_human's randomized command generator instead of replacing it.")
    p.add_argument("--adv-obs-default-scale", type=float, default=0.02)
    p.add_argument("--adv-obs-max-scale", type=float, default=0.10)
    p.add_argument("--adv-command-alpha", type=float, default=0.20)
    p.add_argument("--adv-obs-alpha", type=float, default=0.25)
    p.add_argument("--adv-wind-alpha", type=float, default=0.15)
    p.add_argument("--adv-init-log-std", type=float, default=-1.2,
                   help="Initial adversary Gaussian log std. -1.2 gives std ~= 0.30.")

    # Adversary objective matching the requested formula.
    p.add_argument("--adv-alive-penalty", type=float, default=0.01)
    p.add_argument("--adv-policy-reward-weight", type=float, default=0.05)
    p.add_argument("--adv-w-vel-error", type=float, default=2.0)
    p.add_argument("--adv-w-yaw-error", type=float, default=1.0)
    p.add_argument("--adv-w-attitude", type=float, default=5.0)
    p.add_argument("--adv-w-omega", type=float, default=0.8)
    p.add_argument("--adv-w-force-margin", type=float, default=0.2)
    p.add_argument("--adv-bad-done-bonus", type=float, default=50.0)
    p.add_argument("--adv-linf-penalty", type=float, default=0.20)
    p.add_argument("--adv-smooth-penalty", type=float, default=0.10)
    p.add_argument("--adv-raw-excess-penalty", type=float, default=0.20)
    p.add_argument("--adv-obs-energy-window", type=int, default=50)
    p.add_argument("--adv-obs-energy-budget", type=float, default=50.0)
    p.add_argument("--adv-obs-energy-penalty", type=float, default=0.05)
    p.add_argument("--adv-attitude-safe-rad", type=float, default=0.20)
    p.add_argument("--adv-omega-safe-rad", type=float, default=1.2)
    return p.parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t2n(x):
    return x.detach().cpu().numpy()


def make_policy_args(args, action_dim=None, is_victim=False):
    if is_victim:
        hidden = args.victim_hidden_size
        act_hidden = args.victim_act_hidden_size
        rnn_h = args.victim_recurrent_hidden_size
        rnn_l = args.victim_recurrent_hidden_layers
    else:
        hidden = args.hidden_size
        act_hidden = args.act_hidden_size
        rnn_h = args.recurrent_hidden_size
        rnn_l = args.recurrent_hidden_layers

    ns = SimpleNamespace()
    ns.gain = args.gain
    ns.hidden_size = hidden
    ns.act_hidden_size = act_hidden
    ns.activation_id = args.activation_id
    ns.use_feature_normalization = args.use_feature_normalization
    ns.use_recurrent_policy = args.use_recurrent_policy
    ns.recurrent_hidden_size = rnn_h
    ns.recurrent_hidden_layers = rnn_l
    ns.use_prior = False
    ns.lr = args.lr
    ns.ppo_epoch = args.ppo_epoch
    ns.clip_param = args.clip_param
    ns.use_clipped_value_loss = False
    ns.num_mini_batch = args.num_mini_batch
    ns.value_loss_coef = args.value_loss_coef
    ns.entropy_coef = args.entropy_coef
    ns.use_max_grad_norm = True
    ns.max_grad_norm = args.max_grad_norm
    ns.data_chunk_length = args.data_chunk_length
    ns.buffer_size = args.buffer_size
    ns.n_rollout_threads = args.n_rollout_threads
    ns.gamma = args.gamma
    ns.use_proper_time_limits = False
    ns.use_gae = True
    ns.gae_lambda = args.gae_lambda
    ns.num_actions = action_dim
    return ns


def load_victim(env, args, device):
    actor_args = make_policy_args(args, is_victim=True)
    actor = PPOActor(actor_args, env.observation_space, env.action_space, device)
    state = torch.load(args.victim_ckpt, map_location=device)
    if isinstance(state, dict):
        state = state.get("policy", state.get("state_dict", state))
    actor.load_state_dict(state)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    return actor


def set_initial_action_std(policy, log_std):
    """Start adversary with moderate exploration instead of unit-std noise."""
    modules = [policy.actor.act]
    for module in modules:
        action_out = getattr(module, "action_out", None)
        if action_out is not None and hasattr(action_out, "log_std"):
            with torch.no_grad():
                action_out.log_std.fill_(float(log_std))


def save_adversary(policy, run_dir, episode):
    save_dir = run_dir / f"episode_{episode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.actor.state_dict(), save_dir / "adv_actor_latest.ckpt")
    torch.save(policy.critic.state_dict(), save_dir / "adv_critic_latest.ckpt")
    torch.save(policy.actor.state_dict(), run_dir / "adv_actor_latest.ckpt")
    torch.save(policy.critic.state_dict(), run_dir / "adv_critic_latest.ckpt")


def collect_rollout(env, policy, buffer, args, device):
    policy.prep_rollout()
    n = env.n
    obs = buffer.obs[0].reshape(n, -1)
    rnn_actor = buffer.rnn_states_actor[0].reshape(
        n, args.recurrent_hidden_layers, args.recurrent_hidden_size
    )
    rnn_critic = buffer.rnn_states_critic[0].reshape(
        n, args.recurrent_hidden_layers, args.recurrent_hidden_size
    )
    masks = buffer.masks[0].reshape(n, 1)

    metrics = []
    for step in range(args.buffer_size):
        values, actions, logp, rnn_actor_next, rnn_critic_next = policy.get_actions(
            obs, rnn_actor, rnn_critic, masks
        )
        next_obs, rewards, done, bad_done, exceed, _info = env.step(actions)
        reset = (done | bad_done | exceed).reshape(-1)

        masks_next = torch.ones((n, 1), device=device)
        masks_next[reset] = 0.0
        bad_masks_next = torch.ones((n, 1), device=device)
        bad_masks_next[bad_done.reshape(-1)] = 0.0
        rnn_actor_next[reset] = 0.0
        rnn_critic_next[reset] = 0.0

        buffer.insert(
            t2n(next_obs).reshape(args.n_rollout_threads, 1, -1),
            t2n(actions).reshape(args.n_rollout_threads, 1, -1),
            t2n(rewards).reshape(args.n_rollout_threads, 1, 1),
            t2n(masks_next).reshape(args.n_rollout_threads, 1, 1),
            t2n(logp).reshape(args.n_rollout_threads, 1, 1),
            t2n(values).reshape(args.n_rollout_threads, 1, 1),
            t2n(rnn_actor_next).reshape(args.n_rollout_threads, 1,
                                        args.recurrent_hidden_layers,
                                        args.recurrent_hidden_size),
            t2n(rnn_critic_next).reshape(args.n_rollout_threads, 1,
                                         args.recurrent_hidden_layers,
                                         args.recurrent_hidden_size),
            t2n(bad_masks_next).reshape(args.n_rollout_threads, 1, 1),
        )

        metrics.append(dict(env.last_info))
        obs = next_obs
        rnn_actor = rnn_actor_next
        rnn_critic = rnn_critic_next
        masks = masks_next

    with torch.no_grad():
        next_value = policy.get_values(obs, rnn_critic, masks)
    buffer.compute_returns(t2n(next_value).reshape(args.n_rollout_threads, 1, 1))
    return metrics


def mean_metrics(metrics):
    out = {}
    if not metrics:
        return out
    keys = sorted({k for m in metrics for k in m})
    for key in keys:
        vals = [m[key] for m in metrics if key in m]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def main():
    args = parse_args()
    seed_everything(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    torch.set_num_threads(1)

    run_dir = Path(args.run_dir) if args.run_dir else (
        ROOT / "scripts" / "runs"
        / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
          f"Control_{args.scenario_name}_{args.model_name}_ppo_{args.experiment_name}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = tb.SummaryWriter(run_dir)

    base_env = ControlEnv(
        num_envs=args.n_rollout_threads,
        config=args.scenario_name,
        model=args.model_name,
        random_seed=args.seed,
        device=device,
    )
    victim = load_victim(base_env, args, device)
    env = RCHumanAdversarialEnv(base_env, victim, args, device)

    adv_args = make_policy_args(args, action_dim=env.adv_action_dim)
    adv_policy = PPOPolicy(adv_args, env.observation_space, env.action_space, device)
    set_initial_action_std(adv_policy, args.adv_init_log_std)
    trainer = PPOTrainer(adv_args, device)
    buffer = ReplayBuffer(adv_args, env.num_agents, env.observation_space, env.action_space)
    buffer.obs[0] = t2n(env.reset()).reshape(args.n_rollout_threads, 1, -1)

    requested_episodes = int(
        float(args.num_env_steps) // args.buffer_size // args.n_rollout_threads
    )
    if requested_episodes <= 0:
        raise ValueError("num-env-steps is too small for buffer-size * n-rollout-threads")
    episodes = requested_episodes
    if args.max_iterations > 0:
        episodes = min(episodes, int(args.max_iterations))
    logging.info("run_dir=%s", run_dir)
    logging.info("victim=%s", args.victim_ckpt)
    logging.info(
        "adv_action_dim=%d iterations=%d requested_iterations=%d max_iterations=%d",
        env.adv_action_dim,
        episodes,
        requested_episodes,
        args.max_iterations,
    )

    total_steps = 0
    for episode in range(episodes):
        metrics = collect_rollout(env, adv_policy, buffer, args, device)
        adv_policy.prep_training()
        train_info = trainer.train(adv_policy, buffer)
        buffer.after_update()
        total_steps += args.buffer_size * args.n_rollout_threads

        if episode % args.log_interval == 0:
            scalars = mean_metrics(metrics)
            scalars.update({f"train/{k}": v for k, v in train_info.items()})
            reward_mean = float(np.mean(buffer.rewards))
            rollout_return = float(np.sum(buffer.rewards) / max(args.n_rollout_threads, 1))
            scalars["adv/mean_reward"] = reward_mean
            scalars["adv/rollout_return"] = rollout_return
            scalars["reward/adv_mean_step_reward"] = reward_mean
            scalars["reward/adv_rollout_return_per_env"] = rollout_return
            if "adv/policy_reward" in scalars:
                scalars["reward/victim_mean_step_reward"] = scalars["adv/policy_reward"]
            for k, v in scalars.items():
                writer.add_scalar(k, v, total_steps)
            logging.info(
                "iteration=%d/%d steps=%d adv_rew=%.3f return=%.3f bad_done=%.3f vel=%.3f yaw=%.3f linf=%.3f",
                episode,
                episodes,
                total_steps,
                scalars.get("adv/mean_reward", 0.0),
                scalars.get("adv/rollout_return", 0.0),
                scalars.get("adv/bad_done_rate", 0.0),
                scalars.get("adv/vel_error", 0.0),
                scalars.get("adv/yaw_error", 0.0),
                scalars.get("adv/linf", 0.0),
            )

        if episode % args.save_interval == 0 or episode == episodes - 1:
            save_adversary(adv_policy, run_dir, episode)

    env.close()
    writer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
