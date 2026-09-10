import os
import sys
import time
import csv
import math
import torch
import logging
import numpy as np
from typing import List
from pathlib import Path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from base_runner import Runner, ReplayBuffer
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
from algorithms.ppo.ppo_policy import PPOPolicy as Policy
from basicutils.torch_utils import *

def _t2n(x):
    return x.detach().cpu().numpy()

class F16SimRunner(Runner):

    TERMINATION_INFO_KEYS = (
        'ground_crash_hard_touchdown',
        'ground_crash_penetration',
        'ground_crash_tipover',
        'ground_crash_force',
        'extreme_angle',
        'extreme_omega',
        'extreme_aero_state',
        'high_speed',
        'overload',
        'mission_off_route',
        'mission_altitude_violation',
        'mission_premature_contact',
        'mission_nonfinite',
    )

    def load(self):
        if self.all_args.algorithm_name == 'cpo':
            self.all_args.use_cost_constraints = True
        self.obs_space = self.envs.observation_space
        self.use_privileged_critic = bool(
            getattr(self.envs, 'use_privileged_critic', False)
        )
        self.all_args.use_privileged_critic = self.use_privileged_critic
        self.critic_obs_space = (
            self.envs.critic_observation_space
            if self.use_privileged_critic else None
        )
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.agents

        # policy & algorithm
        self.policy = Policy(
            self.all_args,
            self.obs_space,
            self.act_space,
            device=self.device,
            critic_obs_space=self.critic_obs_space,
        )
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(
            self.all_args,
            self.num_agents,
            self.obs_space,
            self.act_space,
            critic_obs_space=self.critic_obs_space,
        )

        if self.model_dir is not None:
            self.restore()
        if getattr(self.all_args, 'init_actor_ckpt', None):
            self._init_actor_from_checkpoint(self.all_args.init_actor_ckpt)
        self._best_average_reward = -float("inf")
        self._best_tracking_error = float("inf")
        self._best_bad_done_fraction = float("inf")
        self._progress_csv_header_written = False
        # Episode returns are accumulated per vectorized environment.  They
        # cannot be reconstructed from the rollout buffer after an update,
        # because a rollout usually contains many unfinished episodes.
        self._running_episode_returns = np.zeros(
            (self.n_rollout_threads, self.num_agents), dtype=np.float64
        )
        self._rollout_completed_returns = []
        self._rollout_step_reward_sum = 0.0
        self._rollout_transition_count = 0
        self._rollout_done_count = 0
        self._rollout_bad_done_count = 0
        self._rollout_timeout_count = 0
        self._rollout_termination_reasons = {
            key: 0 for key in self.TERMINATION_INFO_KEYS
        }
        self._rollout_failure_phases = {}

    def _begin_rollout_metrics(self):
        self._rollout_completed_returns = []
        self._rollout_step_reward_sum = 0.0
        self._rollout_transition_count = 0
        self._rollout_done_count = 0
        self._rollout_bad_done_count = 0
        self._rollout_timeout_count = 0
        self._rollout_termination_reasons = {
            key: 0 for key in self.TERMINATION_INFO_KEYS
        }
        self._rollout_failure_phases = {}
        env = getattr(self.envs, 'gpu_vec_env', None)
        task = getattr(env, 'task', None)
        if task is not None and hasattr(task, 'begin_training_rollout'):
            task.begin_training_rollout()

    def _record_rollout_transition(self, rewards, dones, bad_dones,
                                   exceed_time_limits, infos=None):
        """Accumulate episode statistics before the environment auto-resets."""
        rewards = np.asarray(rewards, dtype=np.float64).reshape(
            self.n_rollout_threads, self.num_agents, -1
        )[..., 0]
        done_flags = np.asarray(dones, dtype=bool).reshape(
            self.n_rollout_threads, self.num_agents, -1
        )[..., 0]
        bad_flags = np.asarray(bad_dones, dtype=bool).reshape(
            self.n_rollout_threads, self.num_agents, -1
        )[..., 0]
        timeout_flags = np.asarray(exceed_time_limits, dtype=bool).reshape(
            self.n_rollout_threads, self.num_agents, -1
        )[..., 0]

        self._running_episode_returns += rewards
        self._rollout_step_reward_sum += float(rewards.sum())
        self._rollout_transition_count += rewards.size
        reset_flags = done_flags | bad_flags | timeout_flags
        # A safety violation takes precedence over success or timeout if more
        # than one termination condition fires on the same transition.
        bad_events = bad_flags
        done_events = done_flags & ~bad_events
        timeout_events = timeout_flags & ~bad_events & ~done_events
        self._rollout_done_count += int(done_events.sum())
        self._rollout_bad_done_count += int(bad_events.sum())
        self._rollout_timeout_count += int(timeout_events.sum())

        if isinstance(infos, dict):
            flat_bad_events = bad_events.reshape(-1)
            for key in self.TERMINATION_INFO_KEYS:
                value = infos.get(key)
                if value is None:
                    continue
                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                reason = np.asarray(value, dtype=bool).reshape(-1)
                if reason.size == flat_bad_events.size:
                    self._rollout_termination_reasons[key] += int(
                        np.count_nonzero(reason & flat_bad_events)
                    )

            phase = infos.get('mission_phase')
            if phase is not None:
                if torch.is_tensor(phase):
                    phase = phase.detach().cpu().numpy()
                phase = np.asarray(phase).reshape(-1)
                if phase.size == flat_bad_events.size:
                    for phase_id in np.unique(phase[flat_bad_events]):
                        count = int(np.count_nonzero(
                            flat_bad_events & (phase == phase_id)
                        ))
                        phase_id = int(phase_id)
                        self._rollout_failure_phases[phase_id] = (
                            self._rollout_failure_phases.get(phase_id, 0)
                            + count
                        )

        reset_indices = np.argwhere(reset_flags)
        for env_index, agent_index in reset_indices:
            self._rollout_completed_returns.append(
                float(self._running_episode_returns[env_index, agent_index])
            )
            self._running_episode_returns[env_index, agent_index] = 0.0

    def _append_rollout_train_infos(self, train_infos):
        completed_returns = self._rollout_completed_returns
        completed_episodes = len(completed_returns)
        train_infos["average_episode_rewards"] = (
            float(np.mean(completed_returns))
            if completed_episodes > 0 else 0.0
        )
        train_infos["rollout/completed_episode_count"] = completed_episodes
        train_infos["rollout/step_reward_mean"] = (
            self._rollout_step_reward_sum
            / max(self._rollout_transition_count, 1)
        )
        done_count = self._rollout_done_count
        bad_done_count = self._rollout_bad_done_count
        timeout_count = self._rollout_timeout_count
        train_infos["rollout/clean_done_count"] = done_count + timeout_count
        train_infos["rollout/bad_done_count"] = bad_done_count
        train_infos["rollout/timeout_count"] = timeout_count
        train_infos["rollout/done_count"] = done_count
        train_infos["rollout/success_count"] = done_count
        train_infos["rollout/termination_count"] = completed_episodes
        train_infos["rollout/bad_done_rate"] = (
            bad_done_count / max(self._rollout_transition_count, 1)
        )
        train_infos["rollout/bad_done_fraction"] = (
            bad_done_count / max(completed_episodes, 1)
        )
        for key, count in self._rollout_termination_reasons.items():
            train_infos[f"termination/{key}"] = count
        task = getattr(getattr(self.envs, 'gpu_vec_env', None), 'task', None)
        phase_names = getattr(task, 'PHASE_NAMES', ())
        for phase_id, count in self._rollout_failure_phases.items():
            phase_name = (
                phase_names[phase_id]
                if 0 <= phase_id < len(phase_names)
                else str(phase_id)
            )
            train_infos[f"termination/phase_{phase_name}"] = count

    def _init_actor_from_checkpoint(self, ckpt_path):
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict):
            if 'policy' in state:
                state = state['policy']
            elif 'state_dict' in state:
                state = state['state_dict']
        self.policy.actor.load_state_dict(state)
        reset_log_std = getattr(self.all_args, 'reset_action_log_std', None)
        if reset_log_std is not None:
            action_out = getattr(getattr(self.policy.actor, 'act', None), 'action_out', None)
            log_std = getattr(action_out, 'log_std', None)
            if log_std is None:
                raise ValueError(
                    '--reset-action-log-std requires a continuous Box action policy'
                )
            with torch.no_grad():
                log_std.fill_(float(reset_log_std))
            logging.info(
                'reset actor action log_std to %.4f after loading %s',
                float(reset_log_std), ckpt_path,
            )
        logging.info(f"initialised PPO actor from {ckpt_path}")

    def run(self):
        self.warmup() #初始化

        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads #计算总episode

        for episode in range(episodes):
            self._begin_rollout_metrics()
            # global profile
            # profile.enable()
            for step in range(self.buffer_size):
                # Sample actions，从PPO算法中获取动作与价值
                values, actions, hybrid_actions, action_log_probs, \
                    rnn_states_actor, rnn_states_critic, cost_values, \
                    rnn_states_cost_critic = self.collect(step)
                
                # print('net output actions=', actions[0], action_log_probs[0])

                # Obser reward and next obs，智能体与环境交互，更新飞行状态，获取奖励
                obs, rewards, dones, bad_dones, exceed_time_limits, infos = self.envs.step(hybrid_actions)
                self._record_rollout_transition(
                    rewards, dones, bad_dones, exceed_time_limits, infos
                )
                critic_obs = (
                    self.envs.critic_obs()
                    if self.use_privileged_critic else obs
                )
                # print('action:', actions)
                # print(episode, step, rewards)

                # Extra recorded information
                # for info in infos:
                #     if 'heading_turn_counts' in info:
                #         heading_turns_list.append(info['heading_turn_counts'])

                costs = (
                    bad_dones.astype(np.float32)
                    if getattr(self.all_args, 'use_cost_constraints', False)
                    else None
                )
                data = (
                    obs, critic_obs, actions, rewards, dones, bad_dones,
                    exceed_time_limits,
                    action_log_probs, values, rnn_states_actor, rnn_states_critic,
                    costs, cost_values, rnn_states_cost_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            # profile.disable()
            # s = io.StringIO()
            # sortby = pstats.SortKey.CUMULATIVE
            # ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # print(s.getvalue())
            # pdb.set_trace()

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                self._append_rollout_train_infos(train_infos)
                self._append_task_train_infos(train_infos)
                elapsed = end - start
                fps = int(self.total_num_steps / max(elapsed, 1e-6))
                self._update_best_train_infos(train_infos)
                self._log_train_summary(train_infos, episode, episodes, elapsed, fps)
                self._append_progress_csv(train_infos, episode, episodes, elapsed, fps)

                # if len(heading_turns_list):
                #     train_infos["average_heading_turns"] = np.mean(heading_turns_list)
                #     logging.info("average heading turns is {}".format(train_infos["average_heading_turns"]))
                self.log_info(train_infos, self.total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and episode != 0 and self.use_eval:
                self.eval(self.total_num_steps)

            # save model
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

    def _append_task_train_infos(self, train_infos):
        env = getattr(self.envs, 'gpu_vec_env', None)
        task = getattr(env, 'task', None)
        if task is None or not hasattr(task, 'get_training_metrics'):
            return
        metrics = task.get_training_metrics()
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.detach().float().mean().item()
            train_infos[key] = value

    def _as_float(self, infos, key, default=None):
        if key not in infos:
            return default
        value = infos[key]
        if torch.is_tensor(value):
            value = value.detach().float().mean().item()
        elif isinstance(value, np.ndarray):
            value = float(np.mean(value))
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _fmt(self, value, precision=3, default="n/a"):
        if value is None:
            return default
        if not math.isfinite(float(value)):
            return default
        return f"{float(value):.{precision}f}"

    def _fmt_deg(self, value_rad, precision=1, default="n/a"):
        if value_rad is None:
            return default
        return self._fmt(float(value_rad) * 180.0 / math.pi, precision, default)

    def _update_best_train_infos(self, infos):
        avg_reward = self._as_float(infos, "average_episode_rewards")
        if avg_reward is not None and avg_reward > self._best_average_reward:
            self._best_average_reward = avg_reward

        tracking_error = self._as_float(
            infos, "rc_human/tracking_vel_error_mean",
            self._as_float(infos, "rc_human/tracking_error_mean"),
        )
        bad_done_fraction = self._as_float(infos, "rollout/bad_done_fraction")
        if (
            tracking_error is not None
            and bad_done_fraction is not None
            and bad_done_fraction <= 0.0
            and tracking_error < self._best_tracking_error
        ):
            self._best_tracking_error = tracking_error
            self._best_bad_done_fraction = bad_done_fraction

    def _nonzero_mode_summary(self, infos):
        parts = []
        for mode_id in range(10):
            key = f"rc_human/mode_{mode_id}_fraction"
            frac = self._as_float(infos, key)
            if frac is not None and frac > 1e-3:
                parts.append(f"m{mode_id}:{frac:.2f}")
        return " ".join(parts) if parts else "n/a"

    def _mission_phase_names(self):
        task = getattr(getattr(self.envs, 'gpu_vec_env', None), 'task', None)
        return tuple(getattr(task, 'PHASE_NAMES', (
            'takeoff', 'rotor_climb', 'transition', 'fixed_wing',
            'back_transition', 'vertical_landing',
        )))

    def _termination_summary(self, infos):
        parts = []
        for key in self.TERMINATION_INFO_KEYS:
            count = self._as_float(infos, f'termination/{key}', 0.0)
            if count is not None and count > 0.0:
                parts.append(f'{key}={int(count)}')
        for phase_name in self._mission_phase_names():
            count = self._as_float(
                infos, f'termination/phase_{phase_name}', 0.0
            )
            if count is not None and count > 0.0:
                parts.append(f'phase_{phase_name}={int(count)}')
        return ' '.join(parts) if parts else 'none'

    def _log_train_summary(self, infos, episode, episodes, elapsed, fps):
        avg_reward = self._as_float(infos, "average_episode_rewards")
        total_reward = self._as_float(infos, "reward/total_mean")
        success_count = self._as_float(infos, "rollout/success_count", 0.0)
        bad_done = self._as_float(infos, "rollout/bad_done_count", 0.0)
        timeout_count = self._as_float(infos, "rollout/timeout_count", 0.0)
        completed_count = self._as_float(
            infos, "rollout/completed_episode_count", 0.0
        )
        step_reward_mean = self._as_float(
            infos, "rollout/step_reward_mean", 0.0
        )
        bad_done_rate = self._as_float(infos, "rollout/bad_done_rate", 0.0)
        bad_done_fraction = self._as_float(infos, "rollout/bad_done_fraction", 0.0)

        vel_err = self._as_float(
            infos, "rc_human/tracking_vel_error_mean",
            self._as_float(infos, "rc_human/tracking_error_mean"),
        )
        yaw_err = self._as_float(infos, "rc_human/tracking_yaw_error_mean")
        att_err = self._as_float(infos, "rc_human/tracking_attitude_error_mean")
        level_mean = self._as_float(infos, "rc_human/curriculum_level_mean")
        level_max = self._as_float(infos, "rc_human/curriculum_level_max")
        level_limit = self._as_float(infos, "rc_human/curriculum_level_limit")
        rate_limited = self._as_float(infos, "rc_human/command_rate_limited_fraction")
        raw_delta = self._as_float(infos, "rc_human/command_raw_delta_mean")

        policy_loss = self._as_float(infos, "policy_loss")
        value_loss = self._as_float(infos, "value_loss")
        entropy_loss = self._as_float(infos, "policy_entropy_loss")
        ratio = self._as_float(infos, "ratio")
        approx_kl = self._as_float(infos, "approx_kl")
        actor_grad = self._as_float(infos, "actor_grad_norm")
        critic_grad = self._as_float(infos, "critic_grad_norm")
        skipped_updates = self._as_float(infos, "skipped_updates", 0.0)

        reward_terms = []
        for key, label in [
            ("reward/vel_gaussian_mean", "vel"),
            ("reward/rel_tracking_mean", "rel"),
            ("reward/rel_precision_mean", "prec"),
            ("reward/yaw_mean", "yaw"),
            ("reward/yaw_precision_mean", "yawP"),
            ("reward/yaw_rate_mean", "yawR"),
            ("reward/attitude_mean", "att"),
            ("reward/omega_mean", "omega"),
            ("reward/smooth_mean", "smooth"),
            ("reward/overshoot_mean", "over"),
            ("reward/adaptive_damping_mean", "damp"),
            ("reward/speed_margin_mean", "speedM"),
            ("reward/attitude_margin_mean", "attM"),
        ]:
            value = self._as_float(infos, key)
            if value is not None:
                reward_terms.append(f"{label}={value:.3f}")

        logging.info(
            "\n"
            f"[train] scenario={self.all_args.scenario_name} algo={self.algorithm_name} "
            f"exp={self.experiment_name}\n"
            f"        update={episode}/{episodes} steps={self.total_num_steps}/{self.num_env_steps} "
            f"fps={fps} elapsed={elapsed/60.0:.1f}min\n"
            f"        reward avg_ep={self._fmt(avg_reward, 2)} total_step={self._fmt(total_reward, 3)} "
            f"best_avg_ep={self._fmt(self._best_average_reward, 2)}\n"
            f"        tracking vel={self._fmt(vel_err, 4)}m/s yaw={self._fmt_deg(yaw_err)}deg "
            f"att={self._fmt_deg(att_err)}deg\n"
            f"        done success={int(success_count)} bad={int(bad_done)} "
            f"timeout={int(timeout_count)} completed={int(completed_count)} "
            f"bad_rate={self._fmt(bad_done_rate, 5)} "
            f"terminal_bad_frac={self._fmt(bad_done_fraction, 3)} "
            f"best_clean_vel={self._fmt(self._best_tracking_error, 4)}\n"
            f"        rollout step_reward={self._fmt(step_reward_mean, 4)}\n"
            f"        curriculum level={self._fmt(level_mean, 1)}/{self._fmt(level_limit, 0)} "
            f"max={self._fmt(level_max, 0)} modes=[{self._nonzero_mode_summary(infos)}]\n"
            f"        command rate_limited={self._fmt(rate_limited, 2)} "
            f"raw_delta={self._fmt(raw_delta, 3)}\n"
            f"        ppo policy={self._fmt(policy_loss, 4)} value={self._fmt(value_loss, 4)} "
            f"entropy={self._fmt(entropy_loss, 4)} ratio={self._fmt(ratio, 3)} "
            f"kl={self._fmt(approx_kl, 5)} gradA={self._fmt(actor_grad, 2)} "
            f"gradC={self._fmt(critic_grad, 2)} skipped={self._fmt(skipped_updates, 2)}"
        )
        if reward_terms:
            logging.info("        reward_terms " + " ".join(reward_terms))
        if bad_done > 0:
            logging.info(
                "        termination reasons=[{}]".format(
                    self._termination_summary(infos)
                )
            )

        if "constraint/episode_cost" in infos:
            logging.info(
                "        constraint cost={} limit={} lagrange={}->{}".format(
                    self._fmt(self._as_float(infos, "constraint/episode_cost"), 4),
                    self._fmt(self._as_float(infos, "constraint/cost_limit"), 4),
                    self._fmt(self._as_float(infos, "constraint/lagrange_before_update"), 3),
                    self._fmt(self._as_float(infos, "constraint/lagrange_after_update"), 3),
                )
            )
        mission_phase_parts = []
        for phase_name in self._mission_phase_names():
            value = self._as_float(
                infos, f'mission/phase_{phase_name}_fraction'
            )
            if value is not None:
                mission_phase_parts.append(f'{phase_name}={value:.2f}')
        if mission_phase_parts:
            logging.info(
                "        mission phases=[{}] landing_dist={}m waypoint_dist={}m "
                "reward_constraint={} success={} failure={}".format(
                    ' '.join(mission_phase_parts),
                    self._fmt(self._as_float(
                        infos, 'mission/landing_distance_mean'), 1),
                    self._fmt(self._as_float(
                        infos, 'mission/waypoint_distance_mean'), 1),
                    self._fmt(self._as_float(
                        infos, 'mission/reward_constraint_penalty_mean'), 3),
                    self._fmt(self._as_float(
                        infos, 'mission/success_count'), 0),
                    self._fmt(self._as_float(
                        infos, 'mission/failure_count'), 0),
                )
            )
            mission_reach_parts = []
            for phase_name in self._mission_phase_names()[1:]:
                value = self._as_float(
                    infos, f'mission/reach_{phase_name}_rate'
                )
                if value is not None:
                    mission_reach_parts.append(f'{phase_name}={value:.3f}')
            if mission_reach_parts:
                logging.info(
                    "        mission reach=[{}] opportunities={}".format(
                        ' '.join(mission_reach_parts),
                        self._fmt(self._as_float(
                            infos, 'mission/rollout_episode_opportunities'
                        ), 0),
                    )
                )
            hover_stable = self._as_float(
                infos, 'mission/hover_stable_fraction'
            )
            hover_progress = self._as_float(
                infos, 'mission/hover_hold_progress_mean'
            )
            if hover_stable is not None:
                logging.info(
                    "        hover stable_fraction={} hold_progress={}".format(
                        self._fmt(hover_stable, 3),
                        self._fmt(hover_progress, 3),
                    )
                )

    def _append_progress_csv(self, infos, episode, episodes, elapsed, fps):
        csv_path = Path(self.run_dir) / "training_progress.csv"
        fields = [
            "update", "updates_total", "steps", "fps", "elapsed_s",
            "average_episode_rewards", "reward/total_mean",
            "rollout/clean_done_count", "rollout/bad_done_count",
            "rollout/done_count", "rollout/timeout_count",
            "rollout/success_count",
            "rollout/completed_episode_count", "rollout/termination_count",
            "rollout/step_reward_mean", "rollout/bad_done_rate",
            "rollout/bad_done_fraction",
            "rc_human/curriculum_level_mean", "rc_human/curriculum_level_max",
            "rc_human/curriculum_level_limit",
            "rc_human/tracking_vel_error_mean", "rc_human/tracking_error_mean",
            "rc_human/tracking_yaw_error_mean", "rc_human/tracking_attitude_error_mean",
            "rc_human/command_rate_limited_fraction",
            "rc_human/command_raw_delta_mean",
            "reward/vel_gaussian_mean", "reward/rel_tracking_mean",
            "reward/rel_precision_mean", "reward/yaw_mean",
            "reward/yaw_precision_mean", "reward/yaw_rate_mean",
            "reward/attitude_mean", "reward/omega_mean",
            "reward/smooth_mean", "reward/overshoot_mean",
            "reward/adaptive_damping_mean", "reward/speed_margin_mean",
            "reward/attitude_margin_mean",
            "policy_loss", "value_loss", "policy_entropy_loss",
            "ratio", "approx_kl", "actor_grad_norm",
            "critic_grad_norm", "skipped_updates",
            "constraint/episode_cost", "constraint/cost_limit",
            "constraint/lagrange_before_update", "constraint/lagrange_after_update",
        ]
        for mode_id in range(10):
            fields.append(f"rc_human/mode_{mode_id}_fraction")
        fields.extend([
            'mission/waypoint_distance_mean',
            'mission/landing_distance_mean',
            'mission/gps_age_mean',
            'mission/success_count',
            'mission/failure_count',
            'mission/reward_constraint_penalty_mean',
            'mission/hover_stable_fraction',
            'mission/hover_hold_progress_mean',
        ])
        for phase_name in self._mission_phase_names():
            fields.append(f'mission/phase_{phase_name}_fraction')
            fields.append(f'mission/start_{phase_name}_fraction')
            if phase_name != 'takeoff':
                fields.append(f'mission/reach_{phase_name}_count')
                fields.append(f'mission/reach_{phase_name}_rate')
        fields.append('mission/rollout_episode_opportunities')
        for key in self.TERMINATION_INFO_KEYS:
            fields.append(f'termination/{key}')
        for phase_name in self._mission_phase_names():
            fields.append(f'termination/phase_{phase_name}')

        row = {
            "update": episode,
            "updates_total": episodes,
            "steps": self.total_num_steps,
            "fps": fps,
            "elapsed_s": elapsed,
        }
        for field in fields:
            if field in row:
                continue
            value = self._as_float(infos, field)
            row[field] = "" if value is None else value

        need_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if need_header:
                writer.writeheader()
            writer.writerow(row)
                

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        critic_obs = (
            self.envs.critic_obs() if self.use_privileged_critic else obs
        )
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()
        self.buffer.critic_obs[0] = critic_obs.copy()

    def _apply_action_constraints(self, actions, vec_env=None):
        """Build the actuator command sent to the simulator.

        The task owns phase-dependent actuator gates (vertical mode and the
        reverse-transition pusher cap).  The sampled policy action and its
        log-probability remain unchanged in the PPO buffer; deterministic
        flight-mode gates affect only the physical actuator command.
        """
        if vec_env is None:
            vec_env = self.envs
        task_env = getattr(vec_env, 'gpu_vec_env', None)
        task = getattr(task_env, 'task', None)
        if task is not None and hasattr(task, 'maybe_override_action'):
            # BaseEnv.step() auto-resets terminated rows.  Perform that reset
            # before computing phase-dependent gates so a mixed batch cannot
            # apply the previous episode's mode limits to a fresh takeoff.
            # The internal reset is callback-free and therefore does not
            # advance phase hold counters.
            if hasattr(task_env, 'reset'):
                task_env.reset(update_observation=False)
            actions = task.maybe_override_action(task_env, actions)
            # BaseEnv.step consumes this marker and skips its own task gate.
            # Without it vertical-mode differential scaling and the
            # back-transition pusher cap would be applied twice.
            task_env._action_preprocessed = True
        if getattr(self.all_args, 'freeze_head_motor', False):
            actions = actions.clone()
            actions[:, 0] = -1.0
        return actions

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        obs = np.concatenate(self.buffer.obs[step])
        critic_obs = np.concatenate(self.buffer.critic_obs[step])
        rnn_actor = np.concatenate(self.buffer.rnn_states_actor[step])
        rnn_critic = np.concatenate(self.buffer.rnn_states_critic[step])
        masks = np.concatenate(self.buffer.masks[step])
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(
                obs, rnn_actor, rnn_critic, masks,
                critic_obs=critic_obs,
            )
        cost_values = None
        rnn_states_cost_critic = None
        if getattr(self.all_args, 'use_cost_constraints', False):
            rnn_cost_critic = np.concatenate(self.buffer.rnn_states_cost_critic[step])
            cost_values, rnn_states_cost_critic = self.policy.get_cost_values(
                critic_obs, rnn_cost_critic, masks
            )
        # The policy action is the random variable sampled by PPO.  Flight-mode
        # gates are deterministic environment dynamics, so their transformed
        # command is sent to the simulator but must not replace the sampled
        # action/log-prob pair in the rollout buffer.  Treating fixed gate
        # values (for example pusher=-1) as policy samples creates a false KL
        # jump and was the main source of skipped updates in this mission.
        rollout_actions = actions
        stored_actions = actions
        recompute_action_log_probs = False
        #可以在此处测试各种添加扰动的算法，并将noise=‘新增的扰动’
        noise = 'no_noise'
        if noise == 'random_noise':
            delta = torch.rand_like(rollout_actions) * 10. - 5.
            rollout_actions = (1 - 0.02) * actions + 0.02 * delta
            stored_actions = rollout_actions
            recompute_action_log_probs = True
        elif noise == 'sudden_noise':
            adv_flag = 1 if 0.02 >= np.random.random() else 0
            if adv_flag == 1:
                rollout_actions = torch.rand_like(rollout_actions) * 10. - 5.
                stored_actions = rollout_actions
                recompute_action_log_probs = True
            else:
                rollout_actions = actions

        simulator_actions = self._apply_action_constraints(rollout_actions)
        if recompute_action_log_probs:
            action_log_probs, _ = self.policy.actor.evaluate_actions(
                obs, rnn_actor, stored_actions, masks
            )

        # split parallel data [N * M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(
            _t2n(stored_actions), self.n_rollout_threads
        ))
        simulator_actions = np.array(np.split(
            _t2n(simulator_actions), self.n_rollout_threads
        ))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        if getattr(self.all_args, 'use_cost_constraints', False):
            cost_values = np.array(np.split(_t2n(cost_values), self.n_rollout_threads))
            rnn_states_cost_critic = np.array(np.split(
                _t2n(rnn_states_cost_critic), self.n_rollout_threads
            ))
        return (
            values, actions, simulator_actions, action_log_probs,
            rnn_states_actor, rnn_states_critic,
            cost_values, rnn_states_cost_critic,
        )

    def insert(self, data: List[np.ndarray]):
        obs, critic_obs, actions, rewards, dones, bad_dones, exceed_time_limits, action_log_probs, \
            values, rnn_states_actor, rnn_states_critic, costs, cost_values, \
            rnn_states_cost_critic = data

        dones_env = np.any(dones.squeeze(axis=-1), axis=-1)
        bad_dones_env = np.any(bad_dones.squeeze(axis=-1), axis=-1)
        reset_env = np.any(
            (dones | bad_dones | exceed_time_limits).squeeze(axis=-1),
            axis=-1,
        )

        rnn_states_actor[reset_env == True] = np.zeros(((reset_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[reset_env == True] = np.zeros(((reset_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)
        if getattr(self.all_args, 'use_cost_constraints', False):
            rnn_states_cost_critic[reset_env == True] = np.zeros(
                ((reset_env == True).sum(), *rnn_states_cost_critic.shape[1:]),
                dtype=np.float32,
            )

        # The vector environment auto-resets on time limits too.  Treat every
        # reset as a trajectory boundary so GAE and recurrent state cannot
        # leak from a timed-out mission into its freshly reset observation.
        terminal_env = reset_env
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[terminal_env == True] = np.zeros(
            ((terminal_env == True).sum(), self.num_agents, 1),
            dtype=np.float32,
        )

        bad_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        bad_masks[bad_dones_env == True] = np.zeros(((bad_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # if(self.all_args.scenario_name=='tracking' and self.all_args.tracking_cir==True):
            
        # self.envs.gpu_vec_env.task.max_distance[dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.max_distance[dones_env == True] + 30., max = 2000.)
        # self.envs.gpu_vec_env.task.max_distance[bad_dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.max_distance[bad_dones_env == True] - 30., min = 200.)
        # self.envs.gpu_vec_env.task.max_yaw[dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.max_yaw[dones_env == True] + 0.01, max = torch.pi / 6)
        # self.envs.gpu_vec_env.task.min_yaw[dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.min_yaw[dones_env == True] - 0.01, min = -torch.pi / 6)
        # self.envs.gpu_vec_env.task.max_yaw[bad_dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.max_yaw[bad_dones_env == True] - 0.01, min = 0.)
        # self.envs.gpu_vec_env.task.min_yaw[bad_dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.min_yaw[bad_dones_env == True] + 0.01, max = 0.)
        # self.envs.gpu_vec_env.task.max_pitch[dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.max_pitch[dones_env == True] + 0.01, max = torch.pi / 6)
        # self.envs.gpu_vec_env.task.min_pitch[dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.min_pitch[dones_env == True] - 0.01, min = -torch.pi / 6)
        # self.envs.gpu_vec_env.task.max_pitch[bad_dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.max_pitch[bad_dones_env == True] - 0.01, min = 0.)
        # self.envs.gpu_vec_env.task.min_pitch[bad_dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.min_pitch[bad_dones_env == True] + 0.01, max = 0.)
        # print('max_yaw,max_pitch:',self.envs.gpu_vec_env.task.max_yaw[0],self.envs.gpu_vec_env.task.max_pitch[0])
        # print('maxdis:',self.envs.gpu_vec_env.task.max_distance[0])

        self.buffer.insert(
            obs, actions, rewards, masks, action_log_probs, values,
            rnn_states_actor, rnn_states_critic, bad_masks,
            costs=costs,
            cost_value_preds=cost_values,
            rnn_states_cost_critic=rnn_states_cost_critic,
            critic_obs=critic_obs,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)

        while total_episodes < self.eval_episodes:

            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks), deterministic=True)
            eval_actions = self._apply_action_constraints(eval_actions, self.eval_envs)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_bad_dones, eval_exceed_time_limits, eval_infos = self.eval_envs.step(eval_actions)

            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            eval_reset_env = np.all((eval_dones + eval_bad_dones + eval_exceed_time_limits).squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_reset_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_reset_env == True])
            eval_cumulative_rewards[eval_reset_env == True] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]), dtype=np.float32)
            eval_rnn_states[eval_reset_env == True] = np.zeros(((eval_reset_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean(axis=1)  # shape: [num_agents, 1]
        logging.info(" eval average episode rewards: " + str(np.mean(eval_infos['eval_average_episode_rewards'])))
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

    @torch.no_grad()
    def render(self):
        logging.info("\nStart render ...")
        self.render_opponent_index = self.all_args.render_opponent_index
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = self._apply_action_constraints(render_actions)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            
            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_bad_dones, render_exceed_time_limits, render_infos = self.envs.step(render_actions)
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode):
        save_dir = Path(str(self.save_dir) + '/episode_{}'.format(str(episode)))
        os.makedirs(str(save_dir))
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(save_dir) + '/actor_latest.ckpt')
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(save_dir) + '/critic_latest.ckpt')
        if getattr(self.policy, 'cost_critic', None) is not None:
            torch.save(
                self.policy.cost_critic.state_dict(),
                str(save_dir) + '/cost_critic_latest.ckpt',
            )
