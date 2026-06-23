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

    def load(self):
        if self.all_args.algorithm_name == 'cpo':
            self.all_args.use_cost_constraints = True
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.agents

        # policy & algorithm
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)

        if self.model_dir is not None:
            self.restore()
        if getattr(self.all_args, 'init_actor_ckpt', None):
            self._init_actor_from_checkpoint(self.all_args.init_actor_ckpt)
        self._best_average_reward = -float("inf")
        self._best_tracking_error = float("inf")
        self._best_bad_done_fraction = float("inf")
        self._progress_csv_header_written = False

    def _init_actor_from_checkpoint(self, ckpt_path):
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict):
            if 'policy' in state:
                state = state['policy']
            elif 'state_dict' in state:
                state = state['state_dict']
        self.policy.actor.load_state_dict(state)
        logging.info(f"initialised PPO actor from {ckpt_path}")

    def run(self):
        self.warmup() #初始化

        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads #计算总episode

        for episode in range(episodes):
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
                # print('action:', actions)
                # print(episode, step, rewards)

                # Extra recorded information
                # for info in infos:
                #     if 'heading_turn_counts' in info:
                #         heading_turns_list.append(info['heading_turn_counts'])

                costs = bad_dones.astype(np.float32)
                data = (
                    obs, actions, rewards, dones, bad_dones, exceed_time_limits,
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
                completed_episodes = ((self.buffer.masks[1:] == False).sum()
                                      + (self.buffer.bad_masks[1:] == False).sum())
                completed_episodes = completed_episodes.item()
                if completed_episodes > 0:
                    train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / completed_episodes
                else:
                    train_infos["average_episode_rewards"] = 0.0
                    logging.info("no completed episode in this rollout; set average_episode_rewards=0.0")
                clean_done_count = (self.buffer.masks[1:] == 0).sum().item()
                bad_done_count = (self.buffer.bad_masks[1:] == 0).sum().item()
                train_infos["rollout/clean_done_count"] = clean_done_count
                train_infos["rollout/bad_done_count"] = bad_done_count
                train_infos["rollout/termination_count"] = clean_done_count + bad_done_count
                if clean_done_count + bad_done_count > 0:
                    train_infos["rollout/bad_done_fraction"] = (
                        bad_done_count / (clean_done_count + bad_done_count)
                    )
                else:
                    train_infos["rollout/bad_done_fraction"] = 0.0
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

    def _log_train_summary(self, infos, episode, episodes, elapsed, fps):
        avg_reward = self._as_float(infos, "average_episode_rewards")
        total_reward = self._as_float(infos, "reward/total_mean")
        clean_done = self._as_float(infos, "rollout/clean_done_count", 0.0)
        bad_done = self._as_float(infos, "rollout/bad_done_count", 0.0)
        bad_done_fraction = self._as_float(infos, "rollout/bad_done_fraction", 0.0)

        vel_err = self._as_float(
            infos, "rc_human/tracking_vel_error_mean",
            self._as_float(infos, "rc_human/tracking_error_mean"),
        )
        yaw_err = self._as_float(infos, "rc_human/tracking_yaw_error_mean")
        att_err = self._as_float(infos, "rc_human/tracking_attitude_error_mean")
        valid_frac = self._as_float(infos, "rc_human/success_metric_valid_fraction")
        skipped_frac = self._as_float(infos, "rc_human/success_metric_skipped_fraction")

        level_mean = self._as_float(infos, "rc_human/curriculum_level_mean")
        level_max = self._as_float(infos, "rc_human/curriculum_level_max")
        level_limit = self._as_float(infos, "rc_human/curriculum_level_limit")
        transient = self._as_float(infos, "rc_human/command_transient_fraction")
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
            f"att={self._fmt_deg(att_err)}deg valid={self._fmt(valid_frac, 2)} "
            f"skipped={self._fmt(skipped_frac, 2)}\n"
            f"        done clean={int(clean_done)} bad={int(bad_done)} "
            f"bad_frac={self._fmt(bad_done_fraction, 3)} "
            f"best_clean_vel={self._fmt(self._best_tracking_error, 4)}\n"
            f"        curriculum level={self._fmt(level_mean, 1)}/{self._fmt(level_limit, 0)} "
            f"max={self._fmt(level_max, 0)} modes=[{self._nonzero_mode_summary(infos)}]\n"
            f"        command transient={self._fmt(transient, 2)} rate_limited={self._fmt(rate_limited, 2)} "
            f"raw_delta={self._fmt(raw_delta, 3)}\n"
            f"        ppo policy={self._fmt(policy_loss, 4)} value={self._fmt(value_loss, 4)} "
            f"entropy={self._fmt(entropy_loss, 4)} ratio={self._fmt(ratio, 3)} "
            f"kl={self._fmt(approx_kl, 5)} gradA={self._fmt(actor_grad, 2)} "
            f"gradC={self._fmt(critic_grad, 2)} skipped={self._fmt(skipped_updates, 2)}"
        )
        if reward_terms:
            logging.info("        reward_terms " + " ".join(reward_terms))

        if "constraint/episode_cost" in infos:
            logging.info(
                "        constraint cost={} limit={} lagrange={}->{}".format(
                    self._fmt(self._as_float(infos, "constraint/episode_cost"), 4),
                    self._fmt(self._as_float(infos, "constraint/cost_limit"), 4),
                    self._fmt(self._as_float(infos, "constraint/lagrange_before_update"), 3),
                    self._fmt(self._as_float(infos, "constraint/lagrange_after_update"), 3),
                )
            )

    def _append_progress_csv(self, infos, episode, episodes, elapsed, fps):
        csv_path = Path(self.run_dir) / "training_progress.csv"
        fields = [
            "update", "updates_total", "steps", "fps", "elapsed_s",
            "average_episode_rewards", "reward/total_mean",
            "rollout/clean_done_count", "rollout/bad_done_count",
            "rollout/bad_done_fraction",
            "rc_human/curriculum_level_mean", "rc_human/curriculum_level_max",
            "rc_human/curriculum_level_limit",
            "rc_human/tracking_vel_error_mean", "rc_human/tracking_error_mean",
            "rc_human/tracking_yaw_error_mean", "rc_human/tracking_attitude_error_mean",
            "rc_human/success_metric_valid_fraction",
            "rc_human/success_metric_skipped_fraction",
            "rc_human/command_transient_fraction",
            "rc_human/command_rate_limited_fraction",
            "rc_human/command_raw_delta_mean",
            "reward/vel_gaussian_mean", "reward/rel_tracking_mean",
            "reward/rel_precision_mean", "reward/yaw_mean",
            "reward/yaw_precision_mean", "reward/yaw_rate_mean",
            "reward/attitude_mean", "reward/omega_mean",
            "reward/smooth_mean", "reward/overshoot_mean",
            "reward/adaptive_damping_mean",
            "policy_loss", "value_loss", "policy_entropy_loss",
            "ratio", "approx_kl", "actor_grad_norm",
            "critic_grad_norm", "skipped_updates",
            "constraint/episode_cost", "constraint/cost_limit",
            "constraint/lagrange_before_update", "constraint/lagrange_after_update",
        ]
        for mode_id in range(10):
            fields.append(f"rc_human/mode_{mode_id}_fraction")

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
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    def _apply_action_constraints(self, actions):
        """Apply environment-side action constraints before rollout/storage.

        For hover training, the head/pusher motor is disabled.  ``action[0] =
        -1`` maps to zero thrust in ``HybridModel.update``.
        """
        if getattr(self.all_args, 'freeze_head_motor', False):
            actions = actions.clone()
            actions[:, 0] = -1.0
        return actions

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        obs = np.concatenate(self.buffer.obs[step])
        rnn_actor = np.concatenate(self.buffer.rnn_states_actor[step])
        rnn_critic = np.concatenate(self.buffer.rnn_states_critic[step])
        masks = np.concatenate(self.buffer.masks[step])
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(obs, rnn_actor, rnn_critic, masks)
        cost_values = None
        rnn_states_cost_critic = None
        if getattr(self.all_args, 'use_cost_constraints', False):
            rnn_cost_critic = np.concatenate(self.buffer.rnn_states_cost_critic[step])
            cost_values, rnn_states_cost_critic = self.policy.get_cost_values(
                obs, rnn_cost_critic, masks
            )
        rollout_actions = actions
        recompute_action_log_probs = False
        #可以在此处测试各种添加扰动的算法，并将noise=‘新增的扰动’
        noise = 'no_noise'
        if noise == 'random_noise':
            delta = torch.rand_like(rollout_actions) * 10. - 5.
            rollout_actions = (1 - 0.02) * actions + 0.02 * delta
            recompute_action_log_probs = True
        elif noise == 'sudden_noise':
            adv_flag = 1 if 0.02 >= np.random.random() else 0
            if adv_flag == 1:
                rollout_actions = torch.rand_like(rollout_actions) * 10. - 5.
                recompute_action_log_probs = True
            else:
                rollout_actions = actions

        rollout_actions = self._apply_action_constraints(rollout_actions)
        if getattr(self.all_args, 'freeze_head_motor', False):
            recompute_action_log_probs = True

        if recompute_action_log_probs:
            # Keep PPO's stored action and old log-prob aligned with the
            # action that actually produced the transition.
            action_log_probs, _ = self.policy.actor.evaluate_actions(
                obs, rnn_actor, rollout_actions, masks)

        # split parallel data [N * M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(rollout_actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        if getattr(self.all_args, 'use_cost_constraints', False):
            cost_values = np.array(np.split(_t2n(cost_values), self.n_rollout_threads))
            rnn_states_cost_critic = np.array(np.split(
                _t2n(rnn_states_cost_critic), self.n_rollout_threads
            ))
        return (
            values, actions, rollout_actions, action_log_probs,
            rnn_states_actor, rnn_states_critic,
            cost_values, rnn_states_cost_critic,
        )

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, bad_dones, exceed_time_limits, action_log_probs, \
            values, rnn_states_actor, rnn_states_critic, costs, cost_values, \
            rnn_states_cost_critic = data

        dones_env = np.any(dones.squeeze(axis=-1), axis=-1)
        bad_dones_env = np.any(bad_dones.squeeze(axis=-1), axis=-1)
        reset_env = np.any((dones + bad_dones + exceed_time_limits).squeeze(axis=-1), axis=-1)

        rnn_states_actor[reset_env == True] = np.zeros(((reset_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[reset_env == True] = np.zeros(((reset_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)
        if getattr(self.all_args, 'use_cost_constraints', False):
            rnn_states_cost_critic[reset_env == True] = np.zeros(
                ((reset_env == True).sum(), *rnn_states_cost_critic.shape[1:]),
                dtype=np.float32,
            )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

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
            eval_actions = self._apply_action_constraints(eval_actions)
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
