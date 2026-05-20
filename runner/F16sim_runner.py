import os
import sys
import time
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
                values, actions, hybrid_actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)
                
                # print('net output actions=', actions[0], action_log_probs[0])

                # Obser reward and next obs，智能体与环境交互，更新飞行状态，获取奖励
                obs, rewards, dones, bad_dones, exceed_time_limits, infos = self.envs.step(hybrid_actions)
                # print('action:', actions)
                # print(episode, step, rewards)

                # Extra recorded information
                # for info in infos:
                #     if 'heading_turn_counts' in info:
                #         heading_turns_list.append(info['heading_turn_counts'])

                data = obs, actions, rewards, dones, bad_dones, exceed_time_limits, action_log_probs, values, rnn_states_actor, rnn_states_critic

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
                logging.info("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                             .format(self.all_args.scenario_name,
                                     self.algorithm_name,
                                     self.experiment_name,
                                     episode,
                                     episodes,
                                     self.total_num_steps,
                                     self.num_env_steps,
                                     int(self.total_num_steps / (end - start))))

                completed_episodes = ((self.buffer.masks[1:] == False).sum()
                                      + (self.buffer.bad_masks[1:] == False).sum())
                completed_episodes = completed_episodes.item()
                if completed_episodes > 0:
                    train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / completed_episodes
                else:
                    train_infos["average_episode_rewards"] = 0.0
                    logging.info("no completed episode in this rollout; set average_episode_rewards=0.0")
                self._append_task_train_infos(train_infos)
                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

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
        return values, actions, rollout_actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, bad_dones, exceed_time_limits, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        dones_env = np.any(dones.squeeze(axis=-1), axis=-1)
        bad_dones_env = np.any(bad_dones.squeeze(axis=-1), axis=-1)
        reset_env = np.any((dones + bad_dones + exceed_time_limits).squeeze(axis=-1), axis=-1)

        rnn_states_actor[reset_env == True] = np.zeros(((reset_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[reset_env == True] = np.zeros(((reset_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

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

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic, bad_masks)

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
