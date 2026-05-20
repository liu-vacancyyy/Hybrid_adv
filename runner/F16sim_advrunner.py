import os
import sys
import time
import torch
import logging
import numpy as np
import torch.nn as nn
from typing import List
from pathlib import Path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from base_runner import Runner, ReplayBuffer, AdvReplayBuffer
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
from algorithms.ppo.ppo_policy import PPOPolicy as Policy
from algorithms.rparl.adversary import Adversary
from basicutils.torch_utils import *
from algorithms.utils.utils import check

def _t2n(x):
    return x.detach().cpu().numpy()

class F16SimAdvRunner(Runner):

    def load(self):
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.agents

        # policy & algorithm
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)
        self.adv = Adversary(self.all_args, self.obs_space, self.act_space, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)
        self.adv_buffer = AdvReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)

        self.adv_action_lr = self.all_args.adv_action_lr

        if self.model_dir is not None:
            self.restore()

    def train(self):
        self.policy.prep_training()
        train_infos = self.trainer.train(self.policy, self.buffer)
        self.buffer.after_update()
        return train_infos
    
    def adv_train(self):
        self.adv.train()
        adv_train_infos=self.adv_update()
        self.adv_buffer.after_update()
        return adv_train_infos

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    def run(self):
        self.warmup()

        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads

        for episode in range(episodes):
            # global profile
            # profile.enable()
            for step in range(self.buffer_size):
                # Sample actions
                values, actions, hybrid_actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)
                
                # Obser reward and next obs
                obs, rewards, dones, bad_dones, exceed_time_limits, infos = self.envs.step(hybrid_actions)

                # dones_num += np.sum(dones)
                # all_num += dones_num + np.sum(bad_dones) + np.sum(exceed_time_limits)
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
            # print('success rate===', dones_num / all_num)
            # if(dones_num / all_num > max_suc):
            #     max_suc = dones_num / all_num
            #     max_epi = episode
            # print('max_suc===', max_suc)
            # print('max_epi===', max_epi)
            self.compute()
            train_infos = self.train()
            if self.all_args.noise == 'adv_noise':
                adv_train_infos = self.adv_train()
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

        end = time.time()
        print("total time=",end-start,'s')

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        obs = obs = check(np.concatenate(self.buffer.obs[step])).to(**self.tpdv)
        rnn_states_actor = check(np.concatenate(self.buffer.rnn_states_actor[step])).to(**self.tpdv)
        rnn_states_critic = check(np.concatenate(self.buffer.rnn_states_critic[step])).to(**self.tpdv)
        masks = check(np.concatenate(self.buffer.masks[step])).to(**self.tpdv)
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(obs, rnn_states_actor, rnn_states_critic, masks)
        hybrid_actions = actions

        if self.all_args.noise == 'adv_noise':
            noise = torch_rand_float(-1, 1, actions.shape, device=self.device)
            adv_action_init = actions + noise * self.all_args.magnitude
            actions_q = self.adv.evaluate_forward(obs, actions)
            adv_q = self.adv.evaluate_forward(obs, adv_action_init)
            grad_a = compute_grad(actions_q, adv_q, actions, adv_action_init)
            adv_action_new = adv_action_init - (self.adv_action_lr * grad_a)
            adv_q_new = self.adv.evaluate_forward(obs, adv_action_new)

            iter = 0
            adv_action = adv_action_init
            while iter < 5:  # 这里有bug，第一段是所有环境的动作，所以不合适
                # 用env_id的形式屏蔽是好方法，一定注意改一下，不然我怀疑一次就跳出循环了
                # print('Optimizing')
                stop_id = torch.unique((torch.abs(adv_action - adv_action_new) > self.all_args.epsilon).nonzero(as_tuple=False)[:,0])
                adv_q[stop_id] = self.adv.evaluate_forward(obs[stop_id], adv_action[stop_id])
                adv_q_new[stop_id] = self.adv.evaluate_forward(obs[stop_id], adv_action_new[stop_id])

                grad_a[stop_id] = compute_grad(adv_q[stop_id], adv_q_new[stop_id], adv_action[stop_id], adv_action_new[stop_id])  # 这里还有个bug，adv_q_new没有更新，，，adv_q也没有更新，，，后续需要修复代码
                adv_action[stop_id] = adv_action_new[stop_id]
                adv_action_new[stop_id] = adv_action[stop_id] - (self.adv_action_lr * grad_a[stop_id])
                iter += 1
            delta = adv_action_new - actions
            proj_spatial_delta = l2_spatial_project(delta, self.all_args.epsilon)
            hybrid_actions = actions + proj_spatial_delta
        elif self.all_args.noise == 'random_noise':
            delta = torch.rand_like(hybrid_actions) * 10. - 5.
            hybrid_actions = (1 - self.all_args.random_weight) * actions + self.all_args.random_weight * delta
        elif self.all_args.noise == 'sudden_noise':
            adv_flag = 1 if self.all_args.sudden_mag >= np.random.random() else 0
            if adv_flag == 1:
                hybrid_actions = torch.rand_like(hybrid_actions) * 10. - 5.
            else:
                hybrid_actions = actions

        # split parallel data [N * M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        # adv_values = np.array(np.split(_t2n(adv_values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        hybrid_actions = np.array(np.split(_t2n(hybrid_actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        return values, actions, hybrid_actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, bad_dones, exceed_time_limits, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        dones_env = np.any(dones.squeeze(axis=-1), axis=-1)
        bad_dones_env = np.any(bad_dones.squeeze(axis=-1), axis=-1)
        reset_env = np.any((dones + bad_dones + exceed_time_limits).squeeze(axis=-1), axis=-1)
        adv_reset = (~reset_env).reshape(-1, 1, 1)

        rnn_states_actor[reset_env == True] = np.zeros(((reset_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[reset_env == True] = np.zeros(((reset_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        bad_masks[bad_dones_env == True] = np.zeros(((bad_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.envs.gpu_vec_env.task.max_yaw[dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.max_yaw[dones_env == True] + 0.01, max = torch.pi / 3)
        self.envs.gpu_vec_env.task.min_yaw[dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.min_yaw[dones_env == True] - 0.01, min = -torch.pi / 3)
        self.envs.gpu_vec_env.task.max_yaw[bad_dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.max_yaw[bad_dones_env == True] - 0.01, min = 0.)
        self.envs.gpu_vec_env.task.min_yaw[bad_dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.min_yaw[bad_dones_env == True] + 0.01, max = 0.)
        self.envs.gpu_vec_env.task.max_pitch[dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.max_pitch[dones_env == True] + 0.01, max = torch.pi / 6)
        self.envs.gpu_vec_env.task.min_pitch[dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.min_pitch[dones_env == True] - 0.01, min = -torch.pi / 6)
        self.envs.gpu_vec_env.task.max_pitch[bad_dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.max_pitch[bad_dones_env == True] - 0.01, min = 0.)
        self.envs.gpu_vec_env.task.min_pitch[bad_dones_env == True] = torch.clip(self.envs.gpu_vec_env.task.min_pitch[bad_dones_env == True] + 0.01, max = 0.)

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic, bad_masks)
        self.adv_buffer.insert(obs, actions, adv_reset, rewards)

    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)
        if self.all_args.noise == 'adv_noise':
            self.adv_buffer.compute_returns()

    def adv_update(self):
        mean_value_loss = 0  # 值函数损失均值
        generator = self.adv_buffer.mini_batch_generator(self.all_args.adv_num_mini_batch, self.all_args.adv_num_learning_epochs)
        for obs_batch, actions_batch, target_values_batch in generator:
            value_batch = self.adv.evaluate_backrward(obs_batch, actions_batch)
            target_values_batch = check(target_values_batch).to(**self.tpdv)
            value_loss = (target_values_batch - value_batch).pow(2).mean()
            loss = self.all_args.adv_value_loss_coef * value_loss
            self.adv.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.adv.parameters(), self.all_args.adv_max_grad_norm)
            self.adv.optimizer.step()
            mean_value_loss += value_loss.item()

        num_updates = self.all_args.adv_num_learning_epochs * self.all_args.adv_num_mini_batch
        mean_value_loss /= num_updates
        return mean_value_loss

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

    def multi_eval(self):
        logging.info("\nStart multi evaluation...")
        total_episodes, suc_episodes, eval_episode_rewards = 0, 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        print(eval_masks.shape)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        print(eval_rnn_states.shape)

        while total_episodes < 1000:

            self.policy.prep_rollout()
            with torch.no_grad():
                eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                                np.concatenate(eval_rnn_states),
                                                                np.concatenate(eval_masks), deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

                # Obser reward and next obs
                eval_obs, eval_rewards, eval_dones, eval_bad_dones, eval_exceed_time_limits, eval_infos = self.eval_envs.step(eval_actions)

            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            eval_reset_env = np.all((eval_dones + eval_bad_dones + eval_exceed_time_limits).squeeze(axis=-1), axis=-1)
            suc_episodes += np.sum(eval_dones_env)
            total_episodes += np.sum(eval_reset_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_reset_env == True])
            eval_cumulative_rewards[eval_reset_env == True] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]), dtype=np.float32)
            eval_rnn_states[eval_reset_env == True] = np.zeros(((eval_reset_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean(axis=1)  # shape: [num_agents, 1]
        logging.info(" eval average episode rewards: " + str(np.mean(eval_infos['eval_average_episode_rewards'])))
        logging.info("...End evaluation")
        return suc_episodes / 1000

    @torch.no_grad()
    def render(self):
        logging.info("\nStart render ...")
        self.render_opponent_index = self.all_args.render_opponent_index
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
        total_episodes = 0
        dones = 0
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            
            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_bad_dones, render_exceed_time_limits, render_infos = self.envs.step(render_actions)
            render_episode_rewards += render_rewards
            dones_env = np.all(render_dones.squeeze(axis=-1), axis=-1)
            reset_env = np.all((render_dones + render_bad_dones + render_exceed_time_limits).squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(reset_env)
            dones += np.sum(dones_env)
            if total_episodes == 1000 :
                break
            # self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            # # if render_dones.all():
            # #     break
        print("success rate===", dones/total_episodes)
        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode):
        save_dir = Path(str(self.save_dir) + '/episode_{}'.format(str(episode)))
        os.makedirs(str(save_dir))
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(save_dir) + '/actor_latest.pt')
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(save_dir) + '/critic_latest.pt')