#!/usr/bin/env python
import sys
import os
import traceback
import datetime
import torch
import random
import logging
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config
from envs.control_env import ControlEnv
from envs.planning_env import PlanningEnv
from envs.env_wrappers import GPUVecEnv
from algorithms.ppo.ppo_policy import PPOPolicy as Policy
import torch.utils.tensorboard as tb

def _t2n(x):
    return x.detach().cpu().numpy()

def make_eval_env(all_args):
    def get_env_fn():
        def init_env():
            if all_args.env_name == "Control":
                env = ControlEnv(num_envs=all_args.n_rollout_threads, config=all_args.scenario_name, model= all_args.model_name, random_seed=all_args.seed, device=all_args.device)
            elif all_args.env_name == "Planning":
                env = PlanningEnv(num_envs=all_args.n_rollout_threads, config=all_args.scenario_name, model= all_args.model_name, random_seed=all_args.seed, device=all_args.device)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            return env
        return init_env
    return GPUVecEnv([get_env_fn()])

def parse_args(args, parser):
    group = parser.add_argument_group("F16Sim Env parameters")
    group.add_argument("--env-name", type=str, default='Control',
                       help="specify the name of environment")
    group.add_argument('--scenario-name', type=str, default='singlecombat_simple',
                       help="Which scenario to run on")
    group.add_argument('--model-name', type=str, default='F16',
                       help="Which model to run on")
    all_args = parser.parse_known_args(args)[0]
    return all_args

def multi_eval(envs, args, policy, total_epi):
    logging.info("\nStart multi evaluation...")
    total_episodes, suc_episodes, eval_episode_rewards = 0, 0, []
    eval_cumulative_rewards = np.zeros((args.n_rollout_threads, 1, 1), dtype=np.float32)

    eval_obs = envs.reset()
    eval_masks = np.ones((args.n_rollout_threads, 1, 1), dtype=np.float32)
    eval_rnn_states = np.zeros((args.n_rollout_threads, 1, 1, 128), dtype=np.float32)

    while total_episodes < total_epi:

        policy.prep_rollout()
        with torch.no_grad():
            eval_actions, eval_rnn_states = policy.act(np.concatenate(eval_obs),
                                                        np.concatenate(eval_rnn_states),
                                                        np.concatenate(eval_masks), deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), args.n_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), args.n_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_bad_dones, eval_exceed_time_limits, eval_infos = envs.step(eval_actions)

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
    return suc_episodes / total_epi, np.mean(eval_infos['eval_average_episode_rewards'])
        
def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print(all_args)
    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    
    device = torch.device("cpu")
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device(all_args.device)  # use cude mask to control using which GPU
        # torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        # torch.set_num_threads(all_args.n_training_threads)


    # env init
    eval_envs = make_eval_env(all_args)
    j=2
    noise_factor = j #0:airspeed, 1:mass, 2:IMU, 3:alphabeta, 4:pitot, 5:alt 
    vars = []
    sucs_crpa = []
    sucs_rpa = []
    sucs_rn = []
    sucs_sn = []
    sucs_dr = []
    sucs_ppo = []
    rs_crpa = []
    rs_rpa = []
    rs_rn = []
    rs_sn = []
    rs_dr = []
    rs_ppo = []

    policy_crpa = Policy(all_args, eval_envs.observation_space, eval_envs.action_space, device)
    policy_crpa.prep_rollout()
    policy_rpa = Policy(all_args, eval_envs.observation_space, eval_envs.action_space, device)
    policy_rpa.prep_rollout()
    policy_rn = Policy(all_args, eval_envs.observation_space, eval_envs.action_space, device)
    policy_rn.prep_rollout()
    policy_sn = Policy(all_args, eval_envs.observation_space, eval_envs.action_space, device)
    policy_sn.prep_rollout()
    policy_dr = Policy(all_args, eval_envs.observation_space, eval_envs.action_space, device)
    policy_dr.prep_rollout()
    policy_ppo = Policy(all_args, eval_envs.observation_space, eval_envs.action_space, device)
    policy_ppo.prep_rollout()

    policy_crpa.actor.load_state_dict(torch.load( \
        "/home/a/NeuralPlane_stable_V2/scripts/runs/2025-08-08_15-51-20_Control_control_F16_ppo_v1_crpa/episode_640/actor_latest.pt", map_location='cuda:0'))
    policy_rpa.actor.load_state_dict(torch.load( \
        "/home/a/NeuralPlane_stable_V1/scripts/runs/eval_runs/2025-06-26_15-11-03_Control_control_F16_ppo_v1_adv_noise/episode_580/actor_latest.pt", map_location='cuda:0'))
    policy_rn.actor.load_state_dict(torch.load( \
        "/home/a/NeuralPlane_stable_V1/scripts/runs/eval_runs/2025-07-02_16-57-12_Control_control_F16_ppo_v1_random_noise/episode_599/actor_latest.pt", map_location='cuda:0'))
    policy_sn.actor.load_state_dict(torch.load( \
        "/home/a/NeuralPlane_stable_V1/scripts/runs/2025-08-07_14-57-23_Control_control_F16_ppo_v1/episode_590/actor_latest.pt", map_location='cuda:0'))
    policy_dr.actor.load_state_dict(torch.load( \
        "/home/a/actor_latest.pt", map_location='cuda:0'))
    policy_ppo.actor.load_state_dict(torch.load( \
        "/home/a/NeuralPlane_stable_V1/scripts/runs/eval_runs/2025-07-02_14-36-18_Control_control_F16_ppo_v1_no_noise/episode_580/actor_latest.pt", map_location='cuda:0'))
    if noise_factor == 0:
        for i in range(-75, 76, 5):
            vars.append(i)
            eval_envs.gpu_vec_env.model.airspeed = i
            suc, reward = multi_eval(eval_envs, all_args, policy_crpa, 500)
            sucs_crpa.append(suc)
            rs_crpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rpa, 500)
            sucs_rpa.append(suc)
            rs_rpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rn, 500)
            sucs_rn.append(suc)
            rs_rn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_sn, 500)
            sucs_sn.append(suc)
            rs_sn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_dr, 500)
            sucs_dr.append(suc)
            rs_dr.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_ppo, 500)
            sucs_ppo.append(suc)
            rs_ppo.append(reward)
        eval_envs.gpu_vec_env.model.airspeed = 0
    elif noise_factor == 1:
        for i in range(600, 701, 5):
            vars.append(i)
            eval_envs.gpu_vec_env.model.dynamics.m = i
            suc, reward = multi_eval(eval_envs, all_args, policy_crpa, 500)
            sucs_crpa.append(suc)
            rs_crpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rpa, 500)
            sucs_rpa.append(suc)
            rs_rpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rn, 500)
            sucs_rn.append(suc)
            rs_rn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_sn, 500)
            sucs_sn.append(suc)
            rs_sn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_dr, 500)
            sucs_dr.append(suc)
            rs_dr.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_ppo, 500)
            sucs_ppo.append(suc)
            rs_ppo.append(reward)
        eval_envs.gpu_vec_env.model.dynamics.m = 636.94
    elif noise_factor == 2:
        eval_envs.gpu_vec_env.task.add_noise_type = 'IMU_noise'
        for i in np.arange(0., 1.01, 0.05):
            vars.append(i)
            eval_envs.gpu_vec_env.task.IMU_noise_scale = i
            suc, reward = multi_eval(eval_envs, all_args, policy_crpa, 500)
            sucs_crpa.append(suc)
            rs_crpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rpa, 500)
            sucs_rpa.append(suc)
            rs_rpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rn, 500)
            sucs_rn.append(suc)
            rs_rn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_sn, 500)
            sucs_sn.append(suc)
            rs_sn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_dr, 500)
            sucs_dr.append(suc)
            rs_dr.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_ppo, 500)
            sucs_ppo.append(suc)
            rs_ppo.append(reward)
        eval_envs.gpu_vec_env.task.add_noise_type = 'no_noise'
    elif noise_factor == 3:
        eval_envs.gpu_vec_env.task.add_noise_type = 'AOAAOS_noise'
        for i in range(0., 0.31, 0.03):
            vars.append(i)
            eval_envs.gpu_vec_env.task.AOAAOS_noise_scale = i
            suc, reward = multi_eval(eval_envs, all_args, policy_crpa, 500)
            sucs_crpa.append(suc)
            rs_crpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rpa, 500)
            sucs_rpa.append(suc)
            rs_rpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rn, 500)
            sucs_rn.append(suc)
            rs_rn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_sn, 500)
            sucs_sn.append(suc)
            rs_sn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_dr, 500)
            sucs_dr.append(suc)
            rs_dr.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_ppo, 500)
            sucs_ppo.append(suc)
            rs_ppo.append(reward)
        eval_envs.gpu_vec_env.task.add_noise_type = 'no_noise'
    elif noise_factor == 4:
        eval_envs.gpu_vec_env.task.add_noise_type = 'EAS_noise'
        for i in range(0., 0.51, 0.05):
            vars.append(i)
            eval_envs.gpu_vec_env.task.EAS_noise_scale = i
            suc, reward = multi_eval(eval_envs, all_args, policy_crpa, 500)
            sucs_crpa.append(suc)
            rs_crpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rpa, 500)
            sucs_rpa.append(suc)
            rs_rpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rn, 500)
            sucs_rn.append(suc)
            rs_rn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_sn, 500)
            sucs_sn.append(suc)
            rs_sn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_dr, 500)
            sucs_dr.append(suc)
            rs_dr.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_ppo, 500)
            sucs_ppo.append(suc)
            rs_ppo.append(reward)
        eval_envs.gpu_vec_env.task.add_noise_type = 'no_noise'
    elif noise_factor == 5:
        eval_envs.gpu_vec_env.task.add_noise_type = 'altitude_noise'
        for i in range(0, 51, 5):
            vars.append(i)
            eval_envs.gpu_vec_env.task.altitude_noise_scale = i
            suc, reward = multi_eval(eval_envs, all_args, policy_crpa, 500)
            sucs_crpa.append(suc)
            rs_crpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rpa, 500)
            sucs_rpa.append(suc)
            rs_rpa.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_rn, 500)
            sucs_rn.append(suc)
            rs_rn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_sn, 500)
            sucs_sn.append(suc)
            rs_sn.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_dr, 500)
            sucs_dr.append(suc)
            rs_dr.append(reward)
            suc, reward = multi_eval(eval_envs, all_args, policy_ppo, 500)
            sucs_ppo.append(suc)
            rs_ppo.append(reward)
        eval_envs.gpu_vec_env.task.add_noise_type = 'no_noise'
    print('vars=',vars)
    print('sucs_crpa=',sucs_crpa)
    print('sucs_rpa=',sucs_rpa)
    print('sucs_rn=',sucs_rn)
    print('sucs_sn=',sucs_sn)
    print('sucs_dr=',sucs_dr)
    print('sucs_ppo=',sucs_ppo)
    print('rs_crpa=',rs_crpa)
    print('rs_rpa=',rs_rpa)
    print('rs_rn=',rs_rn)
    print('rs_sn=',rs_sn)
    print('rs_dr=',rs_dr)
    print('rs_ppo=',rs_ppo)

    eval_envs.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])