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
    return suc_episodes / total_epi
        
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

    noise_factor = 0 #0:airspeed, 1:mass, 2:IMU, 3:alpha, 4:beta, 5:pitot, 6:alt 
    
    vars = []
    sucs = []
    vs = []
    policy = Policy(all_args, eval_envs.observation_space, eval_envs.action_space, device)
    policy.prep_rollout()

    policy.actor.load_state_dict(torch.load( \
    f"/home/a/demo/NeuralPlane_stable_V1/scripts/runs/2026-01-26_17-17-23_Control_rc_HYBRID_ppo_v1/episode_100/actor_latest.ckpt", map_location='cuda:0'))
    # vars.append(i)
    suc = multi_eval(eval_envs, all_args, policy, 1000)
    sucs.append(suc)

    # i = 150
    # while(i < 680):
    #     policy.actor.load_state_dict(torch.load( \
    #     f"/home/a/NeuralPlane/scripts/runs/2025-11-21_20-09-18_Control_heading_F16_ppo_v1/episode_{i}/actor_latest.pt", map_location='cuda:0'))
    #     vars.append(i)
    #     suc = multi_eval(eval_envs, all_args, policy, 1000)
    #     sucs.append(suc)
    #     vs.append(i)
    #     vs.append(suc)
    #     if(i == 670):
    #         i = 680
    #     else:
    #         i += 10

    # policy.actor.load_state_dict(torch.load( \
    #     "/home/a/NeuralPlane/scripts/runs/2025-07-04_17-15-16_Control_control_F16_ppo_v1/episode_690/actor_latest.pt"))
    # if noise_factor == 0:
    #     for i in range(-100, 101, 2):
    #         vars.append(i)
    #         eval_envs.gpu_vec_env.model.airspeed = 0
    #         suc = multi_eval(eval_envs, all_args, policy, 500)
    #         sucs.append(suc)
    # elif noise_factor == 1:
    #     for i in range(600, 800, 2):
    #         vars.append(i)
    #         eval_envs.gpu_vec_env.model.dynamics.m = i
    #         suc = multi_eval(eval_envs, all_args, policy, 1000)
    #         sucs.append(suc)
    
    # print('vars===',vars)
    print('sucs===',sucs)
    # print('vs===',vs)

    eval_envs.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])