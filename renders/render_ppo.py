import os
import sys
import numpy as np
import torch
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.control_env import ControlEnv
from envs.planning_env import PlanningEnv
from envs.env_wrappers import GPUVecEnv
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.DEBUG)

CURRENT_WORK_PATH = os.getcwd()

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = True
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cuda:0'))
        self.use_prior = False
    
def _t2n(x):
    return x.detach().cpu().numpy()

success_arr = []
for i in range(1):
    max_sc = 0.0
    max_sc_epi_id = 0
    epi_num = i * 10
    episode_rewards = 0
    ego_run_dir = "/home/a/demo/NeuralPlane_stable_V1/scripts/runs/2026-01-05_16-08-39_Control_heading_F16_ppo_v1/episode_250"
    ego_run_dir = "/home/a/NeuralPlane/scripts/runs/2025-11-21_20-09-18_Control_heading_F16_ppo_v1/episode_210"
    ego_run_dir = "/home/a/demo/NeuralPlane_stable_V1/scripts/runs/2026-01-27_00-49-30_Control_rc_HYBRID_ppo_v1/episode_370/"
    ego_run_dir = "/home/a/demo/NeuralPlane_stable_V1/scripts/runs/2026-01-28_00-01-05_Control_rc_HYBRID_ppo_v1/episode_300/"
    ego_run_dir = "/home/a/demo/NeuralPlane_stable_V1/scripts/runs/2026-01-28_15-39-15_Control_rc_HYBRID_ppo_v1/episode_260/"
    ego_run_dir = "/home/a/demo/NeuralPlane_stable_V1/scripts/runs/2026-02-01_17-05-09_Control_rc_HYBRID_ppo_v1/episode_300/"
    ego_run_dir = "/home/a/demo/NeuralPlane_stable_V1/scripts/runs/2026-02-02_17-22-22_Control_rc_HYBRID_ppo_v1/episode_320/"
    ego_run_dir = "/home/a/demo/NeuralPlane_stable_V1/scripts/runs/2026-03-06_16-11-43_Control_tracking_F16_ppo_v1/episode_360/"
    ego_run_dir = "/home/a/demo/NeuralPlane_stable_V1/scripts/runs/2026-03-09_21-31-43_Control_tracking_F16_ppo_v1/episode_160"

    device = "cuda:0"
    config = "tracking"

    env = ControlEnv(num_envs=1, config=config, model='F16', random_seed=5, device=device)
    env.model.airspeed = 0
    args = Args()

    ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device(device))
    ego_policy.eval()
    ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_latest.ckpt"))

    # env.task.target_x_acc = 2 * (torch.rand(size, device=self.device) - 0.5) * 2.5
    # env.task.target_z_acc = 2 * (torch.rand(size, device=self.device) - 0.5) * 2
    # env.task.target_heading = 2 * (torch.rand(size, device=self.device) - 0.5) * torch.pi
    # env.task.target_x_acc = 0
    # env.task.target_z_acc = 0
    # env.task.target_heading = 0

    print("Start render")
    ego_obs = env.reset()
    
    # 状态量
    npos, epos, altitude = env.model.get_position()
    npos_buf = np.mean(_t2n(npos))
    epos_buf = np.mean(_t2n(epos))
    altitude_buf = np.mean(_t2n(altitude))

    roll, pitch, yaw = env.model.get_posture()
    roll_buf = np.mean(_t2n(roll))
    pitch_buf = np.mean(_t2n(pitch))
    yaw_buf = np.mean(_t2n(yaw))

    ax, ay, az = env.model.get_acceleration()
    ax_buf = np.mean(_t2n(ax))
    ay_buf = np.mean(_t2n(ay))
    az_buf = np.mean(_t2n(az))

    vt = env.model.get_vt()
    vt_buf = np.mean(_t2n(vt))

    alpha = env.model.get_AOA()
    alpha_buf = np.mean(_t2n(alpha))

    beta = env.model.get_AOS()
    beta_buf = np.mean(_t2n(beta))

    G = env.model.get_G()
    G_buf = np.mean(_t2n(G))
    # 控制量
    T = env.model.get_thrust()
    T_buf = np.mean(_t2n(T))
    throttle_buf = np.mean(_t2n(T * 0.3048 / 82339 / 0.225))

    el, ail, rud, lef = env.model.get_control_surface()
    el_buf = np.mean(_t2n(el))
    ail_buf = np.mean(_t2n(ail))
    rud_buf = np.mean(_t2n(rud))
    lef_buf = np.mean(_t2n(lef))
    # 目标
    if config == 'heading':
        target_altitude_buf = np.mean(_t2n(env.task.target_altitude))
        target_heading_buf = np.mean(_t2n(env.task.target_heading))
        target_vt_buf = np.mean(_t2n(env.task.target_vt))
    elif config == 'control':
        target_pitch_buf = np.mean(_t2n(env.task.target_pitch))
        target_heading_buf = np.mean(_t2n(env.task.target_heading))
        target_vt_buf = np.mean(_t2n(env.task.target_vt))
    elif config == 'tracking':
        target_npos_buf = np.mean(_t2n(env.task.target_npos))
        target_epos_buf = np.mean(_t2n(env.task.target_epos))
        target_altitude_buf = np.mean(_t2n(env.task.target_altitude))
    elif config == 'rc':
        target_xacc_buf = np.mean(_t2n(env.task.target_vx))
        target_zacc_buf = np.mean(_t2n(env.task.target_vz))
        target_heading_buf = np.mean(_t2n(env.task.target_heading))

    counts = 0
    env.render(count=counts)
    ego_rnn_states = torch.zeros((1, 1, 128), device=torch.device(device))
    masks = torch.ones((1, 1), device=torch.device(device))
    start = time.time()
    unreach_target = 0
    reset_target = 0
    while True:
        # env.task.target_x_acc = 0
        # env.task.target_z_acc = 0
        # env.task.target_heading = 0
        with torch.no_grad():
            ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        # print(ego_actions)
        # Obser reward and next obs
            ego_obs, rewards, dones, bad_dones, exceed_time_limits, infos = env.step(ego_actions, render=True, count=counts)
        unreach_target += int(_t2n(bad_dones))
        reset_target += int(_t2n(dones))

        npos, epos, altitude = env.model.get_position()
        npos_buf = np.hstack((npos_buf, np.mean(_t2n(npos))))
        epos_buf = np.hstack((epos_buf, np.mean(_t2n(epos))))
        altitude_buf = np.hstack((altitude_buf, np.mean(_t2n(altitude))))

        roll, pitch, yaw = env.model.get_posture()
        roll_buf = np.hstack((roll_buf, np.mean(_t2n(roll))))
        pitch_buf = np.hstack((pitch_buf, np.mean(_t2n(pitch))))
        yaw_buf = np.hstack((yaw_buf, np.mean(_t2n(yaw))))

        ax, ay, az = env.model.get_acceleration()
        ax_buf = np.hstack((ax_buf, np.mean(_t2n(ax))))
        ay_buf = np.hstack((ay_buf, np.mean(_t2n(ay))))
        az_buf = np.hstack((az_buf, np.mean(_t2n(az))))

        vt = env.model.get_vt()
        vt_buf = np.hstack((vt_buf, np.mean(_t2n(vt))))

        alpha = env.model.get_AOA()
        alpha_buf = np.hstack((alpha_buf, np.mean(_t2n(alpha))))

        beta = env.model.get_AOS()
        beta_buf = np.hstack((beta_buf, np.mean(_t2n(beta))))

        G = env.model.get_G()
        G_buf = np.hstack((G_buf, np.mean(_t2n(G))))

        T = env.model.get_thrust()
        T_buf = np.hstack((T_buf, np.mean(_t2n(T))))
        throttle_buf = np.hstack((throttle_buf, np.mean(_t2n(T * 0.3048 / 82339 / 0.225))))

        el, ail, rud, lef = env.model.get_control_surface()
        el_buf = np.hstack((el_buf, np.mean(_t2n(el))))
        ail_buf = np.hstack((ail_buf, np.mean(_t2n(ail))))
        rud_buf = np.hstack((rud_buf, np.mean(_t2n(rud))))

        if config == 'heading':
            target_altitude_buf = np.hstack((target_altitude_buf, np.mean(_t2n(env.task.target_altitude))))
            target_heading_buf = np.hstack((target_heading_buf, np.mean(_t2n(env.task.target_heading))))
            target_vt_buf = np.hstack((target_vt_buf, np.mean(_t2n(env.task.target_vt))))
        elif config == 'control':
            target_pitch_buf = np.hstack((target_pitch_buf, np.mean(_t2n(env.task.target_pitch))))
            target_heading_buf = np.hstack((target_heading_buf, np.mean(_t2n(env.task.target_heading))))
            target_vt_buf = np.hstack((target_vt_buf, np.mean(_t2n(env.task.target_vt))))
        elif config == 'tracking':
            target_npos_buf = np.hstack((target_npos_buf, np.mean(_t2n(env.task.target_npos))))
            target_epos_buf = np.hstack((target_epos_buf, np.mean(_t2n(env.task.target_epos))))
            target_altitude_buf = np.hstack((target_altitude_buf, np.mean(_t2n(env.task.target_altitude))))
        elif config == 'rc':
            target_xacc_buf = np.hstack((target_xacc_buf, np.mean(_t2n(env.task.target_vx))))
            target_zacc_buf = np.hstack((target_zacc_buf, np.mean(_t2n(env.task.target_vz))))
            target_heading_buf = np.hstack((target_heading_buf, np.mean(_t2n(env.task.target_heading))))

        counts += 1
        print(counts, _t2n(rewards))
        episode_rewards += _t2n(rewards)
        if counts >= 10000:
            break
    # save result
    np.save('./result/npos.npy', npos_buf)
    np.save('./result/epos.npy', epos_buf)
    np.save('./result/altitude.npy', altitude_buf)
    np.save('./result/roll.npy', roll_buf)
    np.save('./result/pitch.npy', pitch_buf)
    np.save('./result/yaw.npy', yaw_buf)
    np.save('./result/ax.npy', ax_buf)
    np.save('./result/ay.npy', ay_buf)
    np.save('./result/az.npy', az_buf)
    np.save('./result/vt.npy', vt_buf)
    np.save('./result/alpha.npy', alpha_buf)
    np.save('./result/beta.npy', beta_buf)
    np.save('./result/G.npy', G_buf)

    np.save('./result/T.npy', T_buf)
    np.save('./result/throttle.npy', throttle_buf)
    np.save('./result/ail.npy', ail_buf)
    np.save('./result/el.npy', el_buf)
    np.save('./result/rud.npy', rud_buf)

    if config == 'heading':
        np.save('./result/target_altitude.npy', target_altitude_buf)
        np.save('./result/target_heading.npy', target_heading_buf)
        np.save('./result/target_vt.npy', target_vt_buf)
    elif config == 'control':
        np.save('./result/target_pitch.npy', target_pitch_buf)
        np.save('./result/target_heading.npy', target_heading_buf)
        np.save('./result/target_vt.npy', target_vt_buf)
    elif config == 'tracking':
        np.save('./result/target_npos.npy', target_npos_buf)
        np.save('./result/target_epos.npy', target_epos_buf)
        np.save('./result/target_altitude.npy', target_altitude_buf)
    elif config == 'rc':
        np.save('./result/target_xacc_buf', target_xacc_buf)
        np.save('./result/target_zacc_buf', target_zacc_buf)
        np.save('./result/target_heading_buf', target_heading_buf)
    end = time.time()
    print('total time:', end - start)
    print('episode reward:', episode_rewards)
    print('average episode reward:', episode_rewards / (unreach_target + reset_target))
    print('unreach target:', unreach_target)
    print('reset target:', reset_target)
    print('success rate:', reset_target / (reset_target + unreach_target))
    success_arr.append(reset_target / (reset_target + unreach_target))
    if(reset_target / (reset_target + unreach_target) > max_sc):
        max_sc = reset_target / (reset_target + unreach_target)
        max_sc_epi_id = epi_num
    print('max success rate:', max_sc)
    print('max success rate index:', max_sc_epi_id)

print(success_arr)