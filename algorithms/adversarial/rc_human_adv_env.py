import gym
import numpy as np
import torch

from envs.utils.utils import wrap_PI


class RCHumanAdversarialEnv:
    """Adversarial wrapper for rc_human.

    The adversary acts in three bounded spaces:
      1. command space: local vx/vy/vz and yaw-rate commands;
      2. observation space: normalized policy-observation perturbation;
      3. wind space: N/E/D gust commands.

    During adversary training, the task's sensor noise and Dryden random wind
    source are disabled.  Command generation can either be controlled by the
    adversary or left to the original rc_human random command generator.

    The victim policy remains fixed.  This wrapper exposes an RL problem whose
    reward is large when the victim tracks poorly, enters unsafe attitude/rate
    regions, or terminates badly, while penalizing large/jerky perturbations.
    """

    def __init__(self, env, victim_actor, args, device):
        self.env = env
        self.victim_actor = victim_actor
        self.args = args
        self.device = device
        self.n = env.n
        self.num_agents = 1
        self.obs_dim = int(env.observation_space.shape[0])
        self.command_dim = 4
        self.wind_dim = 3
        self.adv_action_dim = self.command_dim + self.obs_dim + self.wind_dim
        self.allowed_obs_attack_idx = torch.tensor(
            [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24],
            dtype=torch.long,
            device=device,
        )

        self.observation_space = env.observation_space
        self.task = env.task
        self.model = env.model
        self._disable_stochastic_sources()
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.adv_action_dim, dtype=np.float32),
            high=np.ones(self.adv_action_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.command_scale = torch.tensor(
            [
                float(getattr(env.config, "rc_human_vx_limit", 1.0)),
                float(getattr(env.config, "rc_human_vy_limit", 1.0)),
                float(getattr(env.config, "rc_human_vz_limit", 1.0)),
                float(getattr(env.config, "rc_human_yaw_rate_limit", 0.6)),
            ],
            device=device,
        ) * float(args.adv_command_frac)

        self.obs_scale = self._build_obs_scale() * float(args.adv_obs_frac)
        self.obs_attack_mask = torch.zeros(self.obs_dim, device=device)
        self.obs_attack_mask[self.allowed_obs_attack_idx] = 1.0
        self.obs_scale = self.obs_scale * self.obs_attack_mask
        self.obs_energy_sigma = torch.clamp(self.obs_scale, min=1e-6)
        self.wind_scale = self._build_wind_scale() * float(args.adv_wind_frac)

        self.adv_command = torch.zeros((self.n, self.command_dim), device=device)
        self.adv_obs = torch.zeros((self.n, self.obs_dim), device=device)
        self.obs_energy_window = max(1, int(args.adv_obs_energy_window))
        self.obs_energy_budget = float(args.adv_obs_energy_budget)
        self.obs_energy_penalty = torch.zeros(self.n, device=device)
        self.obs_energy_buffer = torch.zeros((self.obs_energy_window, self.n), device=device)
        self.obs_energy_ptr = 0
        self.adv_wind = torch.zeros((self.n, self.wind_dim), device=device)
        self.prev_adv_action = torch.zeros((self.n, self.adv_action_dim), device=device)
        self.victim_rnn = torch.zeros(
            (self.n, args.victim_recurrent_hidden_layers, args.victim_recurrent_hidden_size),
            device=device,
        )
        self.victim_masks = torch.ones((self.n, 1), device=device)
        self.current_obs = None
        self.last_info = {}

    @property
    def agents(self):
        return self.num_agents

    def reset(self):
        obs = self.env.reset()
        reset = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self._reset_adv_state(reset)
        if self._use_random_command_base():
            self._enable_random_command_if_needed(reset)
            self.env.task.sync_command(self.env)
        else:
            self._set_neutral_command(reset)
            self._freeze_task_sync()
        obs = self.env.obs()
        self.current_obs = obs
        return obs

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def _reset_adv_state(self, mask):
        self.adv_command[mask] = 0.0
        self.adv_obs[mask] = 0.0
        self.obs_energy_penalty[mask] = 0.0
        self.obs_energy_buffer[:, mask] = 0.0
        self.adv_wind[mask] = 0.0
        self.prev_adv_action[mask] = 0.0
        self.victim_rnn[mask] = 0.0
        self.victim_masks[mask] = 1.0

    def _use_random_command_base(self):
        return bool(self.args.adv_use_random_command) or bool(
            getattr(self.args, "adv_command_random_base", False)
        )

    def _disable_stochastic_sources(self):
        """Make observation noise and wind fully controlled by the adversary."""
        cfg = self.env.config
        cfg.enable_sensor_noise = False
        cfg.enable_dryden_turbulence = False
        cfg.dryden_randomize = False
        cfg.dryden_domain_randomization = False
        cfg.dryden_sigma_curriculum_enable = False
        cfg.dryden_mean_wind_curriculum_enable = False

        self.env.wind_disturbance = None
        if hasattr(self.task, "enable_sensor_noise"):
            self.task.enable_sensor_noise = False
        if hasattr(self.task, "stick_noise_std"):
            self.task.stick_noise_std = 0.0
        if hasattr(self.task, "curriculum_enable") and not self._use_random_command_base():
            self.task.curriculum_enable = False

        if hasattr(self.model, "base_wind_ned"):
            self.model.base_wind_ned.zero_()
        if hasattr(self.model, "set_wind_ned"):
            zeros = torch.zeros(self.n, device=self.device)
            self.model.set_wind_ned(zeros, zeros, zeros)

    def _freeze_task_sync(self):
        if hasattr(self.task, "last_synced_step"):
            self.task.last_synced_step[:] = self.env.step_count.long()

    def _set_neutral_command(self, mask):
        if int(torch.sum(mask).item()) == 0:
            return
        _roll, _pitch, heading = self.model.get_posture()
        self.task.target_vx[mask] = 0.0
        self.task.target_vy[mask] = 0.0
        self.task.target_vz[mask] = 0.0
        self.task.target_yaw_rate[mask] = 0.0
        self.task.target_heading[mask] = heading[mask]
        self.task.target_vn[mask] = 0.0
        self.task.target_ve[mask] = 0.0
        if hasattr(self.task, "raw_vx"):
            self.task.raw_vx[mask] = 0.0
            self.task.raw_vy[mask] = 0.0
            self.task.raw_vz[mask] = 0.0
            self.task.raw_yaw[mask] = 0.0
            self.task.stick_vx[mask] = 0.0
            self.task.stick_vy[mask] = 0.0
            self.task.stick_vz[mask] = 0.0
            self.task.stick_yaw[mask] = 0.0
            self.task.dwell_left[mask] = 10**9

    def _enable_random_command_if_needed(self, mask):
        if not bool(self.args.adv_use_random_command):
            return
        if hasattr(self.task, "curriculum_enable"):
            self.task.curriculum_enable = True
        if hasattr(self.task, "dwell_left"):
            self.task.dwell_left[mask] = 0

    def _build_obs_scale(self):
        cfg = self.env.config
        sensor_vel = float(getattr(cfg, "sensor_vel_std", 0.05))
        sensor_pos = float(getattr(cfg, "sensor_pos_std", 1.0))
        sensor_att = float(getattr(cfg, "sensor_att_std", 0.005))
        sensor_omega = float(getattr(cfg, "sensor_omega_std", 0.0005))

        sx = torch.ones(self.obs_dim, device=self.device) * float(self.args.adv_obs_default_scale)
        # rc_human observation layout in envs/tasks/rc_human_task.py.
        sx[0:3] = sensor_vel
        sx[3] = sensor_att / torch.pi
        sx[4:8] = 0.03
        sx[8] = sensor_pos / 100.0
        sx[9:13] = sensor_att
        sx[13:18] = sensor_vel / 5.0
        sx[18:22] = 0.02
        sx[22:25] = sensor_omega
        sx[25:30] = 0.02
        return torch.clamp(sx, max=float(self.args.adv_obs_max_scale))

    def _build_wind_scale(self):
        cfg = self.env.config
        n = max(abs(float(getattr(cfg, "dryden_mean_wind_north_min", -1.0))),
                abs(float(getattr(cfg, "dryden_mean_wind_north_max", 1.0))))
        e = max(abs(float(getattr(cfg, "dryden_mean_wind_east_min", -1.0))),
                abs(float(getattr(cfg, "dryden_mean_wind_east_max", 1.0))))
        d = max(abs(float(getattr(cfg, "dryden_mean_wind_down_min", -0.15))),
                abs(float(getattr(cfg, "dryden_mean_wind_down_max", 0.15))))
        return torch.tensor([n, e, d], device=self.device)

    def _split_adv_action(self, adv_action):
        adv_action = torch.clamp(adv_action, -1.0, 1.0)
        c_end = self.command_dim
        o_end = c_end + self.obs_dim
        cmd = adv_action[:, :c_end] * self.command_scale
        obs = adv_action[:, c_end:o_end] * self.obs_scale
        wind = adv_action[:, o_end:] * self.wind_scale
        return cmd, obs, wind

    def _lowpass_adv(self, new, old, alpha):
        return (1.0 - alpha) * old + alpha * new

    def _apply_command_attack(self, cmd, update_filter=True):
        if bool(self.args.adv_use_random_command):
            self.env.task.sync_command(self.env)
            return
        task = self.env.task
        random_base = self._use_random_command_base()
        if random_base:
            task.sync_command(self.env)
            base_vx = task.target_vx.clone()
            base_vy = task.target_vy.clone()
            base_vz = task.target_vz.clone()
            base_yaw_rate = task.target_yaw_rate.clone()
            base_heading = task.target_heading.clone()

        a = float(self.args.adv_command_alpha)
        if update_filter:
            self.adv_command = self._lowpass_adv(cmd, self.adv_command, a)

        if random_base:
            target_vx = base_vx + self.adv_command[:, 0]
            target_vy = base_vy + self.adv_command[:, 1]
            target_vz = base_vz + self.adv_command[:, 2]
            target_yaw_rate = base_yaw_rate + self.adv_command[:, 3]
            target_heading = wrap_PI(base_heading + self.adv_command[:, 3] * task.dt)
        else:
            target_vx = self.adv_command[:, 0]
            target_vy = self.adv_command[:, 1]
            target_vz = self.adv_command[:, 2]
            target_yaw_rate = self.adv_command[:, 3]
            target_heading = wrap_PI(task.target_heading + self.adv_command[:, 3] * task.dt)

        task.target_vx = torch.clamp(
            target_vx,
            -task.vx_limit,
            task.vx_limit,
        )
        task.target_vy = torch.clamp(
            target_vy,
            -task.vy_limit,
            task.vy_limit,
        )
        task.target_vz = torch.clamp(
            target_vz,
            -task.vz_limit,
            task.vz_limit,
        )
        task.target_yaw_rate = torch.clamp(
            target_yaw_rate,
            -task.yaw_rate_limit,
            task.yaw_rate_limit,
        )
        task.target_heading = target_heading
        task.target_vn, task.target_ve = task.local_to_ground_velocity(
            task.target_vx, task.target_vy, task.target_heading
        )
        self._freeze_task_sync()

    def _apply_wind_attack(self, wind):
        if not hasattr(self.env.model, "set_wind_gust_ned"):
            return
        if not bool(getattr(self.env.model, "wind_enabled", False)):
            return

        a = float(self.args.adv_wind_alpha)
        self.adv_wind = self._lowpass_adv(wind, self.adv_wind, a)
        self.env.model.set_wind_ned(
            self.adv_wind[:, 0], self.adv_wind[:, 1], self.adv_wind[:, 2]
        )

    def _update_obs_energy(self):
        z = (self.adv_obs / self.obs_energy_sigma) * self.obs_attack_mask
        # Mean normalized energy over attackable channels.  For standard
        # Gaussian sensor noise this is O(1) per step, so the window budget can
        # be interpreted as average normal-noise energy over the window.
        count = float(self.allowed_obs_attack_idx.numel())
        step_energy = torch.sum(z * z, dim=1) / max(count, 1.0)
        self.obs_energy_buffer[self.obs_energy_ptr] = step_energy.detach()
        self.obs_energy_ptr = (self.obs_energy_ptr + 1) % self.obs_energy_window
        window_energy = torch.sum(self.obs_energy_buffer, dim=0)
        excess = torch.relu(window_energy - self.obs_energy_budget)
        self.obs_energy_penalty = excess * excess
        return step_energy, window_energy

    def _perturb_obs(self, obs):
        return obs + self.adv_obs * self.obs_attack_mask

    def _aux_metrics(self):
        task = self.env.task
        model = self.env.model
        roll, pitch, heading = model.get_posture()
        p, q, r = model.get_angular_velocity()
        vx_n, vy_e = model.get_ground_speed()
        vz = model.get_climb_rate()
        local_vx, local_vy = task.ground_to_local_velocity(vx_n, vy_e, heading)

        vel_err = torch.sqrt(
            (local_vx - task.target_vx) ** 2
            + (local_vy - task.target_vy) ** 2
            + (vz - task.target_vz) ** 2
        )
        yaw_err = torch.abs(wrap_PI(heading - task.target_heading))
        att = torch.sqrt(roll * roll + pitch * pitch)
        omega = torch.sqrt(p * p + q * q + r * r)
        f = model.get_control()
        force_margin = torch.relu(f - model.max_F + 0.1).sum(dim=1) + torch.relu(0.1 - f).sum(dim=1)
        return vel_err, yaw_err, att, omega, force_margin

    def _adv_reward(self, policy_reward, done, bad_done, exceed, adv_action):
        adv_action_bounded = torch.clamp(adv_action, -1.0, 1.0)
        adv_action_for_reg = adv_action_bounded.clone()
        prev_for_reg = self.prev_adv_action
        if bool(self.args.adv_use_random_command):
            adv_action_for_reg[:, :self.command_dim] = 0.0
            prev_for_reg = prev_for_reg.clone()
            prev_for_reg[:, :self.command_dim] = 0.0
        vel_err, yaw_err, att, omega, force_margin = self._aux_metrics()
        alive = (~(done | bad_done | exceed)).float()
        near_bad_att = torch.relu(att - float(self.args.adv_attitude_safe_rad))
        near_bad_omega = torch.relu(omega - float(self.args.adv_omega_safe_rad))
        jerk = torch.mean(torch.abs(adv_action_for_reg - prev_for_reg), dim=1)
        linf = torch.max(torch.abs(adv_action_for_reg), dim=1).values
        saturation = (torch.abs(adv_action_for_reg) > 0.98).float().mean(dim=1)
        raw_excess = torch.relu(torch.abs(adv_action) - 1.0).mean(dim=1)
        obs_step_energy, obs_window_energy = self._update_obs_energy()

        horizon_loss = -policy_reward.detach()
        aux = (
            float(self.args.adv_w_vel_error) * vel_err.detach()
            + float(self.args.adv_w_yaw_error) * yaw_err.detach()
            + float(self.args.adv_w_attitude) * near_bad_att.detach()
            + float(self.args.adv_w_omega) * near_bad_omega.detach()
            + float(self.args.adv_w_force_margin) * force_margin.detach()
        )
        terminal = float(self.args.adv_bad_done_bonus) * bad_done.float()
        alive_penalty = -float(self.args.adv_alive_penalty) * alive
        reg = (
            float(self.args.adv_linf_penalty) * linf
            + float(self.args.adv_smooth_penalty) * jerk
            + float(self.args.adv_raw_excess_penalty) * raw_excess
            + float(self.args.adv_obs_energy_penalty) * self.obs_energy_penalty
        )
        reward = alive_penalty + float(self.args.adv_policy_reward_weight) * horizon_loss + aux + terminal - reg

        self.last_info = {
            "adv/vel_error": vel_err.detach().mean().item(),
            "adv/yaw_error": yaw_err.detach().mean().item(),
            "adv/attitude": att.detach().mean().item(),
            "adv/omega": omega.detach().mean().item(),
            "adv/linf": linf.detach().mean().item(),
            "adv/saturation_frac": saturation.detach().mean().item(),
            "adv/raw_excess": raw_excess.detach().mean().item(),
            "adv/jerk": jerk.detach().mean().item(),
            "adv/obs_step_energy": obs_step_energy.detach().mean().item(),
            "adv/obs_window_energy": obs_window_energy.detach().mean().item(),
            "adv/obs_energy_ratio": (obs_window_energy / max(self.obs_energy_budget, 1e-6)).detach().mean().item(),
            "adv/obs_energy_penalty": self.obs_energy_penalty.detach().mean().item(),
            "adv/bad_done_rate": bad_done.float().mean().item(),
            "adv/policy_reward": policy_reward.detach().mean().item(),
            "adv/command_abs": self.adv_command.detach().abs().mean().item(),
            "adv/obs_abs": (self.adv_obs * self.obs_attack_mask).detach().abs().mean().item(),
            "adv/obs_active_frac": ((self.adv_obs * self.obs_attack_mask).detach().abs() > 1e-8).float().mean().item(),
            "adv/wind_abs": self.adv_wind.detach().abs().mean().item(),
        }
        self.prev_adv_action = adv_action_bounded.detach().clone()
        return reward.reshape(-1, 1)

    def step(self, adv_action):
        adv_action = torch.as_tensor(adv_action, dtype=torch.float32, device=self.device)
        if adv_action.ndim == 3:
            adv_action = adv_action.reshape(self.n, -1)

        cmd, obs_delta, wind_delta = self._split_adv_action(adv_action)
        if bool(self.args.adv_use_random_command):
            cmd.zero_()
        self.adv_obs = self._lowpass_adv(obs_delta, self.adv_obs, float(self.args.adv_obs_alpha))
        self.adv_obs = self.adv_obs * self.obs_attack_mask

        reset_before = (
            self.env.is_done.bool()
            | self.env.bad_done.bool()
            | self.env.exceed_time_limit.bool()
        )
        self.env.reset()
        if torch.any(reset_before):
            if self._use_random_command_base():
                self._enable_random_command_if_needed(reset_before)
            else:
                self._set_neutral_command(reset_before)
        if self._use_random_command_base():
            self.env.task.sync_command(self.env)
        else:
            self._freeze_task_sync()
        self._apply_command_attack(cmd)
        self._apply_wind_attack(wind_delta)

        victim_obs = self._perturb_obs(self.env.obs())
        with torch.no_grad():
            victim_action, _, self.victim_rnn = self.victim_actor(
                victim_obs, self.victim_rnn, self.victim_masks,
                deterministic=bool(self.args.victim_deterministic),
            )

        self.env.model.update(victim_action)
        self.env.step_count += 1
        if bool(self.args.adv_use_random_command):
            self.env.task.sync_command(self.env)
        elif bool(getattr(self.args, "adv_command_random_base", False)):
            self._apply_command_attack(cmd, update_filter=False)
        else:
            self._freeze_task_sync()
        obs = self._perturb_obs(self.env.obs())
        done, bad_done, exceed, info = self.env.done(self.env.info())
        if not bool(self.args.adv_use_random_command):
            self._freeze_task_sync()
        policy_reward = self.env.reward()

        reset = done | bad_done | exceed
        if torch.any(reset):
            self._reset_adv_state(reset)
        self.victim_masks = (~reset).float().reshape(-1, 1)
        adv_reward = self._adv_reward(policy_reward, done, bad_done, exceed, adv_action)
        self.current_obs = obs
        return obs, adv_reward, done, bad_done, exceed, info
