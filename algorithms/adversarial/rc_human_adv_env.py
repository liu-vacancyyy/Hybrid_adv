import gym
import numpy as np
import torch

from envs.utils.utils import wrap_PI
from envs.wind.dryden_turbulence import EnvDrydenTurbulence


class RCHumanAdversarialEnv:
    """Adversarial wrapper for rc_human.

    The adversary acts in three bounded spaces:
      1. command space: raw PX4 sticks for vx/vy/vz/yaw;
      2. observation space: normalized policy-observation perturbation;
      3. wind space: Dryden-shaped N/E/D velocity and body p/q/r gust targets.

    During adversary training, the task's sensor noise and random Dryden noise
    source are disabled.  Wind still passes through a deterministic Dryden
    shaping filter driven by the adversary.  Command generation can either be
    controlled by the adversary or left to the original rc_human random command
    generator.  When the adversary controls commands, its raw sticks still pass
    through the rc_human PX4 VTOL-MC manual stick pipeline before reaching
    target_vx/vy/vz/yaw_rate.

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
        self.wind_velocity_dim = 3
        self.wind_pqr_dim = 3 if bool(getattr(env.config, "enable_dryden_angular_turbulence", True)) else 0
        self.wind_dim = self.wind_velocity_dim + self.wind_pqr_dim
        self.command_rate_limit_frac = float(getattr(args, "adv_command_rate_limit_frac", 0.1))
        self.obs_rate_limit_frac = float(getattr(args, "adv_obs_rate_limit_frac", 0.1))
        self.wind_rate_limit_frac = float(getattr(args, "adv_wind_rate_limit_frac", 0.1))
        self.allowed_obs_attack_idx = torch.tensor(
            [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24],
            dtype=torch.long,
            device=device,
        )
        self.obs_attack_dim = int(self.allowed_obs_attack_idx.numel())
        self.adv_action_dim = self.command_dim + self.obs_attack_dim + self.wind_dim

        self.observation_space = env.observation_space
        self.task = env.task
        self.model = env.model
        self.adv_wind_source = None
        self._disable_stochastic_sources()
        self._init_adv_wind_source()
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.adv_action_dim, dtype=np.float32),
            high=np.ones(self.adv_action_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.command_low, self.command_high = self._build_command_bounds()
        self.command_neg_scale = torch.abs(torch.minimum(
            self.command_low, torch.zeros_like(self.command_low)
        ))
        self.command_pos_scale = torch.maximum(
            self.command_high, torch.zeros_like(self.command_high)
        )
        self.command_scale = torch.maximum(
            self.command_neg_scale, self.command_pos_scale
        )

        self.obs_scale = self._build_obs_scale() * float(args.adv_obs_frac)
        self.obs_attack_mask = torch.zeros(self.obs_dim, device=device)
        self.obs_attack_mask[self.allowed_obs_attack_idx] = 1.0
        self.obs_scale = self.obs_scale * self.obs_attack_mask
        self.obs_action_scale = self.obs_scale[self.allowed_obs_attack_idx]
        self.obs_energy_sigma = torch.clamp(self.obs_scale, min=1e-6)
        self.wind_scale = self._build_wind_scale() * float(args.adv_wind_frac)
        self.command_norm_denom = torch.clamp(self.command_scale, min=1e-6)
        self.obs_norm_denom = torch.clamp(self.obs_scale, min=1e-6)
        self.wind_norm_denom = torch.clamp(self.wind_scale, min=1e-6)

        self.adv_command = torch.zeros((self.n, self.command_dim), device=device)
        self.adv_obs = torch.zeros((self.n, self.obs_dim), device=device)
        self.obs_energy_window = max(1, int(args.adv_obs_energy_window))
        self.obs_energy_budget = float(args.adv_obs_energy_budget)
        self.obs_energy_penalty = torch.zeros(self.n, device=device)
        self.obs_energy_buffer = torch.zeros((self.obs_energy_window, self.n), device=device)
        self.obs_energy_ptr = 0
        self.policy_reward_window = max(1, int(getattr(args, "adv_policy_reward_window", 10)))
        self.policy_reward_gamma = float(getattr(args, "adv_policy_reward_gamma", getattr(args, "gamma", 1.0)))
        self.policy_reward_buffer = torch.zeros((self.policy_reward_window, self.n), device=device)
        self.policy_reward_steps = torch.zeros(self.n, dtype=torch.long, device=device)
        self.policy_reward_ptr = 0
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
        self._reset_adv_wind_source(reset)
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
        self.policy_reward_buffer[:, mask] = 0.0
        self.policy_reward_steps[mask] = 0
        self.adv_wind[mask] = 0.0
        self.prev_adv_action[mask] = 0.0
        self.victim_rnn[mask] = 0.0
        self.victim_masks[mask] = 1.0

    def _init_adv_wind_source(self):
        if not hasattr(self.env.model, "set_wind_gust_ned"):
            return
        if not bool(getattr(self.env.model, "wind_enabled", False)):
            return
        self.adv_wind_source = EnvDrydenTurbulence(
            self.env.config,
            self.n,
            self.device,
            self.env.model.dt,
        )

    def _reset_adv_wind_source(self, mask):
        if self.adv_wind_source is None:
            return
        self.adv_wind_source.reset(self.env, mask)
        self._apply_adv_wind_output()

    def _apply_adv_wind_output(self):
        if self.adv_wind_source is None:
            return
        gust_ned = self.adv_wind_source.gust_ned
        gust_pqr = self.adv_wind_source.gust_pqr_body
        self.env.model.set_wind_gust_ned(
            gust_ned[:, 0],
            gust_ned[:, 1],
            gust_ned[:, 2],
            pqr_body=gust_pqr,
        )

    def _use_random_command_base(self):
        return bool(self.args.adv_use_random_command) or bool(
            getattr(self.args, "adv_command_random_base", False)
        )

    def _disable_stochastic_sources(self):
        """Make observation noise and Dryden excitation controlled by the adversary."""
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
        if hasattr(self.model, "set_wind_body_pqr"):
            zeros = torch.zeros(self.n, device=self.device)
            self.model.set_wind_body_pqr(zeros, zeros, zeros)

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
            if hasattr(self.task, "desired_raw_vx"):
                self.task.desired_raw_vx[mask] = 0.0
                self.task.desired_raw_vy[mask] = 0.0
                self.task.desired_raw_vz[mask] = 0.0
                self.task.desired_raw_yaw[mask] = 0.0
            self.task.stick_vx[mask] = 0.0
            self.task.stick_vy[mask] = 0.0
            self.task.stick_vz[mask] = 0.0
            self.task.stick_yaw[mask] = 0.0
            if hasattr(self.task, "vx_forward_limit"):
                self.task.vx_forward_limit[mask] = self.task.vx_max
            self.task.dwell_left[mask] = 10**9
            if hasattr(self.task, "command_transient_left"):
                self.task.command_transient_left[mask] = 0
            if hasattr(self.task, "mode5_release_state"):
                self.task.mode5_release_state[mask] = 0
                self.task.mode5_hold_elapsed[mask] = 0
                self.task.mode5_recovery_left[mask] = 0
                self.task.mode5_pre_release_raw[mask] = 0.0

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
        sx[0] = sensor_vel / max(float(getattr(cfg, "rc_human_vx_limit", 5.0)), 1e-6)
        sx[1] = sensor_vel / max(float(getattr(cfg, "rc_human_vy_limit", 1.0)), 1e-6)
        sx[2] = sensor_vel / max(float(getattr(cfg, "rc_human_vz_limit", 1.0)), 1e-6)
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
        values = [n, e, d]
        if self.wind_pqr_dim:
            values.extend([
                abs(float(getattr(cfg, "dryden_pqr_attack_p_max", 0.6))),
                abs(float(getattr(cfg, "dryden_pqr_attack_q_max", 0.6))),
                abs(float(getattr(cfg, "dryden_pqr_attack_r_max", 0.5))),
            ])
        return torch.tensor(values, device=self.device)

    def _build_command_bounds(self):
        frac = min(max(float(self.args.adv_command_frac), 0.0), 1.0)
        span = torch.ones(self.command_dim, device=self.device) * frac
        low = -span
        high = span
        return low, high

    def _scale_command_action(self, action):
        return torch.where(
            action >= 0.0,
            action * self.command_pos_scale.reshape(1, -1),
            action * self.command_neg_scale.reshape(1, -1),
        )

    def _split_adv_action(self, adv_action):
        adv_action = torch.clamp(adv_action, -1.0, 1.0)
        c_end = self.command_dim
        o_end = c_end + self.obs_attack_dim
        cmd = self._scale_command_action(adv_action[:, :c_end])
        obs_compact = adv_action[:, c_end:o_end] * self.obs_action_scale.reshape(1, -1)
        obs = torch.zeros((adv_action.shape[0], self.obs_dim), dtype=adv_action.dtype, device=self.device)
        obs[:, self.allowed_obs_attack_idx] = obs_compact
        wind = adv_action[:, o_end:] * self.wind_scale
        return cmd, obs, wind

    def _lowpass_adv(self, new, old, alpha):
        return (1.0 - alpha) * old + alpha * new

    def _rate_limit_adv(self, new, old, limit_frac, scale):
        limit_frac = float(limit_frac)
        if limit_frac <= 0.0:
            return new
        max_delta = torch.abs(scale).reshape(1, -1) * limit_frac
        return old + torch.clamp(new - old, -max_delta, max_delta)

    def _advance_random_base_raw(self):
        task = self.env.task
        steps = self.env.step_count.long()
        need_sync = steps > task.last_synced_step
        guard = 0
        while torch.any(need_sync):
            mask = need_sync
            task._update_mode5_release_state(self.env, mask)
            resample = mask & (task.dwell_left <= 0) & (task.mode5_release_state == 0)
            task._resample_raw_sticks(resample)
            if hasattr(task, "_apply_raw_stick_rate_limit"):
                task._apply_raw_stick_rate_limit(mask)
            task.dwell_left[mask] = torch.clamp(task.dwell_left[mask] - 1, min=0)
            task.command_transient_left[mask] = torch.clamp(
                task.command_transient_left[mask] - 1,
                min=0,
            )
            task.last_synced_step[mask] += 1
            guard += 1
            if guard > 4:
                task.last_synced_step[mask] = steps[mask]
            need_sync = steps > task.last_synced_step

        return torch.stack((
            task.raw_vx,
            task.raw_vy,
            task.raw_vz,
            task.raw_yaw,
        ), dim=1)

    def _apply_px4_stick_command(self, raw_stick, update_transient=True):
        task = self.env.task
        raw_stick = torch.clamp(raw_stick, -1.0, 1.0)
        if update_transient and hasattr(task, "command_transient_left"):
            prev_raw = torch.stack((
                task.raw_vx,
                task.raw_vy,
                task.raw_vz,
                task.raw_yaw,
            ), dim=1)
            delta_raw = torch.max(torch.abs(raw_stick - prev_raw), dim=1).values
            task.command_transient_left[:] = torch.clamp(
                task.command_transient_left - 1,
                min=0,
            )
            changed = delta_raw > task.command_transient_threshold
            if torch.any(changed):
                task.command_transient_left[changed] = max(
                    task.command_transient_grace_steps,
                    0,
                )

        task.raw_vx[:] = raw_stick[:, 0]
        task.raw_vy[:] = raw_stick[:, 1]
        task.raw_vz[:] = raw_stick[:, 2]
        task.raw_yaw[:] = raw_stick[:, 3]
        if hasattr(task, "desired_raw_vx"):
            task.desired_raw_vx[:] = raw_stick[:, 0]
            task.desired_raw_vy[:] = raw_stick[:, 1]
            task.desired_raw_vz[:] = raw_stick[:, 2]
            task.desired_raw_yaw[:] = raw_stick[:, 3]

        if hasattr(task, "vx_forward_limit") and not self._use_random_command_base():
            task.vx_forward_limit[:] = task.vx_max
        mask = torch.ones(task.n, dtype=torch.bool, device=task.device)
        task._apply_px4_vtol_mc_manual_sticks(mask, self.env)

    def _apply_command_attack(self, raw_stick, update_filter=True):
        if bool(self.args.adv_use_random_command):
            self.env.task.sync_command(self.env)
            return
        task = self.env.task
        random_base = self._use_random_command_base()
        if random_base:
            base_raw = self._advance_random_base_raw()
        else:
            base_raw = None

        a = float(self.args.adv_command_alpha)
        if update_filter:
            target = self._lowpass_adv(raw_stick, self.adv_command, a)
            self.adv_command = self._rate_limit_adv(
                target,
                self.adv_command,
                self.command_rate_limit_frac,
                self.command_scale,
            )

        if random_base:
            raw_stick = torch.clamp(base_raw + self.adv_command, -1.0, 1.0)
        else:
            raw_stick = self.adv_command
        self._apply_px4_stick_command(raw_stick, update_transient=not random_base)
        self._freeze_task_sync()

    def _apply_wind_attack(self, wind):
        if self.adv_wind_source is None:
            return

        a = float(self.args.adv_wind_alpha)
        target = self._lowpass_adv(wind, self.adv_wind, a)
        self.adv_wind = self._rate_limit_adv(
            target,
            self.adv_wind,
            self.wind_rate_limit_frac,
            self.wind_scale,
        )
        wind_velocity_target = self.adv_wind[:, :self.wind_velocity_dim]
        if self.wind_pqr_dim:
            wind_pqr_target = self.adv_wind[:, self.wind_velocity_dim:]
        else:
            wind_pqr_target = None
        self.adv_wind_source.step(
            excitation_ned=wind_velocity_target,
            excitation_pqr_body=wind_pqr_target,
        )
        self._apply_adv_wind_output()

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

        local_vx_error, local_vy_error = task.ground_to_local_velocity(
            vx_n - task.target_vn, vy_e - task.target_ve, heading
        )
        axis_vel_err = torch.stack((
            torch.abs(local_vx_error),
            torch.abs(local_vy_error),
            torch.abs(vz - task.target_vz),
        ), dim=1)
        axis_vel_err_max = torch.max(axis_vel_err, dim=1).values
        vel_err = torch.sqrt(torch.sum(axis_vel_err * axis_vel_err, dim=1))
        yaw_err = torch.abs(wrap_PI(heading - task.target_heading))
        att = torch.sqrt(roll * roll + pitch * pitch)
        omega = torch.sqrt(p * p + q * q + r * r)
        f = model.get_control()
        force_margin = torch.relu(f - model.max_F + 0.1).sum(dim=1) + torch.relu(0.1 - f).sum(dim=1)
        return (
            vel_err,
            axis_vel_err_max,
            yaw_err,
            att,
            omega,
            force_margin,
            torch.abs(local_vx_error),
            torch.abs(local_vy_error),
            torch.abs(vz - task.target_vz),
        )

    def _range_penalty(self, rms, low, _high):
        low = float(low)
        if low <= 0.0:
            return torch.zeros_like(rms)
        return torch.relu(low - rms) ** 2

    def _state_norm_stats(self, value, denom, mask=None):
        z = value / denom.reshape(1, -1)
        if mask is not None:
            active = mask > 0.0
            if int(torch.sum(active).item()) == 0:
                zero = torch.zeros(self.n, device=self.device)
                return zero, zero, zero
            z = z[:, active]
        abs_z = torch.abs(z)
        rms = torch.sqrt(torch.mean(z * z, dim=1))
        mean_abs = torch.mean(abs_z, dim=1)
        linf = torch.max(abs_z, dim=1).values
        return rms, mean_abs, linf

    def _action_jerk_stats(self, action, prev_action):
        c_end = self.command_dim
        o_end = c_end + self.obs_attack_dim
        cmd_delta = torch.abs(action[:, :c_end] - prev_action[:, :c_end])
        obs_delta = torch.abs(action[:, c_end:o_end] - prev_action[:, c_end:o_end])
        wind_delta = torch.abs(action[:, o_end:] - prev_action[:, o_end:])
        cmd_jerk = torch.mean(cmd_delta, dim=1)
        obs_jerk = torch.mean(obs_delta, dim=1)
        wind_jerk = torch.mean(wind_delta, dim=1)
        return cmd_jerk, obs_jerk, wind_jerk

    def _update_policy_reward_return(self, policy_reward):
        reward = policy_reward.detach().reshape(-1)
        self.policy_reward_buffer[self.policy_reward_ptr] = reward
        self.policy_reward_ptr = (self.policy_reward_ptr + 1) % self.policy_reward_window
        self.policy_reward_steps = torch.clamp(
            self.policy_reward_steps + 1,
            max=self.policy_reward_window,
        )

        proximal_return = torch.zeros(self.n, device=self.device)
        discount = 1.0
        for offset in range(self.policy_reward_window):
            valid = (self.policy_reward_steps > offset).float()
            idx = (self.policy_reward_ptr - 1 - offset) % self.policy_reward_window
            proximal_return = proximal_return + discount * self.policy_reward_buffer[idx] * valid
            discount *= self.policy_reward_gamma
        return proximal_return

    def _adv_reward(self, policy_reward, done, bad_done, exceed, adv_action):
        adv_action_bounded = torch.clamp(adv_action, -1.0, 1.0)
        adv_action_for_reg = adv_action_bounded.clone()
        prev_for_reg = self.prev_adv_action
        if bool(self.args.adv_use_random_command):
            adv_action_for_reg[:, :self.command_dim] = 0.0
            prev_for_reg = prev_for_reg.clone()
            prev_for_reg[:, :self.command_dim] = 0.0
        (
            vel_err,
            axis_vel_err_max,
            yaw_err,
            att,
            omega,
            force_margin,
            vx_err_abs,
            vy_err_abs,
            vz_err_abs,
        ) = self._aux_metrics()
        alive = (~(done | bad_done | exceed)).float()
        near_bad_att = torch.relu(att - float(self.args.adv_attitude_safe_rad))
        near_bad_omega = torch.relu(omega - float(self.args.adv_omega_safe_rad))
        jerk = torch.mean(torch.abs(adv_action_for_reg - prev_for_reg), dim=1)
        cmd_jerk, obs_jerk, wind_jerk = self._action_jerk_stats(
            adv_action_for_reg, prev_for_reg
        )
        linf = torch.max(torch.abs(adv_action_for_reg), dim=1).values
        saturation = (torch.abs(adv_action_for_reg) > 0.98).float().mean(dim=1)
        raw_excess = torch.relu(torch.abs(adv_action) - 1.0).mean(dim=1)
        obs_step_energy, obs_window_energy = self._update_obs_energy()
        cmd_rms, cmd_abs, cmd_linf = self._state_norm_stats(
            self.adv_command, self.command_norm_denom
        )
        obs_rms, obs_abs_norm, obs_linf = self._state_norm_stats(
            self.adv_obs, self.obs_norm_denom, self.obs_attack_mask
        )
        wind_rms, wind_abs_norm, wind_linf = self._state_norm_stats(
            self.adv_wind, self.wind_norm_denom
        )
        if bool(self.args.adv_use_random_command):
            cmd_range_penalty = torch.zeros_like(cmd_rms)
            cmd_jerk = torch.zeros_like(cmd_jerk)
            cmd_rms = torch.zeros_like(cmd_rms)
            cmd_abs = torch.zeros_like(cmd_abs)
            cmd_linf = torch.zeros_like(cmd_linf)
        else:
            cmd_range_penalty = self._range_penalty(
                cmd_rms,
                self.args.adv_command_target_rms_min,
                self.args.adv_command_target_rms_max,
            )
        obs_range_penalty = self._range_penalty(
            obs_rms,
            self.args.adv_obs_target_rms_min,
            self.args.adv_obs_target_rms_max,
        )
        wind_range_penalty = self._range_penalty(
            wind_rms,
            self.args.adv_wind_target_rms_min,
            self.args.adv_wind_target_rms_max,
        )
        yaw_margin = torch.relu(
            yaw_err - np.deg2rad(float(self.args.adv_yaw_margin_deg))
        )
        axis_vel_margin = torch.relu(
            axis_vel_err_max - float(self.args.adv_axis_vel_margin)
        )

        policy_proximal_return = self._update_policy_reward_return(policy_reward)
        horizon_loss = -policy_proximal_return
        reward_policy = float(self.args.adv_policy_reward_weight) * horizon_loss
        reward_vel = float(self.args.adv_w_vel_error) * vel_err.detach()
        reward_axis_vel = float(self.args.adv_w_axis_vel_error) * axis_vel_err_max.detach()
        reward_yaw = float(self.args.adv_w_yaw_error) * yaw_err.detach()
        reward_axis_vel_margin = float(self.args.adv_w_vel_bad_margin) * axis_vel_margin.detach()
        reward_yaw_margin = float(self.args.adv_w_yaw_bad_margin) * yaw_margin.detach()
        reward_attitude = float(self.args.adv_w_attitude) * near_bad_att.detach()
        reward_omega = float(self.args.adv_w_omega) * near_bad_omega.detach()
        reward_force_margin = float(self.args.adv_w_force_margin) * force_margin.detach()
        aux = (
            reward_vel
            + reward_axis_vel
            + reward_yaw
            + reward_axis_vel_margin
            + reward_yaw_margin
            + reward_attitude
            + reward_omega
            + reward_force_margin
        )
        terminal = float(self.args.adv_bad_done_bonus) * bad_done.float()
        alive_penalty = -float(self.args.adv_alive_penalty) * alive
        reg_linf = float(self.args.adv_linf_penalty) * linf
        reg_command_range = float(self.args.adv_command_range_penalty) * cmd_range_penalty
        reg_obs_range = float(self.args.adv_obs_range_penalty) * obs_range_penalty
        reg_wind_range = float(self.args.adv_wind_range_penalty) * wind_range_penalty
        reg_saturation = float(self.args.adv_saturation_penalty) * saturation
        reg_raw_excess = float(self.args.adv_raw_excess_penalty) * raw_excess
        reg_obs_energy = float(self.args.adv_obs_energy_penalty) * self.obs_energy_penalty
        reg = (
            reg_linf
            + reg_command_range
            + reg_obs_range
            + reg_wind_range
            + reg_saturation
            + reg_raw_excess
            + reg_obs_energy
        )
        reward = alive_penalty + reward_policy + aux + terminal - reg

        task = self.env.task
        raw_sticks = torch.stack((
            task.raw_vx,
            task.raw_vy,
            task.raw_vz,
            task.raw_yaw,
        ), dim=1)
        filtered_sticks = torch.stack((
            task.stick_vx,
            task.stick_vy,
            task.stick_vz,
            task.stick_yaw,
        ), dim=1)
        target_commands = torch.stack((
            task.target_vx,
            task.target_vy,
            task.target_vz,
            task.target_yaw_rate,
        ), dim=1)

        wind_ned_abs = torch.zeros(3, device=self.device)
        wind_ned_mean = torch.zeros(3, device=self.device)
        wind_body_abs = torch.zeros(3, device=self.device)
        wind_pqr_abs = torch.zeros(3, device=self.device)
        wind_force_abs = torch.zeros(3, device=self.device)
        wind_moment_abs = torch.zeros(3, device=self.device)
        wind_force_norm = torch.zeros(self.n, device=self.device)
        wind_moment_norm = torch.zeros(self.n, device=self.device)
        airspeed = torch.zeros(self.n, device=self.device)
        ground_body_speed = torch.zeros(self.n, device=self.device)
        if hasattr(self.env.model, "get_wind_ned"):
            wn, we, wd = self.env.model.get_wind_ned()
            wind_ned = torch.stack((wn, we, wd), dim=1)
            wind_ned_abs = wind_ned.detach().abs().mean(dim=0)
            wind_ned_mean = wind_ned.detach().mean(dim=0)
        if hasattr(self.env.model, "get_wind_body"):
            wx, wy, wz = self.env.model.get_wind_body()
            wind_body = torch.stack((wx, wy, wz), dim=1)
            wind_body_abs = wind_body.detach().abs().mean(dim=0)
        if hasattr(self.env.model, "get_wind_pqr_body"):
            wp, wq, wr = self.env.model.get_wind_pqr_body()
            wind_pqr = torch.stack((wp, wq, wr), dim=1)
            wind_pqr_abs = wind_pqr.detach().abs().mean(dim=0)
        if hasattr(self.env.model, "get_wind_force_body"):
            fx, fy, fz = self.env.model.get_wind_force_body()
            wind_force = torch.stack((fx, fy, fz), dim=1)
            wind_force_abs = wind_force.detach().abs().mean(dim=0)
            wind_force_norm = torch.sqrt(torch.sum(wind_force * wind_force, dim=1))
        if hasattr(self.env.model, "get_wind_moment_body"):
            mx, my, mz = self.env.model.get_wind_moment_body()
            wind_moment = torch.stack((mx, my, mz), dim=1)
            wind_moment_abs = wind_moment.detach().abs().mean(dim=0)
            wind_moment_norm = torch.sqrt(torch.sum(wind_moment * wind_moment, dim=1))
        if hasattr(self.env.model, "get_air_relative_velocity_body"):
            ua, va, wa = self.env.model.get_air_relative_velocity_body()
            airspeed = torch.sqrt((ua * ua + va * va + wa * wa).clamp_min(0.0))
        if hasattr(self.env.model, "s"):
            u, v, w = self.env.model.s[:, 6], self.env.model.s[:, 7], self.env.model.s[:, 8]
            ground_body_speed = torch.sqrt((u * u + v * v + w * w).clamp_min(0.0))

        self.last_info = {
            "adv/vel_error": vel_err.detach().mean().item(),
            "adv/vx_error_abs": vx_err_abs.detach().mean().item(),
            "adv/vy_error_abs": vy_err_abs.detach().mean().item(),
            "adv/vz_error_abs": vz_err_abs.detach().mean().item(),
            "adv/axis_vel_error": axis_vel_err_max.detach().mean().item(),
            "adv/axis_vel_margin": axis_vel_margin.detach().mean().item(),
            "adv/yaw_error": yaw_err.detach().mean().item(),
            "adv/yaw_margin": yaw_margin.detach().mean().item(),
            "adv/attitude": att.detach().mean().item(),
            "adv/omega": omega.detach().mean().item(),
            "adv/linf": linf.detach().mean().item(),
            "adv/saturation_frac": saturation.detach().mean().item(),
            "adv/raw_excess": raw_excess.detach().mean().item(),
            "adv/jerk": jerk.detach().mean().item(),
            "adv/command_jerk": cmd_jerk.detach().mean().item(),
            "adv/obs_jerk": obs_jerk.detach().mean().item(),
            "adv/wind_jerk": wind_jerk.detach().mean().item(),
            "adv/obs_step_energy": obs_step_energy.detach().mean().item(),
            "adv/obs_window_energy": obs_window_energy.detach().mean().item(),
            "adv/obs_energy_ratio": (obs_window_energy / max(self.obs_energy_budget, 1e-6)).detach().mean().item(),
            "adv/obs_energy_penalty": self.obs_energy_penalty.detach().mean().item(),
            "adv/command_range_penalty": cmd_range_penalty.detach().mean().item(),
            "adv/obs_range_penalty": obs_range_penalty.detach().mean().item(),
            "adv/wind_range_penalty": wind_range_penalty.detach().mean().item(),
            "adv/bad_done_rate": bad_done.float().mean().item(),
            "adv/policy_reward": policy_reward.detach().mean().item(),
            "adv/policy_proximal_return": policy_proximal_return.detach().mean().item(),
            "adv/policy_reward_window": float(self.policy_reward_window),
            "adv/command_abs": self.adv_command.detach().abs().mean().item(),
            "adv/command_norm_rms": cmd_rms.detach().mean().item(),
            "adv/command_norm_abs": cmd_abs.detach().mean().item(),
            "adv/command_norm_linf": cmd_linf.detach().mean().item(),
            "adv/obs_abs": (self.adv_obs * self.obs_attack_mask).detach().abs().mean().item(),
            "adv/obs_norm_rms": obs_rms.detach().mean().item(),
            "adv/obs_norm_abs": obs_abs_norm.detach().mean().item(),
            "adv/obs_norm_linf": obs_linf.detach().mean().item(),
            "adv/obs_active_frac": ((self.adv_obs * self.obs_attack_mask).detach().abs() > 1e-8).float().mean().item(),
            "adv/wind_abs": self.adv_wind.detach().abs().mean().item(),
            "adv/wind_norm_rms": wind_rms.detach().mean().item(),
            "adv/wind_norm_abs": wind_abs_norm.detach().mean().item(),
            "adv/wind_norm_linf": wind_linf.detach().mean().item(),
            "adv/raw_stick_abs": raw_sticks.detach().abs().mean().item(),
            "adv/raw_stick_linf": raw_sticks.detach().abs().max(dim=1).values.mean().item(),
            "adv/filtered_stick_abs": filtered_sticks.detach().abs().mean().item(),
            "adv/filtered_stick_linf": filtered_sticks.detach().abs().max(dim=1).values.mean().item(),
            "adv/target_command_abs": target_commands.detach().abs().mean().item(),
            "adv/target_command_linf": target_commands.detach().abs().max(dim=1).values.mean().item(),
            "adv/target_vx_abs": task.target_vx.detach().abs().mean().item(),
            "adv/target_vy_abs": task.target_vy.detach().abs().mean().item(),
            "adv/target_vz_abs": task.target_vz.detach().abs().mean().item(),
            "adv/target_yaw_rate_abs": task.target_yaw_rate.detach().abs().mean().item(),
            "adv/wind_north_mean": wind_ned_mean[0].item(),
            "adv/wind_east_mean": wind_ned_mean[1].item(),
            "adv/wind_down_mean": wind_ned_mean[2].item(),
            "adv/wind_north_abs": wind_ned_abs[0].item(),
            "adv/wind_east_abs": wind_ned_abs[1].item(),
            "adv/wind_down_abs": wind_ned_abs[2].item(),
            "adv/wind_body_x_abs": wind_body_abs[0].item(),
            "adv/wind_body_y_abs": wind_body_abs[1].item(),
            "adv/wind_body_z_abs": wind_body_abs[2].item(),
            "adv/wind_p_abs": wind_pqr_abs[0].item(),
            "adv/wind_q_abs": wind_pqr_abs[1].item(),
            "adv/wind_r_abs": wind_pqr_abs[2].item(),
            "adv/wind_force_x_abs": wind_force_abs[0].item(),
            "adv/wind_force_y_abs": wind_force_abs[1].item(),
            "adv/wind_force_z_abs": wind_force_abs[2].item(),
            "adv/wind_force_norm": wind_force_norm.detach().mean().item(),
            "adv/wind_moment_l_abs": wind_moment_abs[0].item(),
            "adv/wind_moment_m_abs": wind_moment_abs[1].item(),
            "adv/wind_moment_n_abs": wind_moment_abs[2].item(),
            "adv/wind_moment_norm": wind_moment_norm.detach().mean().item(),
            "adv/airspeed_mean": airspeed.detach().mean().item(),
            "adv/ground_body_speed_mean": ground_body_speed.detach().mean().item(),
            "adv/reward_policy": reward_policy.detach().mean().item(),
            "adv/reward_aux": aux.detach().mean().item(),
            "adv/reward_terminal": terminal.detach().mean().item(),
            "adv/reward_alive_penalty": alive_penalty.detach().mean().item(),
            "adv/reward_reg": reg.detach().mean().item(),
            "adv/reward_vel": reward_vel.detach().mean().item(),
            "adv/reward_axis_vel": reward_axis_vel.detach().mean().item(),
            "adv/reward_yaw": reward_yaw.detach().mean().item(),
            "adv/reward_axis_vel_margin": reward_axis_vel_margin.detach().mean().item(),
            "adv/reward_yaw_margin": reward_yaw_margin.detach().mean().item(),
            "adv/reward_attitude": reward_attitude.detach().mean().item(),
            "adv/reward_omega": reward_omega.detach().mean().item(),
            "adv/reward_force_margin": reward_force_margin.detach().mean().item(),
            "adv/reg_linf": reg_linf.detach().mean().item(),
            "adv/reg_command_range": reg_command_range.detach().mean().item(),
            "adv/reg_obs_range": reg_obs_range.detach().mean().item(),
            "adv/reg_wind_range": reg_wind_range.detach().mean().item(),
            "adv/reg_saturation": reg_saturation.detach().mean().item(),
            "adv/reg_raw_excess": reg_raw_excess.detach().mean().item(),
            "adv/reg_obs_energy": reg_obs_energy.detach().mean().item(),
        }
        self.prev_adv_action = adv_action_bounded.detach().clone()
        return reward.reshape(-1, 1)

    def step(self, adv_action):
        adv_action = torch.as_tensor(adv_action, dtype=torch.float32, device=self.device).detach()
        if adv_action.ndim == 3:
            adv_action = adv_action.reshape(self.n, -1)
        if adv_action.shape[1] != self.adv_action_dim:
            raise ValueError(
                f"adversary action dim mismatch: got {adv_action.shape[1]}, "
                f"expected {self.adv_action_dim}"
            )

        cmd, obs_delta, wind_delta = self._split_adv_action(adv_action)
        if bool(self.args.adv_use_random_command):
            cmd.zero_()
        obs_target = self._lowpass_adv(obs_delta, self.adv_obs, float(self.args.adv_obs_alpha))
        self.adv_obs = self._rate_limit_adv(
            obs_target,
            self.adv_obs,
            self.obs_rate_limit_frac,
            self.obs_scale,
        )
        self.adv_obs = self.adv_obs * self.obs_attack_mask

        reset_before = (
            self.env.is_done.bool()
            | self.env.bad_done.bool()
            | self.env.exceed_time_limit.bool()
        )
        self.env.reset()
        if torch.any(reset_before):
            self._reset_adv_wind_source(reset_before)
            if self._use_random_command_base():
                self._enable_random_command_if_needed(reset_before)
            else:
                self._set_neutral_command(reset_before)
        if bool(self.args.adv_use_random_command):
            self.env.task.sync_command(self.env)
        elif bool(getattr(self.args, "adv_command_random_base", False)):
            pass
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
        adv_reward = self._adv_reward(policy_reward, done, bad_done, exceed, adv_action)
        if torch.any(reset):
            self._reset_adv_state(reset)
        self.victim_masks = (~reset).float().reshape(-1, 1)
        self.current_obs = obs.detach()
        return self.current_obs, adv_reward.detach(), done, bad_done, exceed, info
