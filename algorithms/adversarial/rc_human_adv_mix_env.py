import copy
from types import SimpleNamespace

import gym
import numpy as np
import torch

from algorithms.adversarial.ppo_actor import AdversarialPPOActor
from envs.wind.dryden_turbulence import EnvDrydenTurbulence


class RCHumanAdvMixTrainEnv:
    """Training wrapper for robust rc_human PPO.

    A fixed fraction of vectorized environments receives command/observation/
    wind attacks from a frozen adversary.  The remaining environments use the
    normal rc_human command generator, but curriculum levels are sampled
    uniformly across the full level range instead of the adaptive curriculum.
    The PPO policy still controls the aircraft action in every environment.
    """

    def __init__(self, env, args, device):
        self.env = env
        self.args = args
        self.device = device
        self.num_envs = env.num_envs
        self.num_agents = env.num_agents
        self.n = env.n
        self.task = env.task
        self.model = env.model
        self.config = env.config

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_observation = env.num_observation
        self.num_actions = env.num_actions

        self.adv_fraction = float(getattr(args, "adv_mix_frac", 0.10))
        adv_count = int(round(self.n * self.adv_fraction))
        if self.adv_fraction > 0.0:
            adv_count = max(1, adv_count)
        adv_count = min(max(adv_count, 0), self.n)
        perm = torch.randperm(self.n, device=device)
        self.adv_mask = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.adv_mask[perm[:adv_count]] = True
        self.normal_mask = ~self.adv_mask

        self.level_count = min(
            int(getattr(args, "uniform_curriculum_levels", 120)),
            int(self.task.max_curriculum_level) + 1,
        )
        if self.level_count <= 0:
            raise ValueError("uniform_curriculum_levels must be positive")
        self.level_cursor = 0
        self.level_order = torch.randperm(self.level_count, device=device)

        self.command_dim = 4
        self.wind_velocity_dim = 3
        self.wind_pqr_dim = 3 if bool(
            getattr(env.config, "enable_dryden_angular_turbulence", True)
        ) else 0
        self.wind_dim = self.wind_velocity_dim + self.wind_pqr_dim
        self.allowed_obs_attack_idx = torch.tensor(
            [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24],
            dtype=torch.long,
            device=device,
        )
        self.obs_attack_dim = int(self.allowed_obs_attack_idx.numel())
        self.adv_action_dim = self.command_dim + self.obs_attack_dim + self.wind_dim
        self.adv_action_space = gym.spaces.Box(
            low=-np.ones(self.adv_action_dim, dtype=np.float32),
            high=np.ones(self.adv_action_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.command_rate_limit_frac = float(
            getattr(args, "adv_command_rate_limit_frac", 0.0)
        )
        self.obs_rate_limit_frac = float(getattr(args, "adv_obs_rate_limit_frac", 0.1))
        self.wind_rate_limit_frac = float(getattr(args, "adv_wind_rate_limit_frac", 0.1))

        self.command_low, self.command_high = self._build_command_bounds()
        self.command_scale = torch.maximum(
            torch.abs(torch.minimum(self.command_low, torch.zeros_like(self.command_low))),
            torch.maximum(self.command_high, torch.zeros_like(self.command_high)),
        )
        self.obs_scale = self._build_obs_scale() * float(args.adv_obs_frac)
        self.obs_attack_mask = torch.zeros(self.num_observation, device=device)
        self.obs_attack_mask[self.allowed_obs_attack_idx] = 1.0
        self.obs_scale = self.obs_scale * self.obs_attack_mask
        self.obs_action_scale = self.obs_scale[self.allowed_obs_attack_idx]
        self.wind_scale = self._build_wind_scale() * float(args.adv_wind_frac)

        self.adv_command = torch.zeros((self.n, self.command_dim), device=device)
        self.adv_obs = torch.zeros((self.n, self.num_observation), device=device)
        self.adv_wind = torch.zeros((self.n, self.wind_dim), device=device)
        self.prev_adv_action = torch.zeros((self.n, self.adv_action_dim), device=device)
        self.adv_rnn = torch.zeros((self.n, 1, 1), device=device)
        self.adv_masks = torch.ones((self.n, 1), device=device)
        self.current_obs = None

        self.adv_wind_source = self._make_adv_wind_source()
        self.adv_actor = self._load_adv_actor(args.adv_ckpt)
        self._patch_uniform_curriculum_sampling()

    @property
    def agents(self):
        return self.num_agents

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def reset(self):
        obs = self.env.reset()
        reset = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self._assign_uniform_levels(reset & self.normal_mask)
        self._reset_random_wind(reset)
        self._reset_adv_state(reset & self.adv_mask)
        self._reset_adv_wind_source(reset & self.adv_mask)
        self._apply_combined_wind()
        self._freeze_adv_sync()
        self.current_obs = self._obs_with_adv()
        return self.current_obs

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.ndim == 3:
            action = action.reshape(self.n, -1)

        reset_before = (
            self.env.is_done.bool()
            | self.env.bad_done.bool()
            | self.env.exceed_time_limit.bool()
        )
        self.env.reset()
        if torch.any(reset_before):
            self._assign_uniform_levels(reset_before & self.normal_mask)
            self._reset_random_wind(reset_before)
            self._reset_adv_state(reset_before & self.adv_mask)
            self._reset_adv_wind_source(reset_before & self.adv_mask)
            self._apply_combined_wind()

        self._freeze_adv_sync()
        if self.current_obs is None or torch.any(reset_before):
            self.current_obs = self._obs_with_adv()
        adv_action = self._adv_act(self.current_obs)
        self._apply_adv_command(adv_action)
        self._advance_wind(adv_action)

        if hasattr(self.task, "maybe_override_action"):
            action = self.task.maybe_override_action(self.env, action)
        self.model.update(action)
        self.env.step_count += 1

        self._freeze_adv_sync()
        obs = self._obs_with_adv()
        self._freeze_adv_sync()
        info = self.env.info()
        done, bad_done, exceed_time_limit, info = self.env.done(info)
        self._freeze_adv_sync()
        reward = self.env.reward().reshape(-1, 1)
        self._freeze_adv_sync()
        self.task.step(self.env)

        reset = done | bad_done | exceed_time_limit
        if torch.any(reset):
            self._reset_adv_state(reset & self.adv_mask)
        self.adv_masks = (~reset).float().reshape(-1, 1)
        self.current_obs = obs.detach()
        return self.current_obs, reward, done, bad_done, exceed_time_limit, info

    def _patch_uniform_curriculum_sampling(self):
        task = self.task
        self._original_sample_command_level = task._sample_command_level
        self._original_update_curriculum = task._update_curriculum_from_last_episode
        self._original_get_training_metrics = task.get_training_metrics

        def sample_command_level(idx):
            idx = idx.to(device=self.device, dtype=torch.long)
            levels = self._draw_uniform_levels(int(idx.numel()))
            task.curriculum_level[idx] = levels
            return levels

        def update_curriculum_noop(_env, _reset):
            return None

        def get_training_metrics():
            metrics = self._original_get_training_metrics()
            metrics.update(self.get_training_metrics())
            return metrics

        task._sample_command_level = sample_command_level
        task._update_curriculum_from_last_episode = update_curriculum_noop
        task.get_training_metrics = get_training_metrics

    def get_training_metrics(self):
        normal_levels = self.task.curriculum_level[self.normal_mask]
        adv_levels = self.task.curriculum_level[self.adv_mask]
        return {
            "rc_human_mix/adversarial_env_frac": self.adv_mask.float().mean(),
            "rc_human_mix/normal_env_frac": self.normal_mask.float().mean(),
            "rc_human_mix/uniform_level_count": torch.tensor(
                float(self.level_count), device=self.device
            ),
            "rc_human_mix/normal_level_mean": normal_levels.float().mean()
            if normal_levels.numel()
            else torch.tensor(0.0, device=self.device),
            "rc_human_mix/adv_level_mean": adv_levels.float().mean()
            if adv_levels.numel()
            else torch.tensor(0.0, device=self.device),
            "rc_human_mix/adv_command_abs": self.adv_command[self.adv_mask].abs().mean()
            if torch.any(self.adv_mask)
            else torch.tensor(0.0, device=self.device),
            "rc_human_mix/adv_obs_abs": (self.adv_obs[self.adv_mask] * self.obs_attack_mask).abs().mean()
            if torch.any(self.adv_mask)
            else torch.tensor(0.0, device=self.device),
            "rc_human_mix/adv_wind_abs": self.adv_wind[self.adv_mask].abs().mean()
            if torch.any(self.adv_mask)
            else torch.tensor(0.0, device=self.device),
        }

    def _draw_uniform_levels(self, size):
        if size <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        chunks = []
        remaining = size
        while remaining > 0:
            available = self.level_count - self.level_cursor
            take = min(remaining, available)
            chunks.append(self.level_order[self.level_cursor:self.level_cursor + take])
            self.level_cursor += take
            remaining -= take
            if self.level_cursor >= self.level_count:
                self.level_order = torch.randperm(self.level_count, device=self.device)
                self.level_cursor = 0
        return torch.cat(chunks, dim=0).long()

    def _assign_uniform_levels(self, mask):
        if int(torch.sum(mask).item()) == 0:
            return
        idx = torch.where(mask)[0]
        self.task.curriculum_level[idx] = self._draw_uniform_levels(int(idx.numel()))

    def _build_command_bounds(self):
        frac = min(max(float(self.args.adv_command_frac), 0.0), 1.0)
        span = torch.ones(self.command_dim, device=self.device) * frac
        return -span, span

    def _build_obs_scale(self):
        cfg = self.env.config
        sensor_vel = float(getattr(cfg, "sensor_vel_std", 0.05))
        sensor_pos = float(getattr(cfg, "sensor_pos_std", 1.0))
        sensor_att = float(getattr(cfg, "sensor_att_std", 0.005))
        sensor_omega = float(getattr(cfg, "sensor_omega_std", 0.0005))

        sx = torch.ones(self.num_observation, device=self.device) * float(
            self.args.adv_obs_default_scale
        )
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
        values = [
            max(abs(float(getattr(cfg, "dryden_mean_wind_north_min", -1.0))),
                abs(float(getattr(cfg, "dryden_mean_wind_north_max", 1.0)))),
            max(abs(float(getattr(cfg, "dryden_mean_wind_east_min", -1.0))),
                abs(float(getattr(cfg, "dryden_mean_wind_east_max", 1.0)))),
            max(abs(float(getattr(cfg, "dryden_mean_wind_down_min", -0.15))),
                abs(float(getattr(cfg, "dryden_mean_wind_down_max", 0.15)))),
        ]
        if self.wind_pqr_dim:
            values.extend([
                abs(float(getattr(cfg, "dryden_pqr_attack_p_max", 0.6))),
                abs(float(getattr(cfg, "dryden_pqr_attack_q_max", 0.6))),
                abs(float(getattr(cfg, "dryden_pqr_attack_r_max", 0.5))),
            ])
        return torch.tensor(values, device=self.device)

    def _scale_command_action(self, action):
        pos_scale = torch.maximum(self.command_high, torch.zeros_like(self.command_high))
        neg_scale = torch.abs(torch.minimum(self.command_low, torch.zeros_like(self.command_low)))
        return torch.where(action >= 0.0, action * pos_scale, action * neg_scale)

    def _split_adv_action(self, adv_action):
        adv_action = torch.clamp(adv_action, -1.0, 1.0)
        c_end = self.command_dim
        o_end = c_end + self.obs_attack_dim
        cmd = self._scale_command_action(adv_action[:, :c_end])
        obs_compact = adv_action[:, c_end:o_end] * self.obs_action_scale.reshape(1, -1)
        obs = torch.zeros(
            (adv_action.shape[0], self.num_observation),
            dtype=adv_action.dtype,
            device=self.device,
        )
        obs[:, self.allowed_obs_attack_idx] = obs_compact
        wind = adv_action[:, o_end:] * self.wind_scale.reshape(1, -1)
        return cmd, obs, wind

    def _lowpass_adv(self, new, old, alpha):
        return (1.0 - alpha) * old + alpha * new

    def _rate_limit_adv(self, new, old, limit_frac, scale):
        limit_frac = float(limit_frac)
        if limit_frac <= 0.0:
            return new
        max_delta = torch.abs(scale).reshape(1, -1) * limit_frac
        return old + torch.clamp(new - old, -max_delta, max_delta)

    def _reset_adv_state(self, mask):
        if int(torch.sum(mask).item()) == 0:
            return
        self.adv_command[mask] = 0.0
        self.adv_obs[mask] = 0.0
        self.adv_wind[mask] = 0.0
        self.prev_adv_action[mask] = 0.0
        self.adv_rnn[mask] = 0.0
        self.adv_masks[mask] = 1.0

    def _make_adv_wind_source(self):
        if not hasattr(self.env.model, "set_wind_gust_ned"):
            return None
        if not bool(getattr(self.env.model, "wind_enabled", False)):
            return None
        cfg = copy.copy(self.env.config)
        cfg.enable_dryden_turbulence = False
        cfg.dryden_randomize = False
        cfg.dryden_domain_randomization = False
        cfg.dryden_sigma_curriculum_enable = False
        cfg.dryden_mean_wind_curriculum_enable = False
        return EnvDrydenTurbulence(cfg, self.n, self.device, self.env.model.dt)

    def _reset_random_wind(self, mask):
        if self.env.wind_disturbance is None:
            return
        self.env.wind_disturbance.reset(self.env, mask)

    def _reset_adv_wind_source(self, mask):
        if self.adv_wind_source is None:
            return
        self.adv_wind_source.reset(self.env, mask)

    def _apply_combined_wind(self):
        if not hasattr(self.model, "set_wind_gust_ned"):
            return
        if not bool(getattr(self.model, "wind_enabled", False)):
            return
        gust = torch.zeros((self.n, 3), device=self.device)
        pqr = torch.zeros((self.n, 3), device=self.device)
        if self.env.wind_disturbance is not None:
            gust = self.env.wind_disturbance.gust_ned.clone()
            pqr = self.env.wind_disturbance.gust_pqr_body.clone()
        if self.adv_wind_source is not None and torch.any(self.adv_mask):
            gust[self.adv_mask] = self.adv_wind_source.gust_ned[self.adv_mask]
            pqr[self.adv_mask] = self.adv_wind_source.gust_pqr_body[self.adv_mask]
        self.model.set_wind_gust_ned(
            gust[:, 0],
            gust[:, 1],
            gust[:, 2],
            pqr_body=pqr,
        )

    def _load_adv_actor(self, ckpt_path):
        if not ckpt_path:
            raise ValueError("--adv-ckpt is required for adv-mix robust training")
        actor_args = SimpleNamespace(
            hidden_size=getattr(self.args, "adv_hidden_size", "128 128 128"),
            activation_id=int(getattr(self.args, "adv_activation_id", 1)),
            use_feature_normalization=False,
            adv_action_bound=1.0,
            tpdv=dict(dtype=torch.float32, device=self.device),
        )
        actor = AdversarialPPOActor(
            actor_args,
            self.observation_space,
            self.adv_action_space,
            self.device,
        )
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict):
            state = state.get("policy", state.get("state_dict", state))
        actor.load_state_dict(state)
        actor.eval()
        for param in actor.parameters():
            param.requires_grad_(False)
        return actor

    @torch.no_grad()
    def _adv_act(self, obs):
        adv_action = torch.zeros((self.n, self.adv_action_dim), device=self.device)
        if not torch.any(self.adv_mask):
            return adv_action
        actor_obs = obs.detach()
        action, _, self.adv_rnn = self.adv_actor(
            actor_obs,
            self.adv_rnn,
            self.adv_masks,
            deterministic=not bool(getattr(self.args, "stochastic_adv", False)),
        )
        adv_action[self.adv_mask] = action[self.adv_mask]
        return adv_action

    def _apply_adv_command(self, adv_action):
        if not torch.any(self.adv_mask):
            return
        cmd, obs_delta, _wind_delta = self._split_adv_action(adv_action)
        obs_target = self._lowpass_adv(
            obs_delta, self.adv_obs, float(self.args.adv_obs_alpha)
        )
        self.adv_obs = self._rate_limit_adv(
            obs_target,
            self.adv_obs,
            self.obs_rate_limit_frac,
            self.obs_scale,
        ) * self.obs_attack_mask

        target = self._lowpass_adv(
            cmd, self.adv_command, float(self.args.adv_command_alpha)
        )
        self.adv_command = self._rate_limit_adv(
            target,
            self.adv_command,
            self.command_rate_limit_frac,
            self.command_scale,
        )
        raw = self.adv_command
        task = self.task
        mask = self.adv_mask
        task.raw_vx[mask] = raw[mask, 0]
        task.raw_vy[mask] = raw[mask, 1]
        task.raw_vz[mask] = raw[mask, 2]
        task.raw_yaw[mask] = raw[mask, 3]
        if hasattr(task, "desired_raw_vx"):
            task.desired_raw_vx[mask] = raw[mask, 0]
            task.desired_raw_vy[mask] = raw[mask, 1]
            task.desired_raw_vz[mask] = raw[mask, 2]
            task.desired_raw_yaw[mask] = raw[mask, 3]
        task.vx_forward_limit[mask] = task.vx_max
        task._apply_px4_vtol_mc_manual_sticks(mask, self.env)
        self._freeze_adv_sync()

    def _advance_wind(self, adv_action):
        if self.env.wind_disturbance is not None:
            self.env.wind_disturbance.step()
        if self.adv_wind_source is not None and torch.any(self.adv_mask):
            _cmd, _obs_delta, wind_delta = self._split_adv_action(adv_action)
            target = self._lowpass_adv(
                wind_delta, self.adv_wind, float(self.args.adv_wind_alpha)
            )
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
        self._apply_combined_wind()

    def _freeze_adv_sync(self):
        if not torch.any(self.adv_mask):
            return
        if hasattr(self.task, "last_synced_step"):
            self.task.last_synced_step[self.adv_mask] = self.env.step_count[
                self.adv_mask
            ].long()

    def _obs_with_adv(self):
        self._freeze_adv_sync()
        obs = self.env.obs()
        if torch.any(self.adv_mask):
            self._freeze_adv_sync()
            clean_obs = self.task.get_clean_obs(self.env)
            obs[self.adv_mask] = clean_obs[self.adv_mask] + self.adv_obs[self.adv_mask]
        return obs.detach()
