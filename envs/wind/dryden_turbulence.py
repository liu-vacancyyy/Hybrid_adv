import math

import torch


class EnvDrydenTurbulence:
    """Vectorised Dryden gust generator for environment disturbances.

    The generator uses the standard low-altitude Dryden structure:

    - longitudinal gust: first-order shaping filter;
    - lateral / vertical gusts: second-order Dryden shaping filters with
      numerator zero;
    - optional per-episode mean wind and intensity curriculum.

    Output is wind velocity in NED/world axes, in m/s.  The aircraft model is
    responsible for converting that wind to body axes before computing relative
    airspeed, alpha and beta.
    """

    def __init__(self, config, n, device, dt):
        self.config = config
        self.n = n
        self.device = device
        self.dt = float(dt)

        self.sigma_ref = float(getattr(config, 'dryden_sigma_ref', 1.5))
        self.sigma_ref_min = float(getattr(config, 'dryden_sigma_ref_min', self.sigma_ref))
        self.sigma_ref_max = float(getattr(config, 'dryden_sigma_ref_max', self.sigma_ref))
        self.sigma_curriculum_enable = bool(
            getattr(config, 'dryden_sigma_curriculum_enable', False)
        )
        self.domain_randomization = bool(
            getattr(config, 'dryden_domain_randomization',
                    getattr(config, 'dryden_randomize', True))
        )
        self.sigma_scale_min = float(getattr(config, 'dryden_sigma_scale_min', 0.8))
        self.sigma_scale_max = float(getattr(config, 'dryden_sigma_scale_max', 1.2))
        self.airspeed_min = float(getattr(config, 'dryden_airspeed_min', 1.0))
        self.airspeed_max = float(getattr(config, 'dryden_airspeed_max', 8.0))
        self.altitude_min = float(getattr(config, 'dryden_altitude_min', 5.0))
        self.altitude_max = float(getattr(config, 'dryden_altitude_max', 120.0))
        self.length_scale_min = float(getattr(config, 'dryden_length_scale_min', 0.7))
        self.length_scale_max = float(getattr(config, 'dryden_length_scale_max', 1.3))
        self.random_direction = bool(getattr(config, 'dryden_random_direction', True))
        self.direction_deg = float(getattr(config, 'dryden_direction_deg', 0.0))
        self.mean_wind_curriculum_enable = bool(
            getattr(config, 'dryden_mean_wind_curriculum_enable', False)
        )
        self.mean_wind_scale_min = float(getattr(config, 'dryden_mean_wind_scale_min', 1.0))
        self.mean_wind_scale_max = float(getattr(config, 'dryden_mean_wind_scale_max', 1.0))

        self.gust_dryden_axes = torch.zeros((n, 3), device=device)
        self.gust_ned = torch.zeros((n, 3), device=device)
        self.mean_wind_ned = torch.zeros((n, 3), device=device)
        self.second_order_pos = torch.zeros((n, 2), device=device)
        self.second_order_rate = torch.zeros((n, 2), device=device)
        self.sigma = torch.ones((n, 3), device=device) * self.sigma_ref
        self.tau = torch.ones((n, 3), device=device)
        self.direction = torch.zeros(n, device=device)

    def _rand_uniform(self, low, high, size):
        low = float(low)
        high = float(high)
        if high <= low:
            return torch.full((size,), low, device=self.device)
        return torch.rand(size, device=self.device) * (high - low) + low

    def _fixed_or_random(self, attr_name, low, high, size, default=None):
        if self.domain_randomization:
            return self._rand_uniform(low, high, size)
        if default is None:
            default = low
        value = float(getattr(self.config, attr_name, default))
        return torch.full((size,), value, device=self.device)

    def _curriculum_progress(self, env, reset_mask, size):
        if env is None or not hasattr(env, 'task'):
            return torch.ones(size, device=self.device)
        task = env.task
        if not hasattr(task, 'curriculum_level') or not hasattr(task, 'max_curriculum_level'):
            return torch.ones(size, device=self.device)
        max_level = max(float(task.max_curriculum_level), 1.0)
        return (task.curriculum_level[reset_mask].float() / max_level).clamp(0.0, 1.0)

    def _progress_range(self, lo, hi, progress):
        return float(lo) + progress * (float(hi) - float(lo))

    def reset(self, env=None, reset_mask=None):
        if reset_mask is None:
            reset_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        else:
            reset_mask = reset_mask.to(device=self.device, dtype=torch.bool)
        size = int(reset_mask.sum().item())
        if size == 0:
            return self.gust_ned

        progress = self._curriculum_progress(env, reset_mask, size)
        sigma_ref = (
            self._progress_range(self.sigma_ref_min, self.sigma_ref_max, progress)
            if self.sigma_curriculum_enable
            else torch.full((size,), self.sigma_ref, device=self.device)
        )

        sigma_scale = self._fixed_or_random(
            'dryden_sigma_scale', self.sigma_scale_min, self.sigma_scale_max,
            size, default=1.0
        ).clamp_min(0.0)
        sigma = sigma_ref * sigma_scale
        self.sigma[reset_mask, :] = sigma[:, None]

        mean_n = self._fixed_or_random(
            'dryden_mean_wind_north',
            getattr(self.config, 'dryden_mean_wind_north_min', 0.0),
            getattr(self.config, 'dryden_mean_wind_north_max', 0.0),
            size, default=0.0
        )
        mean_e = self._fixed_or_random(
            'dryden_mean_wind_east',
            getattr(self.config, 'dryden_mean_wind_east_min', 0.0),
            getattr(self.config, 'dryden_mean_wind_east_max', 0.0),
            size, default=0.0
        )
        mean_d = self._fixed_or_random(
            'dryden_mean_wind_down',
            getattr(self.config, 'dryden_mean_wind_down_min', 0.0),
            getattr(self.config, 'dryden_mean_wind_down_max', 0.0),
            size, default=0.0
        )
        if self.mean_wind_curriculum_enable:
            mean_scale = self._progress_range(
                self.mean_wind_scale_min, self.mean_wind_scale_max, progress
            )
            mean_n = mean_n * mean_scale
            mean_e = mean_e * mean_scale
            mean_d = mean_d * mean_scale
        self.mean_wind_ned[reset_mask, :] = torch.stack((mean_n, mean_e, mean_d), dim=1)

        airspeed = self._fixed_or_random(
            'dryden_airspeed', self.airspeed_min, self.airspeed_max, size,
            default=0.5 * (self.airspeed_min + self.airspeed_max)
        ).clamp_min(0.1)

        altitude = self._fixed_or_random(
            'dryden_altitude', self.altitude_min, self.altitude_max, size,
            default=0.5 * (self.altitude_min + self.altitude_max)
        ).clamp_min(1.0)
        scale = self._fixed_or_random(
            'dryden_length_scale', self.length_scale_min, self.length_scale_max,
            size, default=1.0
        ).clamp_min(0.05)

        # Standard low-altitude Dryden length-scale trend: longitudinal and
        # lateral gusts are more correlated than vertical gusts.  Keep the
        # model in metres and expose the scale multiplier for domain randomiza-
        # tion instead of tying strength to altitude, because the paper fixes
        # turbulence intensity with sigma = 1.5 m/s.
        h = altitude.clamp(5.0, 300.0)
        l_long = (200.0 + 0.5 * h) * scale
        l_vert = (50.0 + 0.2 * h) * scale
        self.tau[reset_mask, 0] = (l_long / airspeed).clamp_min(self.dt)
        self.tau[reset_mask, 1] = (l_long / airspeed).clamp_min(self.dt)
        self.tau[reset_mask, 2] = (l_vert / airspeed).clamp_min(self.dt)

        if self.random_direction:
            self.direction[reset_mask] = self._rand_uniform(
                -math.pi, math.pi, size
            )
        else:
            self.direction[reset_mask] = math.radians(self.direction_deg)

        self.gust_dryden_axes[reset_mask, :] = 0.0
        self.second_order_pos[reset_mask, :] = 0.0
        self.second_order_rate[reset_mask, :] = 0.0
        self._update_ned(reset_mask)
        return self.gust_ned

    def step(self):
        tau = self.tau.clamp_min(self.dt)
        decay_u = torch.exp(-self.dt / tau[:, 0])
        noise_u = self.sigma[:, 0] * torch.sqrt((1.0 - decay_u * decay_u).clamp_min(0.0))
        self.gust_dryden_axes[:, 0] = (
            decay_u * self.gust_dryden_axes[:, 0]
            + noise_u * torch.randn(self.n, device=self.device)
        )

        sqrt_dt = math.sqrt(self.dt)
        for out_axis, state_axis in ((1, 0), (2, 1)):
            a = 1.0 / tau[:, out_axis]
            b = a / math.sqrt(3.0)
            gain = self.sigma[:, out_axis] * torch.sqrt(3.0 * a)

            x = self.second_order_pos[:, state_axis]
            xd = self.second_order_rate[:, state_axis]
            noise = torch.randn(self.n, device=self.device) * sqrt_dt
            xd_new = xd + (-(a * a) * x - 2.0 * a * xd) * self.dt + noise
            x_new = x + xd_new * self.dt

            self.second_order_pos[:, state_axis] = x_new
            self.second_order_rate[:, state_axis] = xd_new
            self.gust_dryden_axes[:, out_axis] = gain * (xd_new + b * x_new)

        self._update_ned()
        return self.gust_ned

    def _update_ned(self, mask=None):
        if mask is None:
            axes = self.gust_dryden_axes
            direction = self.direction
            out = self.gust_ned
            mean = self.mean_wind_ned
        else:
            axes = self.gust_dryden_axes[mask]
            direction = self.direction[mask]
            out = self.gust_ned[mask]
            mean = self.mean_wind_ned[mask]

        along_n = torch.cos(direction)
        along_e = torch.sin(direction)
        lateral_n = -along_e
        lateral_e = along_n

        out[:, 0] = mean[:, 0] + axes[:, 0] * along_n + axes[:, 1] * lateral_n
        out[:, 1] = mean[:, 1] + axes[:, 0] * along_e + axes[:, 1] * lateral_e
        out[:, 2] = mean[:, 2] + axes[:, 2]

        if mask is not None:
            self.gust_ned[mask] = out
