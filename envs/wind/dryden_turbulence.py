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

    If ``enable_dryden_angular_turbulence`` is set, it also generates a
    body-frame angular-rate gust ``gust_pqr_body`` in rad/s.  Adversarial
    training can pass bounded velocity / angular-rate targets into ``step``;
    the same Dryden time constants then shape those targets instead of applying
    raw step changes to the vehicle.
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
        self.external_tau_scale = float(getattr(config, 'dryden_external_tau_scale', 1.0))
        self.external_tau_min = float(getattr(config, 'dryden_external_tau_min', self.dt))
        self.external_tau_max = float(getattr(config, 'dryden_external_tau_max', 0.0))

        self.enable_angular = bool(getattr(config, 'enable_dryden_angular_turbulence', True))
        self.pqr_sigma_ref = float(getattr(config, 'dryden_pqr_sigma_ref', 0.12))
        self.pqr_sigma_scale_min = float(getattr(config, 'dryden_pqr_sigma_scale_min', 0.8))
        self.pqr_sigma_scale_max = float(getattr(config, 'dryden_pqr_sigma_scale_max', 1.2))
        self.pqr_tau_p = float(getattr(config, 'dryden_pqr_tau_p', getattr(config, 'dryden_pqr_tau', 0.8)))
        self.pqr_tau_q = float(getattr(config, 'dryden_pqr_tau_q', getattr(config, 'dryden_pqr_tau', 0.8)))
        self.pqr_tau_r = float(getattr(config, 'dryden_pqr_tau_r', getattr(config, 'dryden_pqr_tau', 1.0)))

        self.gust_dryden_axes = torch.zeros((n, 3), device=device)
        self.gust_ned = torch.zeros((n, 3), device=device)
        self.gust_pqr_body = torch.zeros((n, 3), device=device)
        self.mean_wind_ned = torch.zeros((n, 3), device=device)
        self.second_order_pos = torch.zeros((n, 2), device=device)
        self.second_order_rate = torch.zeros((n, 2), device=device)
        self.sigma = torch.ones((n, 3), device=device) * self.sigma_ref
        self.tau = torch.ones((n, 3), device=device)
        self.pqr_sigma = torch.ones((n, 3), device=device) * self.pqr_sigma_ref
        self.pqr_tau = torch.ones((n, 3), device=device)
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
        if hasattr(self.config, 'dryden_length_longitudinal_m'):
            l_long = torch.full(
                (size,),
                float(getattr(self.config, 'dryden_length_longitudinal_m')),
                device=self.device,
            ) * scale
        else:
            l_long = (200.0 + 0.5 * h) * scale
        if hasattr(self.config, 'dryden_length_lateral_m'):
            l_lat = torch.full(
                (size,),
                float(getattr(self.config, 'dryden_length_lateral_m')),
                device=self.device,
            ) * scale
        else:
            l_lat = l_long
        if hasattr(self.config, 'dryden_length_vertical_m'):
            l_vert = torch.full(
                (size,),
                float(getattr(self.config, 'dryden_length_vertical_m')),
                device=self.device,
            ) * scale
        else:
            l_vert = (50.0 + 0.2 * h) * scale
        self.tau[reset_mask, 0] = (l_long / airspeed).clamp_min(self.dt)
        self.tau[reset_mask, 1] = (l_lat / airspeed).clamp_min(self.dt)
        self.tau[reset_mask, 2] = (l_vert / airspeed).clamp_min(self.dt)

        pqr_scale = self._fixed_or_random(
            'dryden_pqr_sigma_scale', self.pqr_sigma_scale_min, self.pqr_sigma_scale_max,
            size, default=1.0
        ).clamp_min(0.0)
        self.pqr_sigma[reset_mask, :] = self.pqr_sigma_ref * pqr_scale[:, None]
        self.pqr_tau[reset_mask, 0] = max(self.pqr_tau_p, self.dt)
        self.pqr_tau[reset_mask, 1] = max(self.pqr_tau_q, self.dt)
        self.pqr_tau[reset_mask, 2] = max(self.pqr_tau_r, self.dt)

        if self.random_direction:
            self.direction[reset_mask] = self._rand_uniform(
                -math.pi, math.pi, size
            )
        else:
            self.direction[reset_mask] = math.radians(self.direction_deg)

        self.gust_dryden_axes[reset_mask, :] = 0.0
        self.gust_pqr_body[reset_mask, :] = 0.0
        self.second_order_pos[reset_mask, :] = 0.0
        self.second_order_rate[reset_mask, :] = 0.0
        self._update_ned(reset_mask)
        return self.gust_ned

    def step(self, excitation_ned=None, excitation_pqr_body=None):
        if excitation_ned is not None:
            self._step_external_velocity(excitation_ned)
        else:
            self._step_random_velocity()

        if not self.enable_angular:
            self.gust_pqr_body.zero_()
        elif excitation_pqr_body is not None:
            self._step_external_pqr(excitation_pqr_body)
        else:
            self._step_random_pqr()

        return self.gust_ned

    def _step_random_velocity(self):
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

    def _effective_external_tau(self, base_tau):
        tau = base_tau.clamp_min(self.dt) * max(self.external_tau_scale, 0.0)
        tau = tau.clamp_min(max(self.external_tau_min, self.dt))
        if self.external_tau_max > 0.0:
            tau = tau.clamp_max(max(self.external_tau_max, self.dt))
        return tau

    def _ned_to_axes(self, gust_ned):
        gust_ned = torch.as_tensor(gust_ned, dtype=torch.float32, device=self.device)
        if gust_ned.ndim == 1:
            gust_ned = gust_ned.reshape(1, 3).expand(self.n, 3)
        delta = gust_ned - self.mean_wind_ned
        along_n = torch.cos(self.direction)
        along_e = torch.sin(self.direction)
        n = delta[:, 0]
        e = delta[:, 1]
        d = delta[:, 2]
        along = n * along_n + e * along_e
        lateral = -n * along_e + e * along_n
        return torch.stack((along, lateral, d), dim=1)

    def _step_external_velocity(self, target_ned):
        target_axes = self._ned_to_axes(target_ned)
        tau = self._effective_external_tau(self.tau)
        decay = torch.exp(-self.dt / tau)
        self.gust_dryden_axes = (
            decay * self.gust_dryden_axes
            + (1.0 - decay) * target_axes
        )
        self.second_order_pos.zero_()
        self.second_order_rate.zero_()
        self._update_ned()

    def _step_random_pqr(self):
        tau = self.pqr_tau.clamp_min(self.dt)
        decay = torch.exp(-self.dt / tau)
        noise = self.pqr_sigma * torch.sqrt((1.0 - decay * decay).clamp_min(0.0))
        self.gust_pqr_body = (
            decay * self.gust_pqr_body
            + noise * torch.randn((self.n, 3), device=self.device)
        )

    def _step_external_pqr(self, target_pqr):
        target_pqr = torch.as_tensor(target_pqr, dtype=torch.float32, device=self.device)
        if target_pqr.ndim == 1:
            target_pqr = target_pqr.reshape(1, 3).expand(self.n, 3)
        tau = self._effective_external_tau(self.pqr_tau)
        decay = torch.exp(-self.dt / tau)
        self.gust_pqr_body = (
            decay * self.gust_pqr_body
            + (1.0 - decay) * target_pqr
        )

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
