import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from reward_functions.velocity_hover_reward import (
    VelocityHoverEventReward,
    VelocityHoverReward,
)
from hybrid_termination_conditions.low_altitude import LowAltitude
from hybrid_termination_conditions.extreme_angle import ExtremeAngle
from hybrid_termination_conditions.extreme_omega import ExtremeOmega
from hybrid_termination_conditions.extreme_state import ExtremeState
from hybrid_termination_conditions.high_speed import HighSpeed
from termination_conditions.hover_timeout_done import HoverTimeoutDone
from utils.utils import wrap_PI


class VelocityHoverTask(BaseTask):
    """Track upper-level velocity commands while hovering.

    This task is intended for the first Gazebo hover policy. The commanded
    velocity defaults to vx=vy=vz=0 and the reset randomization is handled by
    the model config. No position observation or position reward is used.

    Observation layout when command channels are enabled (default, dim=27):
        0-2.  velocity error vx/vy/vz normalized by command limits
        3-5.  commanded vx/vy/vz normalized by command limits
        6.    heading hold error / pi
        7-10. roll sin/cos, pitch sin/cos
        11.   vt / velocity_hover_vt_norm
        12-14. measured vx/vy/vz normalized by command limits
        15-17. body angular rates p/q/r
        18-21. alpha sin/cos, beta sin/cos
        22-26. five motor thrust observations / hover_force_norm
    """

    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.task_name = 'velocity_hover'

        self.target_vx_value = float(getattr(config, 'velocity_hover_target_vx', 0.0))
        self.target_vy_value = float(getattr(config, 'velocity_hover_target_vy', 0.0))
        self.target_vz_value = float(getattr(config, 'velocity_hover_target_vz', 0.0))
        self.vx_norm = float(getattr(config, 'velocity_hover_vx_norm', 1.0))
        self.vy_norm = float(getattr(config, 'velocity_hover_vy_norm', 1.0))
        self.vz_norm = float(getattr(config, 'velocity_hover_vz_norm', 1.0))
        self.vt_norm = float(getattr(config, 'velocity_hover_vt_norm', 5.0))
        self.force_norm = float(getattr(config, 'hover_force_norm', 15.0))
        self.include_command_obs = bool(
            getattr(config, 'velocity_hover_include_command_obs', True)
        )

        self.enable_sensor_noise = bool(getattr(config, 'enable_sensor_noise', True))
        self.sensor_vel_std = float(getattr(config, 'sensor_vel_std', 0.05))
        self.sensor_att_std = float(getattr(config, 'sensor_att_std', 0.003))
        self.sensor_omega_std = float(getattr(config, 'sensor_omega_std', 0.0003394))

        expected_obs = 27 if self.include_command_obs else 24
        if self.num_observation != expected_obs:
            self.num_observation = expected_obs
            self.load_observation_space()

        self.target_vx = torch.zeros(self.n, device=self.device)
        self.target_vy = torch.zeros(self.n, device=self.device)
        self.target_vz = torch.zeros(self.n, device=self.device)
        self.target_heading = torch.zeros(self.n, device=self.device)

        self.reward_functions = [
            VelocityHoverReward(self.config),
            VelocityHoverEventReward(self.config),
        ]
        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeAngle(self.config),
            ExtremeOmega(self.config),
            ExtremeState(self.config),
            HighSpeed(self.config),
            HoverTimeoutDone(self.config),
        ]

    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = done | bad_done | exceed_time_limit
        size = int(torch.sum(reset).item())
        if size == 0:
            return

        _, _, heading = env.model.get_posture()
        self.target_vx[reset] = self.target_vx_value
        self.target_vy[reset] = self.target_vy_value
        self.target_vz[reset] = self.target_vz_value
        self.target_heading[reset] = heading[reset].clone()

    def step(self, env):
        pass

    def _apply_sensor_noise(self, roll, pitch, heading, vx, vy, vz, p, q, r):
        roll = wrap_PI(roll + torch.randn_like(roll) * self.sensor_att_std)
        pitch = wrap_PI(pitch + torch.randn_like(pitch) * self.sensor_att_std)
        heading = wrap_PI(heading + torch.randn_like(heading) * self.sensor_att_std)
        vx = vx + torch.randn_like(vx) * self.sensor_vel_std
        vy = vy + torch.randn_like(vy) * self.sensor_vel_std
        vz = vz + torch.randn_like(vz) * self.sensor_vel_std
        p = p + torch.randn_like(p) * self.sensor_omega_std
        q = q + torch.randn_like(q) * self.sensor_omega_std
        r = r + torch.randn_like(r) * self.sensor_omega_std
        return roll, pitch, heading, vx, vy, vz, p, q, r

    def _build_obs(self, env, add_sensor_noise):
        roll, pitch, heading = env.model.get_posture()
        vt = env.model.get_vt()
        vx, vy = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()
        p, q, r = env.model.get_angular_velocity()
        sa, ca, sb, cb = env.model.get_aero_sincos()
        f0, f1, f2, f3, f4 = env.model.get_F()

        if add_sensor_noise:
            roll, pitch, heading, vx, vy, vz, p, q, r = self._apply_sensor_noise(
                roll, pitch, heading, vx, vy, vz, p, q, r
            )
            vt = torch.sqrt((vx * vx + vy * vy + vz * vz).clamp_min(0.0))

        nvx = max(self.vx_norm, 1e-6)
        nvy = max(self.vy_norm, 1e-6)
        nvz = max(self.vz_norm, 1e-6)
        delta_yaw = wrap_PI(self.target_heading - heading).reshape(-1, 1) / torch.pi

        obs_parts = [
            ((vx - self.target_vx) / nvx).reshape(-1, 1),
            ((vy - self.target_vy) / nvy).reshape(-1, 1),
            ((vz - self.target_vz) / nvz).reshape(-1, 1),
        ]
        if self.include_command_obs:
            obs_parts.extend([
                (self.target_vx / nvx).reshape(-1, 1),
                (self.target_vy / nvy).reshape(-1, 1),
                (self.target_vz / nvz).reshape(-1, 1),
            ])

        obs_parts.extend([
            delta_yaw,
            torch.sin(roll).reshape(-1, 1),
            torch.cos(roll).reshape(-1, 1),
            torch.sin(pitch).reshape(-1, 1),
            torch.cos(pitch).reshape(-1, 1),
            (vt / max(self.vt_norm, 1e-6)).reshape(-1, 1),
            (vx / nvx).reshape(-1, 1),
            (vy / nvy).reshape(-1, 1),
            (vz / nvz).reshape(-1, 1),
            p.reshape(-1, 1),
            q.reshape(-1, 1),
            r.reshape(-1, 1),
            sa.reshape(-1, 1),
            ca.reshape(-1, 1),
            sb.reshape(-1, 1),
            cb.reshape(-1, 1),
            (f0 / self.force_norm).reshape(-1, 1),
            (f1 / self.force_norm).reshape(-1, 1),
            (f2 / self.force_norm).reshape(-1, 1),
            (f3 / self.force_norm).reshape(-1, 1),
            (f4 / self.force_norm).reshape(-1, 1),
        ])
        return torch.hstack(tuple(obs_parts))

    def get_obs(self, env):
        return self._build_obs(env, add_sensor_noise=self.enable_sensor_noise)

    def get_clean_obs(self, env):
        return self._build_obs(env, add_sensor_noise=False)
