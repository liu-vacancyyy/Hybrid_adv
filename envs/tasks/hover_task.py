import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from reward_functions.hover_reward import HoverReward
from reward_functions.hover_event_driven_reward import HoverEventDrivenReward
from hybrid_termination_conditions.low_altitude import LowAltitude
from hybrid_termination_conditions.extreme_angle import ExtremeAngle
from hybrid_termination_conditions.extreme_omega import ExtremeOmega
from hybrid_termination_conditions.high_speed import HighSpeed
from termination_conditions.hover_timeout_done import HoverTimeoutDone
from utils.utils import wrap_PI

# Position observations are expressed as the displacement FROM the aircraft TO
# the target ("delta to target") in the world horizontal plane and vertical
# axis. We assume the platform can sense these deltas (vision / optical flow /
# motion capture / GPS with a known target) even though it cannot measure its
# own absolute (npos, epos).
HOVER_POS_NORM = 5.0   # m
HOVER_ALT_NORM = 2.0     # m


class HoverTask(BaseTask):
    """Hover at a fixed (npos, epos, altitude, heading) locked at spawn.

    Observation (dim = 24, all dimensionless):
        0.  delta_n / hover_pos_norm     (target_npos - npos)
        1.  delta_e / hover_pos_norm
        2.  delta_alt / hover_alt_norm
        3.  wrap_PI(delta_yaw) / pi
        4-5.  roll  sin, cos
        6-7.  pitch sin, cos
        8.   vt / 10
        9.   vx / 5
        10.  vy / 5
        11.  vz / 5
        12.  P
        13.  Q
        14.  R
        15-16. alpha sin, cos
        17-18. beta  sin, cos
        19-23. F_head, F_rf, F_lb, F_lf, F_rb (all / 7N)
    """

    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.task_name = 'hover'

        # Targets locked at spawn
        self.target_npos     = torch.zeros(self.n, device=self.device)
        self.target_epos     = torch.zeros(self.n, device=self.device)
        self.target_altitude = torch.zeros(self.n, device=self.device)
        self.target_heading  = torch.zeros(self.n, device=self.device)

        self.noise_scale = getattr(config, 'noise_scale', 0.01)
        self.pos_norm = float(getattr(config, 'hover_pos_norm', HOVER_POS_NORM))
        self.alt_norm = float(getattr(config, 'hover_alt_norm', HOVER_ALT_NORM))

        # ---- Sensor noise (zero-mean Gaussian, NO bias) ----
        # Applied only inside _build_obs(env, add_sensor_noise=True).  The
        # privileged expert calls get_clean_obs(env) and bypasses noise.
        self.enable_sensor_noise = getattr(config, 'enable_sensor_noise', True)
        self.sensor_pos_std   = float(getattr(config, 'sensor_pos_std',   1.0))
        self.sensor_vel_std   = float(getattr(config, 'sensor_vel_std',   0.05))
        self.sensor_att_std   = float(getattr(config, 'sensor_att_std',   0.005))
        self.sensor_omega_std = float(getattr(config, 'sensor_omega_std', 0.0005))

        # NOTE: spawn perturbations (attitude / velocity / omega) are applied
        # by HybridModel.reset() following the LearningToFly axis-angle scheme.
        # Do NOT re-perturb here.

        self.reward_functions = [
            HoverReward(self.config),
            HoverEventDrivenReward(self.config),
        ]
        self.termination_conditions = [
            LowAltitude(self.config),
            ExtremeAngle(self.config),
            ExtremeOmega(self.config),
            HighSpeed(self.config),
            HoverTimeoutDone(self.config),
        ]

    # ------------------------------------------------------------------ #
    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = done | bad_done | exceed_time_limit
        size = int(torch.sum(reset).item())
        if size == 0:
            return

        # Lock target = post-perturbation spawn state (== "hold current pose").
        npos, epos, altitude = env.model.get_position()
        _, _, heading = env.model.get_posture()
        self.target_npos[reset]     = npos[reset].clone()
        self.target_epos[reset]     = epos[reset].clone()
        self.target_altitude[reset] = altitude[reset].clone()
        self.target_heading[reset]  = heading[reset].clone()

    def step(self, env):
        pass

    # ------------------------------------------------------------------ #
    def _apply_sensor_noise(self, roll, pitch, heading, vx, vy, vz, P, Q, R):
        """Zero-mean Gaussian white noise on attitude / lin-vel / ang-rate."""
        roll    = wrap_PI(roll    + torch.randn_like(roll)    * self.sensor_att_std)
        pitch   = wrap_PI(pitch   + torch.randn_like(pitch)   * self.sensor_att_std)
        heading = wrap_PI(heading + torch.randn_like(heading) * self.sensor_att_std)
        vx = vx + torch.randn_like(vx) * self.sensor_vel_std
        vy = vy + torch.randn_like(vy) * self.sensor_vel_std
        vz = vz + torch.randn_like(vz) * self.sensor_vel_std
        P  = P  + torch.randn_like(P)  * self.sensor_omega_std
        Q  = Q  + torch.randn_like(Q)  * self.sensor_omega_std
        R  = R  + torch.randn_like(R)  * self.sensor_omega_std
        return roll, pitch, heading, vx, vy, vz, P, Q, R

    def _build_obs(self, env, add_sensor_noise):
        npos, epos, altitude   = env.model.get_position()
        roll, pitch, heading   = env.model.get_posture()       # clean ground truth
        vt                     = env.model.get_vt()
        vx, vy                 = env.model.get_ground_speed()  # clean ground truth
        vz                     = env.model.get_climb_rate()    # clean ground truth
        P, Q, R                = env.model.get_angular_velocity()  # clean ground truth
        sa, ca, sb, cb         = env.model.get_aero_sincos()   # no atan2, bounded [-1,1]
        F_head, F_rf, F_lb, F_lf, F_rb = env.model.get_F()

        # Zero-mean Gaussian sensor noise on the observation path only.
        # Position noise is added to (npos, epos, altitude) before computing
        # delta-to-target (target is constant, so this equals additive noise
        # on delta).
        if add_sensor_noise:
            npos     = npos     + torch.randn_like(npos)     * self.sensor_pos_std
            epos     = epos     + torch.randn_like(epos)     * self.sensor_pos_std
            altitude = altitude + torch.randn_like(altitude) * self.sensor_pos_std
            roll, pitch, heading, vx, vy, vz, P, Q, R = self._apply_sensor_noise(
                roll, pitch, heading, vx, vy, vz, P, Q, R)
            vt = torch.sqrt((vx * vx + vy * vy + vz * vz).clamp_min(0.0))

        delta_n   = (self.target_npos     - npos     ).reshape(-1, 1) / self.pos_norm
        delta_e   = (self.target_epos     - epos     ).reshape(-1, 1) / self.pos_norm
        delta_alt = (self.target_altitude - altitude ).reshape(-1, 1) / self.alt_norm
        delta_yaw = wrap_PI((self.target_heading - heading).reshape(-1, 1)) / torch.pi

        obs = torch.hstack((
            delta_n, delta_e, delta_alt, delta_yaw,
            torch.sin(roll).reshape(-1, 1),  torch.cos(roll).reshape(-1, 1),
            torch.sin(pitch).reshape(-1, 1), torch.cos(pitch).reshape(-1, 1),
            (vt / 10.0).reshape(-1, 1),
            (vx / 5.0).reshape(-1, 1),
            (vy / 5.0).reshape(-1, 1),
            (vz / 5.0).reshape(-1, 1),
            P.reshape(-1, 1), Q.reshape(-1, 1), R.reshape(-1, 1),
            sa.reshape(-1, 1), ca.reshape(-1, 1),
            sb.reshape(-1, 1), cb.reshape(-1, 1),
            (F_head / 7.0).reshape(-1, 1),
            (F_rf   / 7.0).reshape(-1, 1),
            (F_lb   / 7.0).reshape(-1, 1),
            (F_lf   / 7.0).reshape(-1, 1),
            (F_rb   / 7.0).reshape(-1, 1),
        ))
        return obs

    def get_obs(self, env):
        return self._build_obs(env, add_sensor_noise=self.enable_sensor_noise)

    def get_clean_obs(self, env):
        return self._build_obs(env, add_sensor_noise=False)
