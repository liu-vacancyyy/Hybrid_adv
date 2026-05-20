import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from reward_functions.rc_reward import RCReward
from reward_functions.rc_event_driven_reward import RCEventDrivenReward
from hybrid_termination_conditions.low_altitude import LowAltitude
from hybrid_termination_conditions.extreme_angle import ExtremeAngle
from hybrid_termination_conditions.extreme_omega import ExtremeOmega
from hybrid_termination_conditions.overload import Overload
from hybrid_termination_conditions.high_speed import HighSpeed
from hybrid_termination_conditions.extreme_state import ExtremeState
from termination_conditions.unreach_rc import UnreachRC
from utils.utils import wrap_PI


class RCTask(BaseTask):
    '''
    Control target heading with control surface
    '''
    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)

        self.task_name = 'rc'
        # ----- Tracked targets (per env) -----
        self.target_heading = torch.zeros(self.n, device=self.device)
        self.target_vx      = torch.zeros(self.n, device=self.device)
        self.target_vz      = torch.zeros(self.n, device=self.device)
        # Internal yaw-rate state: humans push the rudder stick which produces
        # a yaw-rate command; heading is then the integral of that command.
        self.target_yaw_rate = torch.zeros(self.n, device=self.device)

        self.noise_scale = getattr(self.config, 'noise_scale', 0.01)
        self.add_noise_type = 'no_add' #IMU_noise, AOAAOS_noise, EAS_noise, altitude_noise
        self.IMU_noise_scale = 0 #max:2
        self.AOAAOS_noise_scale = 0 #max:2
        self.EAS_noise_scale = 0 #max:2
        self.altitude_noise_scale = 0 #max:100

        # ============================================================
        # Human-like RC command generator (per-env vectorised)
        #
        # For each tracked channel (vx, vz, yaw_rate) we run an Ornstein-
        # Uhlenbeck process pulled toward a *piecewise-constant* mean ``mu``.
        # ``mu`` is resampled every ``dwell ~ U(dwell_min, dwell_max)`` steps
        # to mimic a pilot "setting" a stick position, holding for a while,
        # then changing it.  ``heading`` is then the integral of
        # ``target_yaw_rate`` (with PI wrapping).
        # ============================================================
        self._dt = float(getattr(self.config, 'dt', 0.02))
        self.theta_vel  = float(getattr(self.config, 'rc_ou_theta_vel',  0.6))
        self.theta_yaw  = float(getattr(self.config, 'rc_ou_theta_yaw',  0.8))
        self.sigma_vx   = float(getattr(self.config, 'rc_ou_sigma_vx',   0.4))
        self.sigma_vz   = float(getattr(self.config, 'rc_ou_sigma_vz',   0.3))
        self.sigma_yawr = float(getattr(self.config, 'rc_ou_sigma_yawr', 0.4))
        self.max_vx       = float(getattr(self.config, 'rc_max_vx',       2.5))
        self.max_vz       = float(getattr(self.config, 'rc_max_vz',       2.0))
        self.max_yaw_rate = float(getattr(self.config, 'rc_max_yaw_rate', 0.6))    # rad/s
        self.mu_vx_range       = float(getattr(self.config, 'rc_mu_vx_range',       2.5))
        self.mu_vz_range       = float(getattr(self.config, 'rc_mu_vz_range',       1.5))
        self.mu_yaw_rate_range = float(getattr(self.config, 'rc_mu_yaw_rate_range', 0.4))
        self.dwell_min = int(getattr(self.config, 'rc_dwell_min_steps', 100))   # 2.0s @ 50Hz
        self.dwell_max = int(getattr(self.config, 'rc_dwell_max_steps', 400))   # 8.0s @ 50Hz
        self.alt_high  = float(getattr(self.config, 'rc_alt_high', 95.0))
        self.alt_low   = float(getattr(self.config, 'rc_alt_low',   5.0))

        # Per-env OU mean for each channel; resampled when dwell counter hits 0.
        self.mu_vx       = torch.zeros(self.n, device=self.device)
        self.mu_vz       = torch.zeros(self.n, device=self.device)
        self.mu_yaw_rate = torch.zeros(self.n, device=self.device)
        # Steps remaining until next mu resample, per env.
        self.dwell_left  = torch.zeros(self.n, dtype=torch.long, device=self.device)

        self.reward_functions = [
            RCReward(self.config),
            RCEventDrivenReward(self.config),
        ]
        
        self.termination_conditions = [
            Overload(self.config),
            LowAltitude(self.config),
            HighSpeed(self.config),
            ExtremeAngle(self.config),
            ExtremeOmega(self.config),
            ExtremeState(self.config),
            # Timeout(self.config),
            UnreachRC(self.config, device)
        ]

    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = (done | bad_done) | exceed_time_limit
        size = int(torch.sum(reset).item())
        if size == 0:
            return

        vx, _vy = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()
        _roll, _pitch, heading = env.model.get_posture()

        # Initial command = current state -> step 0 has zero tracking error
        self.target_vx[reset]      = vx[reset]
        self.target_vz[reset]      = vz[reset]
        self.target_heading[reset] = heading[reset]
        self.target_yaw_rate[reset] = 0.0
        # Force an immediate mu resample on the next step() call.
        self.dwell_left[reset]     = 0
        self.mu_vx[reset]          = 0.0
        self.mu_vz[reset]          = 0.0
        self.mu_yaw_rate[reset]    = 0.0

    # ---- helpers --------------------------------------------------------
    def _resample_mu(self, mask):
        """Sample new piecewise-constant OU means + dwell counters for envs in ``mask``."""
        size = int(mask.sum().item())
        if size == 0:
            return
        d = self.device
        self.mu_vx[mask]       = (torch.rand(size, device=d) * 2 - 1) * self.mu_vx_range
        self.mu_vz[mask]       = (torch.rand(size, device=d) * 2 - 1) * self.mu_vz_range
        self.mu_yaw_rate[mask] = (torch.rand(size, device=d) * 2 - 1) * self.mu_yaw_rate_range
        self.dwell_left[mask]  = torch.randint(
            self.dwell_min, self.dwell_max + 1, (size,), device=d, dtype=torch.long)

    def step(self, env):
        """Advance the human-like OU command generator by one dt.

        - vx, vz: OU(theta_vel, mu_vx/vz, sigma_vx/vz) with piecewise mu
        - yaw_rate: OU(theta_yaw, mu_yaw_rate, sigma_yawr) with piecewise mu;
                    heading = wrap_PI(heading + yaw_rate * dt)
        - mu_* are resampled every ``dwell ~ U(dwell_min, dwell_max)`` steps
        - vz target is suppressed when within altitude soft fence
        """
        dt = self._dt
        sqdt = float(dt) ** 0.5

        # 1) Resample mu where the dwell timer hit zero
        resample_mask = self.dwell_left <= 0
        self._resample_mu(resample_mask)
        # Decrement the dwell timer (clamp at 0 to avoid underflow).
        self.dwell_left = torch.clamp(self.dwell_left - 1, min=0)

        # 2) OU step on each command channel
        n = self.target_vx.shape[0]
        d = self.device
        eps_vx   = torch.randn(n, device=d)
        eps_vz   = torch.randn(n, device=d)
        eps_yawr = torch.randn(n, device=d)

        self.target_vx += (self.theta_vel * (self.mu_vx - self.target_vx) * dt
                           + self.sigma_vx * sqdt * eps_vx)
        self.target_vz += (self.theta_vel * (self.mu_vz - self.target_vz) * dt
                           + self.sigma_vz * sqdt * eps_vz)
        self.target_yaw_rate += (self.theta_yaw * (self.mu_yaw_rate - self.target_yaw_rate) * dt
                                 + self.sigma_yawr * sqdt * eps_yawr)

        # 3) Saturate stick deflections
        self.target_vx       = torch.clamp(self.target_vx,       -self.max_vx,       self.max_vx)
        self.target_vz       = torch.clamp(self.target_vz,       -self.max_vz,       self.max_vz)
        self.target_yaw_rate = torch.clamp(self.target_yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        # 4) Soft altitude fence: stop pushing vz further away from the safe band
        _npos, _epos, altitude = env.model.get_position()
        high_mask = altitude > self.alt_high
        low_mask  = altitude < self.alt_low
        self.target_vz[high_mask & (self.target_vz > 0)] = 0.0
        self.target_vz[low_mask  & (self.target_vz < 0)] = 0.0
        # Also bias the OU mean back inside the band so the next dwell window
        # does not immediately push the platform back outside.
        self.mu_vz[high_mask & (self.mu_vz > 0)] = 0.0
        self.mu_vz[low_mask  & (self.mu_vz < 0)] = 0.0

        # 5) Integrate yaw-rate into heading
        self.target_heading = wrap_PI(self.target_heading + self.target_yaw_rate * dt)
    
    # def get_obs(self, env):
    #     """
    #     Convert simulation states into the format of observation_space.

    #     observation(dim 22):
    #         0. ego_delta_x_acc      (unit: m/ss)
    #         1. ego_delta_z_acc       (unit m/ss)
    #         2. ego_delta_heading            (unit: rad)
    #         3. ego_altitude            (unit: m)
    #         4. ego_roll_sin
    #         5. ego_roll_cos
    #         6. ego_pitch_sin
    #         7. ego_pitch_cos
    #         8. ego_vt                  (unit: m/s)
    #         9. ego_alpha_sin
    #         10. ego_alpha_cos
    #         11. ego_beta_sin
    #         12. ego_beta_cos
    #         13. ego_P                  (unit: rad/s)
    #         14. ego_Q                  (unit: rad/s)
    #         15. ego_R                  (unit: rad/s)
    #         16. ego_hF                  (unit: %)
    #         17. ego_lfF                 (unit: %)
    #         18. ego_rfF                (unit: %)
    #         19. ego_lbF                (unit: %)
    #         20. ego_rbF                (unit: %)
    #         21. EAS2TAS
    #     """
    #     npos, epos, altitude = env.model.get_position()
    #     if(self.add_noise_type == 'altitude_noise'):
    #         altitude += torch.randn_like(altitude) * self.altitude_noise_scale
    #     roll, pitch, heading = env.model.get_posture()
    #     vt = env.model.get_vt()
    #     EAS = env.model.get_EAS()
    #     if(self.add_noise_type == 'EAS_noise'):
    #         vt += torch.randn_like(vt) * self.EAS_noise_scale / 100. * vt
    #         EAS += torch.randn_like(EAS) * self.EAS_noise_scale / 100. * EAS
    #     alpha = env.model.get_AOA()
    #     beta = env.model.get_AOS()
    #     # print('alpha:',alpha[0])
    #     # print('beta:',beta[0])
    #     # if(self.add_noise_type == 'AOAAOS_noise'):
    #     #     alpha += torch.randn_like(alpha) * self.AOAAOS_noise_scale * 180.0 / torch.pi
    #     #     beta += torch.randn_like(beta) * self.AOAAOS_noise_scale * 180.0 / torch.pi
    #     P, Q, R = env.model.get_angular_velocity()
    #     # if(self.add_noise_type == 'IMU_noise'):
    #     #     P += torch.randn_like(P) * self.IMU_noise_scale * torch.pi / 180.0
    #     #     Q += torch.randn_like(Q) * self.IMU_noise_scale * torch.pi / 180.0
    #     #     R += torch.randn_like(R) * self.IMU_noise_scale * torch.pi / 180.0
    #     F1, F2, F3, F4, F5 = env.model.get_F()
    #     eas2tas = env.model.get_EAS2TAS()
    #     ax, ay, az = env.model.get_acceleration()

    #     norm_delta_x_acc = (ax - self.target_x_acc).reshape(-1, 1) / 5
    #     norm_delta_heading = wrap_PI((heading - self.target_heading).reshape(-1, 1)) / torch.pi
    #     norm_delta_z_acc = (az - self.target_z_acc).reshape(-1, 1) / 5
    #     norm_altitude = altitude.reshape(-1, 1) / 100
    #     roll_sin = torch.sin(roll.reshape(-1, 1))
    #     roll_cos = torch.cos(roll.reshape(-1, 1))
    #     pitch_sin = torch.sin(pitch.reshape(-1, 1))
    #     pitch_cos = torch.cos(pitch.reshape(-1, 1))
    #     norm_EAS = EAS.reshape(-1, 1) / 10
    #     alpha_sin = torch.sin(alpha.reshape(-1, 1))
    #     alpha_cos = torch.cos(alpha.reshape(-1, 1))
    #     beta_sin = torch.sin(beta.reshape(-1, 1))
    #     beta_cos = torch.cos(beta.reshape(-1, 1))
    #     norm_P = P.reshape(-1, 1)
    #     norm_Q = Q.reshape(-1, 1)
    #     norm_R = R.reshape(-1, 1)
    #     norm_F1 = F1.reshape(-1, 1) / 7
    #     norm_F2 = F2.reshape(-1, 1) / 7
    #     norm_F3 = F3.reshape(-1, 1) / 7
    #     norm_F4 = F4.reshape(-1, 1) / 7
    #     norm_F5 = F5.reshape(-1, 1) / 7
    #     obs = torch.hstack((norm_delta_x_acc, norm_delta_z_acc))
    #     obs = torch.hstack((obs, norm_delta_heading))
    #     obs = torch.hstack((obs, norm_altitude))
    #     obs = torch.hstack((obs, roll_sin))
    #     obs = torch.hstack((obs, roll_cos))
    #     obs = torch.hstack((obs, pitch_sin))
    #     obs = torch.hstack((obs, pitch_cos))
    #     obs = torch.hstack((obs, norm_EAS))
    #     obs = torch.hstack((obs, alpha_sin))
    #     obs = torch.hstack((obs, alpha_cos))
    #     obs = torch.hstack((obs, beta_sin))
    #     obs = torch.hstack((obs, beta_cos))
    #     obs = torch.hstack((obs, norm_P))
    #     obs = torch.hstack((obs, norm_Q))
    #     obs = torch.hstack((obs, norm_R))
    #     obs = torch.hstack((obs, norm_F1))
    #     obs = torch.hstack((obs, norm_F2))
    #     obs = torch.hstack((obs, norm_F3))
    #     obs = torch.hstack((obs, norm_F4))
    #     obs = torch.hstack((obs, norm_F5))
    #     obs = torch.hstack((obs, eas2tas.reshape(-1, 1)))
    #     noise = torch.randn_like(obs) * self.noise_scale
    #     if(self.add_noise_type == 'IMU_noise'):
    #         noise[:, 13:16] = 0
    #     elif(self.add_noise_type == 'AOAAOS_noise'):
    #         noise[:, 9:13] = 0
    #     elif(self.add_noise_type == 'EAS_noise'):
    #         noise[:, 2] = 0
    #         noise[:, 8] = 0
    #         noise[:, 21] = 0
    #     elif(self.add_noise_type == 'altitude_noise'):
    #         noise[:, 3] = 0
    #     noise_obs = obs + noise
    #     # print(noise_obs[0])
    #     return noise_obs
    
    def get_obs(self, env):
        """
        Convert simulation states into the format of observation_space.

        observation(dim 22):
            0. ego_delta_vx      (unit: m/s)
            1. ego_delta_vz       (unit m/s)
            2. ego_delta_heading            (unit: rad)
            3. ego_altitude            (unit: m)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego_vt                  (unit: m/s)
            9. ego_alpha_sin
            10. ego_alpha_cos
            11. ego_beta_sin
            12. ego_beta_cos
            13. ego_P                  (unit: rad/s)
            14. ego_Q                  (unit: rad/s)
            15. ego_R                  (unit: rad/s)
            16. ego_hF                  (unit: %)
            17. ego_lfF                 (unit: %)
            18. ego_rfF                (unit: %)
            19. ego_lbF                (unit: %)
            20. ego_rbF                (unit: %)
            21. EAS2TAS
        """
        npos, epos, altitude = env.model.get_position()
        if(self.add_noise_type == 'altitude_noise'):
            altitude += torch.randn_like(altitude) * self.altitude_noise_scale
        roll, pitch, heading = env.model.get_posture()
        vt = env.model.get_vt()
        EAS = env.model.get_EAS()
        if(self.add_noise_type == 'EAS_noise'):
            vt += torch.randn_like(vt) * self.EAS_noise_scale / 100. * vt
            EAS += torch.randn_like(EAS) * self.EAS_noise_scale / 100. * EAS
        alpha = env.model.get_AOA()
        beta = env.model.get_AOS()
        # print('alpha:',alpha[0])
        # print('beta:',beta[0])
        # if(self.add_noise_type == 'AOAAOS_noise'):
        #     alpha += torch.randn_like(alpha) * self.AOAAOS_noise_scale * 180.0 / torch.pi
        #     beta += torch.randn_like(beta) * self.AOAAOS_noise_scale * 180.0 / torch.pi
        P, Q, R = env.model.get_angular_velocity()
        # if(self.add_noise_type == 'IMU_noise'):
        #     P += torch.randn_like(P) * self.IMU_noise_scale * torch.pi / 180.0
        #     Q += torch.randn_like(Q) * self.IMU_noise_scale * torch.pi / 180.0
        #     R += torch.randn_like(R) * self.IMU_noise_scale * torch.pi / 180.0
        F1, F2, F3, F4, F5 = env.model.get_F()
        eas2tas = env.model.get_EAS2TAS()
        # ax, ay, az = env.model.get_acceleration()
        vx, vy = env.model.get_ground_speed()
        vz = env.model.get_climb_rate()

        norm_delta_vx = (vx - self.target_vx).reshape(-1, 1) / 5
        norm_delta_heading = wrap_PI((heading - self.target_heading).reshape(-1, 1)) / torch.pi
        norm_delta_vz = (vz - self.target_vz).reshape(-1, 1) / 5
        norm_altitude = altitude.reshape(-1, 1) / 100
        roll_sin = torch.sin(roll.reshape(-1, 1))
        roll_cos = torch.cos(roll.reshape(-1, 1))
        pitch_sin = torch.sin(pitch.reshape(-1, 1))
        pitch_cos = torch.cos(pitch.reshape(-1, 1))
        norm_EAS = EAS.reshape(-1, 1) / 10
        alpha_sin = torch.sin(alpha.reshape(-1, 1))
        alpha_cos = torch.cos(alpha.reshape(-1, 1))
        beta_sin = torch.sin(beta.reshape(-1, 1))
        beta_cos = torch.cos(beta.reshape(-1, 1))
        norm_P = P.reshape(-1, 1)
        norm_Q = Q.reshape(-1, 1)
        norm_R = R.reshape(-1, 1)
        norm_F1 = F1.reshape(-1, 1) / 7
        norm_F2 = F2.reshape(-1, 1) / 7
        norm_F3 = F3.reshape(-1, 1) / 7
        norm_F4 = F4.reshape(-1, 1) / 7
        norm_F5 = F5.reshape(-1, 1) / 7
        obs = torch.hstack((norm_delta_vx, norm_delta_vz))
        obs = torch.hstack((obs, norm_delta_heading))
        obs = torch.hstack((obs, norm_altitude))
        obs = torch.hstack((obs, roll_sin))
        obs = torch.hstack((obs, roll_cos))
        obs = torch.hstack((obs, pitch_sin))
        obs = torch.hstack((obs, pitch_cos))
        obs = torch.hstack((obs, norm_EAS))
        obs = torch.hstack((obs, alpha_sin))
        obs = torch.hstack((obs, alpha_cos))
        obs = torch.hstack((obs, beta_sin))
        obs = torch.hstack((obs, beta_cos))
        obs = torch.hstack((obs, norm_P))
        obs = torch.hstack((obs, norm_Q))
        obs = torch.hstack((obs, norm_R))
        obs = torch.hstack((obs, norm_F1))
        obs = torch.hstack((obs, norm_F2))
        obs = torch.hstack((obs, norm_F3))
        obs = torch.hstack((obs, norm_F4))
        obs = torch.hstack((obs, norm_F5))
        obs = torch.hstack((obs, eas2tas.reshape(-1, 1)))
        noise = torch.randn_like(obs) * self.noise_scale
        if(self.add_noise_type == 'IMU_noise'):
            noise[:, 13:16] = 0
        elif(self.add_noise_type == 'AOAAOS_noise'):
            noise[:, 9:13] = 0
        elif(self.add_noise_type == 'EAS_noise'):
            noise[:, 2] = 0
            noise[:, 8] = 0
            noise[:, 21] = 0
        elif(self.add_noise_type == 'altitude_noise'):
            noise[:, 3] = 0
        noise_obs = obs + noise
        # print(noise_obs[0])
        return noise_obs
