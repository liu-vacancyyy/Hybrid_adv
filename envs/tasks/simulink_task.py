import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from reward_functions.simulink_reward import SimulinkReward
from reward_functions.simulink_event_driven_reward import SimulinkEventDrivenReward
from termination_conditions.simulink_extreme_state import SimulinkExtremeState
from termination_conditions.simulink_divergence import SimulinkDivergence
from termination_conditions.simulink_done import SimulinkDone


class SimulinkTask(BaseTask):
    '''
    Control target angle (theta) for Simulink model.
    Generates random angle targets at each episode reset.

    Simulink model states (dim 4):
        0. u       - longitudinal velocity
        1. w       - vertical velocity
        2. q       - pitch rate
        3. theta   - pitch angle

    Simulink model control (dim 1):
        0. appc    - pitch command
    '''
    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)

        self.task_name = 'simulink'

        # Target angle
        self.target_theta = torch.zeros(self.n, device=self.device)

        # Last tracking error (for reward shaping)
        self.last_delta_theta = torch.zeros(self.n, device=self.device)

        # Target angle range
        self.max_theta_target = getattr(self.config, 'max_theta_target', 2.1)
        self.min_theta_target = getattr(self.config, 'min_theta_target', 1.9)

        # Initial theta at episode start (for overshoot calculation)
        self.initial_theta = torch.zeros(self.n, device=self.device)
        # Step magnitude: |target - initial|
        self.step_magnitude = torch.zeros(self.n, device=self.device)
        # Counter: how many steps theta stays within 2% settling band
        self.settle_count = torch.zeros(self.n, device=self.device)

        # Integral error (accumulated delta_theta * dt) for eliminating steady-state error
        self.integral_error = torch.zeros(self.n, device=self.device)
        self.dt = getattr(self.config, 'dt', 0.005)

        # Control amplitude limit (action is clamped to [-1, 1])
        self.control_limit = 0.35

        # Noise scale for observation
        self.noise_scale = getattr(self.config, 'noise_scale', 0.01)

        # --- PID takeover: when rel_error < 0.5%, switch to PID for rest of episode ---
        self.pid_active = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.pid_threshold = 0.02  # 2% relative error
        # PI-qD parameters (same as standalone PID baseline)
        self.pid_Kp = 5.0
        self.pid_Ki = 0.08
        self.pid_Kd = -2.0
        self.pid_output_limit = 0.35
        self.pid_integral_limit = 0.08
        self.pid_back_calc_coeff = 0.15
        self.pid_integral = torch.zeros(self.n, device=self.device)

        # Reward functions
        self.reward_functions = [
            SimulinkReward(self.config),
            SimulinkEventDrivenReward(self.config),
        ]

        # Termination conditions
        self.termination_conditions = [
            SimulinkExtremeState(self.config),
            SimulinkDivergence(self.config),
            SimulinkDone(self.config),
        ]

    def reset(self, env):
        """
        Reset target angle when episodes end.
        Initialize target_theta near current theta with a random offset.
        """
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = (done | bad_done) | exceed_time_limit
        size = torch.sum(reset)

        # Current pitch angle
        theta = env.model.s[:, 3]

        # Record initial theta before setting new target
        self.initial_theta[reset] = theta[reset]

        # Random target: uniform in [-2, +2], rejecting dead zone (-0.03, 0.03)
        n = int(size.item())
        target = (torch.rand(n, device=self.device) * 2.0 - 1.0) * 2.0
        in_dead_zone = target.abs() < 0.03
        while in_dead_zone.any():
            target[in_dead_zone] = (torch.rand(int(in_dead_zone.sum().item()), device=self.device) * 2.0 - 1.0) * 2.0
            in_dead_zone = target.abs() < 0.03
        self.target_theta[reset] = target

        # Compute step magnitude = |target - initial_theta|, floor at 0.1 rad
        self.step_magnitude[reset] = torch.clamp(
            torch.abs(self.target_theta[reset] - self.initial_theta[reset]),
            min=0.1
        )

        # Reset settle counter
        self.settle_count[reset] = 0.0

        # Reset integral error
        self.integral_error[reset] = 0.0

        # Initialize last tracking error
        self.last_delta_theta[reset] = theta[reset] - self.target_theta[reset]

        # Reset PID takeover state
        self.pid_active[reset] = False
        self.pid_integral[reset] = 0.0

    def step(self, env):
        """
        Target angle stays constant within an episode.
        Track how many steps theta stays within 2% settling band.
        Accumulate integral error for steady-state error elimination.
        """
        theta = env.model.s[:, 3]
        delta_theta = theta - self.target_theta
        error = torch.abs(delta_theta)
        within_band = error <= 0.02 * self.step_magnitude
        # Reset settle_count to 0 if outside 2% band, otherwise increment
        self.settle_count = (self.settle_count + 1.0) * within_band.float()

        # Accumulate integral error with anti-windup clamp
        self.integral_error += delta_theta * self.dt
        self.integral_error = torch.clamp(self.integral_error, -5.0, 5.0)

        # Debug: print settle_count for first 10 agents
        n_print = min(10, self.settle_count.shape[0])
        sc_vals = ', '.join([f'{self.settle_count[i].item():.0f}' for i in range(n_print)])
        print(f"[settle_count] {sc_vals}")

    def get_obs(self, env):
        """
        Convert simulation states into the format of observation_space.

        observation (dim 9):
            0. delta_theta     - angle tracking error (normalized by π)
            1. norm_u          - longitudinal velocity (normalized)
            2. norm_w          - vertical velocity (normalized)
            3. q               - pitch rate
            4. sin(theta)      - pitch angle sine
            5. cos(theta)      - pitch angle cosine
            6. last_control    - recent control input
            7. integral_error  - accumulated error (for steady-state elimination)
            8. step_magnitude  - step size (normalized by 2.0)
        """
        u = env.model.s[:, 0]
        w = env.model.s[:, 1]
        q = env.model.s[:, 2]
        theta = env.model.s[:, 3]
        control = env.model.u[:, 0]

        # Tracking error — normalized by 2.0 (max possible step)
        delta_theta = (theta - self.target_theta).reshape(-1, 1) / 2.0

        # Normalized states
        norm_u = u.reshape(-1, 1) / 100.0
        norm_w = w.reshape(-1, 1) / 10.0
        norm_q = q.reshape(-1, 1)
        theta_sin = torch.sin(theta.reshape(-1, 1))
        theta_cos = torch.cos(theta.reshape(-1, 1))
        norm_control = control.reshape(-1, 1)

        # Integral error (clamped to [-5.0, 5.0] in step())
        norm_integral = self.integral_error.reshape(-1, 1) / 5.0

        # Step magnitude: normalized by 2.0 (max step)
        # Range: [0.05, 1.0] for steps ∈ [0.1, 2.0]
        norm_step_mag = (self.step_magnitude / 2.0).reshape(-1, 1)

        obs = torch.hstack((
            delta_theta,
            norm_u,
            norm_w,
            norm_q,
            theta_sin,
            theta_cos,
            norm_control,
            norm_integral,
            norm_step_mag,
        ))

        # Add observation noise
        # noise = torch.randn_like(obs) * self.noise_scale
        # obs = obs + noise

        return obs

    def maybe_override_action(self, env, action):
        """
        Check relative error; if < 0.5%, activate PID for this agent permanently
        (until episode reset). Return (possibly overridden) action.
        """
        theta = env.model.s[:, 3]
        q = env.model.s[:, 2]
        error = self.target_theta - theta          # signed error
        rel_error = torch.abs(error) / self.step_magnitude

        # Activate PID for agents that just crossed the 0.5% threshold
        newly_active = (~self.pid_active) & (rel_error < self.pid_threshold)
        if newly_active.any():
            self.pid_active[newly_active] = True
            self.pid_integral[newly_active] = 0.0  # fresh PID integral

        if not self.pid_active.any():
            return action

        # Compute PID output for active agents (vectorized)
        P = self.pid_Kp * error
        I = self.pid_Ki * self.pid_integral
        D = self.pid_Kd * q
        output_unsat = P + I + D
        output_sat = torch.clamp(output_unsat, -self.pid_output_limit, self.pid_output_limit)

        # Anti-windup back-calculation
        sat_error = output_sat - output_unsat
        self.pid_integral = self.pid_integral + (error + self.pid_back_calc_coeff * sat_error) * self.dt
        self.pid_integral = torch.clamp(self.pid_integral, -self.pid_integral_limit, self.pid_integral_limit)

        # Convert PID output (physical domain) to action domain [-1, 1]
        pid_action = (output_sat / self.pid_output_limit).reshape_as(action)

        # Replace action only for pid_active agents
        mask = self.pid_active.reshape(-1, 1).expand_as(action)
        action = torch.where(mask, pid_action, action)
        return action
