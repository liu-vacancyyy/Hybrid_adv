import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from reward_function_base import BaseRewardFunction


class SimulinkReward(BaseRewardFunction):
    """
    Reward for fast, overshoot-free tracking — V4 (fast settling).

    Key insight: ALL reward signals use RELATIVE error (error / step_magnitude).

    Settling time problem in V3:
      Huber quadratic zone kills gradient near target → policy stalls at 2-5% rel.
      Step-function band bonus has zero gradient → no signal between 1% and 2%.

    V4 fix: remove Huber, use pure linear tracking + exponential precision.
      Linear tracking: constant gradient -3.0 at ALL distances  → never slows down
      Exponential precision: strong pull toward zero, peak +2.0  → dominates near target
      Together: policy drives straight to target with uniformly strong signal.
      Damping prevents overshoot even with continuous "go faster" gradient.

    Components:
      1. In-band bonus:    +1.5/step when |error| <= 1% of step (still useful as milestone)
      2. Tracking (linear): -3.0 * rel_error (constant gradient, no dead zone)
      3. Precision (exp):  +2.0 * exp(-30*rel) (smooth, strongest near target)
                           gradient = 60*exp(-30*rel): at rel=0.02 → 33 (!), rel=0.1 → 3
      4. Overshoot:        -10 * overshoot_ratio
      5. Adaptive damping: -1.5 * q² * exp(-10*rel) (stronger to counteract #2+#3)
      6. Integral penalty:  -0.2 * |integral_error|

    Reward profile (rel distance → per-step reward, no overshoot, q≈0):
      rel=0.00:  0.0 + 2.00 + 1.50 = +3.50  (in-band, peak)
      rel=0.01: -0.03 + 1.48 + 1.50 = +2.95  (1% band edge)
      rel=0.02: -0.06 + 1.10 + 0.00 = +1.04  (2% band, still strong!)
      rel=0.05: -0.15 + 0.45 + 0.00 = +0.30  (positive!  → drives inward)
      rel=0.10: -0.30 + 0.10 + 0.00 = -0.20
      rel=0.50: -1.50 + 0.00 + 0.00 = -1.50
      rel=1.00: -3.00 + 0.00 + 0.00 = -3.00

    Settling budget: at rel=0.02 (just outside 2% band), reward = +1.04/step
      → vs rel=0.01: +2.95.  Gap = 1.91/step.  10 extra steps at 2% = -19 lost.
      → Strong incentive to push from 2% → 1% ASAP.
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env):
        theta = env.model.s[:, 3]
        q = env.model.s[:, 2]

        delta_theta = theta - task.target_theta
        error = torch.abs(delta_theta)

        # Relative error: 0 at target, ~1 at episode start
        rel_error = error / task.step_magnitude

        # ---- 1. Tracking: pure linear on relative error ----
        # Constant gradient -3.0 everywhere → no slowdown near target
        # rel=0→0, 0.1→-0.3, 0.5→-1.5, 1.0→-3.0
        reward_tracking = -3.0 * rel_error

        # ---- 2. Precision: smooth exponential pull toward target ----
        # Merges old in-band bonus (+1.5) and precision (+2.0) into single smooth term.
        # Eliminates discontinuous jump at 1% band boundary.
        # rel=0→+3.5, 0.01→+2.59, 0.02→+1.92, 0.05→+0.78, 0.1→+0.17, 0.5→+0.0
        # Gradient: d/d(rel) = -105*exp(-30*rel)
        #   at rel=0.02: -57,  at rel=0.05: -22,  at rel=0.1: -5
        reward_precision = 3.5 * torch.exp(-30.0 * rel_error)

        # ---- 3. Overshoot penalty: dead-zone at 2%, then -10/unit ----
        # Allow ≤2% overshoot for free — slight overshoot helps settle faster.
        #   1% → 0,  2% → 0,  3% → -0.1,  5% → -0.3,  10% → -0.8
        overshoot = (theta - task.target_theta) * torch.sign(task.target_theta - task.initial_theta)
        overshoot_amount = torch.clamp(overshoot, min=0.0)
        overshoot_ratio = overshoot_amount / task.step_magnitude
        excess_ratio = torch.clamp(overshoot_ratio - 0.02, min=0.0)
        reward_overshoot = -10.0 * excess_ratio

        # ---- 4. Adaptive damping: stronger to counteract aggressive tracking+precision ----
        # Without strong damping, the continuous tracking gradient → overshoot
        proximity = torch.exp(-10.0 * rel_error)
        reward_damping = -1.5 * q ** 2 * proximity

        # ---- 5. Integral penalty: fight persistent steady-state offset ----
        # Normalize by step_magnitude to keep penalty scale-invariant
        reward_integral = -0.2 * torch.abs(task.integral_error) / task.step_magnitude

        reward = (reward_tracking + reward_precision +
                  reward_overshoot + reward_damping + reward_integral)

        # Debug: print reward components for first 3 agents
        n_print = min(3, error.shape[0])
        for i in range(n_print):
            print(f"[Agent {i}] rel={rel_error[i].item():.4f} "
                  f"trk={reward_tracking[i].item():+.4f} "
                  f"prec={reward_precision[i].item():+.4f} "
                  f"ovsh={reward_overshoot[i].item():+.4f} "
                  f"damp={reward_damping[i].item():+.4f} "
                  f"int={reward_integral[i].item():+.4f} "
                  f"| Σ={reward[i].item():+.4f}")

        return reward
    
    # def get_reward_old(self, task, env):
    #     theta = env.model.s[:, 3]       # current pitch angle
    #     q = env.model.s[:, 2]           # pitch rate
    #     control = env.model.u[:, 0]     # control input

    #     delta_theta = theta - task.target_theta
    #     error = torch.abs(delta_theta)
    #     # error=0.1→-0.30, 0.05→-0.15, 0.01→-0.03, 0→0
    #     reward_theta = -3.0 * error

    #     # ---- 2. Exponential precision bonus: steep gradient near zero ----
    #     # k=100 → half-max at 0.007 rad, focuses all gradient on last 1% of approach
    #     # error=0→+0.50, 0.002→+0.41, 0.005→+0.30, 0.01→+0.18, 0.05→+0.003
    #     reward_precision = 0.5 * torch.exp(-100.0 * error)

    #     # ---- 3. Pitch rate penalty: gentle smoothness ----
    #     # q=0→0, q=1→-0.01, q=3→-0.09
    #     reward_q = -0.01 * q ** 2

    #     # ---- 4. Overshoot penalty (percentage-based, any overshoot) ----
    #     # Penalize as soon as theta crosses target in the step direction
    #     # Uses overshoot/step_magnitude so 10% overshoot gets same penalty regardless of step size
    #     # step=0.1, overshoot=0.01 (10%) → -0.5;  step=0.01, overshoot=0.001 (10%) → -0.5
    #     overshoot = (theta - task.target_theta) * torch.sign(task.target_theta - task.initial_theta)
    #     overshoot_amount = torch.clamp(overshoot, min=0.0)
    #     overshoot_ratio = overshoot_amount / (task.step_magnitude + 1e-8)
    #     reward_overshoot = -5.0 * overshoot_ratio

    #     # ---- 5. Saturation penalty: mild discouragement ----
    #     is_saturated = torch.abs(control) >= 0.95 * task.control_limit
    #     reward_saturation = -0.1 * is_saturated.float()

    #     # ---- 6. Two-tier settling band (relative to step_magnitude) ----
    #     # Scale precision requirement with step difficulty
    #     # Outer: 10% of step (min 0.002), Inner: 2% of step (min 0.0004)
    #     outer_band = torch.clamp(0.10 * task.step_magnitude, min=0.002)
    #     inner_band = torch.clamp(0.02 * task.step_magnitude, min=0.0004)

    #     in_outer = (error <= outer_band).float()
    #     in_inner = (error <= inner_band).float()

    #     # Outer: error=0→+0.20, edge→0
    #     reward_outer = in_outer * 0.2 * (outer_band - error) / (outer_band + 1e-8)
    #     # Inner: error=0→+0.60, edge→0
    #     reward_inner = in_inner * 0.6 * (inner_band - error) / (inner_band + 1e-8)

    #     reward_settle = reward_outer + reward_inner

    #     # ---- 7. Integral error penalty: fight steady-state offset ----
    #     # integral clamped to ±0.5, so max penalty = -0.10/step
    #     reward_integral = -0.2 * torch.abs(task.integral_error)

    #     # Debug: print reward components for first 10 agents
    #     n_print = min(10, reward_theta.shape[0])
    #     for i in range(n_print):
    #         total_i = (reward_theta[i] + reward_precision[i] + reward_q[i] +
    #                     reward_overshoot[i] + reward_saturation[i] +
    #                     reward_settle[i] + reward_integral[i]).item()
    #         print(f"[Agent {i}] "
    #               f"θ={reward_theta[i].item():+.4f} "
    #               f"prec={reward_precision[i].item():+.4f} "
    #               f"q={reward_q[i].item():+.4f} "
    #               f"ovsh={reward_overshoot[i].item():+.4f} "
    #               f"sat={reward_saturation[i].item():+.4f} "
    #               f"sett={reward_settle[i].item():+.4f} "
    #               f"int={reward_integral[i].item():+.4f} "
    #               f"| Σ={total_i:+.4f}")

    #     return (reward_theta + reward_precision + reward_q +
    #             reward_overshoot + reward_saturation + reward_settle + reward_integral)
