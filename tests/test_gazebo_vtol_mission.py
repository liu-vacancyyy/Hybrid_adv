import math
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import gym


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'envs'))
sys.path.insert(0, str(REPO_ROOT / 'envs' / 'models'))

from envs.control_env import ControlEnv
from envs.env_wrappers import GPUVecEnv
from models.gazebo_model import GazeboModel
from runner.F16sim_runner import F16SimRunner
from algorithms.utils.act import ACTLayer
from termination_conditions.vtol_mission_termination import VTOLMissionBoundary


def gps_config():
    return SimpleNamespace(
        num_states=12,
        num_controls=8,
        dt=0.02,
        solver='euler',
        gps_enable=True,
        gps_update_rate_hz=10.0,
        gps_horizontal_std=0.0,
        gps_vertical_std=0.0,
        ground_contact_enable=False,
        spawn_mode='air',
        min_altitude=10.0,
        max_altitude=10.0,
        init_roll_range=0.0,
        init_pitch_range=0.0,
        init_yaw_range=0.0,
        init_vel_range=0.0,
        init_omega_range=0.0,
        dr_mass=0.0,
        dr_inertia=0.0,
    )


def reset_model(model):
    flags = torch.ones(model.n, dtype=torch.bool, device=model.device)
    env = SimpleNamespace(
        is_done=flags,
        bad_done=flags,
        exceed_time_limit=flags,
    )
    model.reset(env)


class GazeboVTOLMissionTest(unittest.TestCase):
    def test_continuous_entropy_is_not_divided_by_batch_size(self):
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(8,))
        layer = ACTLayer(
            action_space,
            input_dim=4,
            hidden_size=[],
            activation_id=1,
            gain=0.01,
            action_log_std_init=-1.4,
        )
        features_one = torch.zeros((1, 4))
        action_one, _ = layer(features_one)
        _, entropy_one = layer.evaluate_actions(
            features_one, action_one
        )
        features_batch = torch.zeros((128, 4))
        actions_batch = action_one.expand(128, -1).clone()
        _, entropy_batch = layer.evaluate_actions(
            features_batch, actions_batch
        )

        self.assertAlmostEqual(
            float(entropy_one.mean()), float(entropy_batch.mean()), places=5
        )
        self.assertGreater(float(entropy_batch.mean()), 0.01)

    def test_runner_timeout_is_a_ppo_boundary_and_return_is_counted(self):
        runner = F16SimRunner.__new__(F16SimRunner)
        runner.n_rollout_threads = 2
        runner.num_agents = 1
        runner.envs = SimpleNamespace()
        runner.all_args = SimpleNamespace(use_cost_constraints=False)
        runner._running_episode_returns = torch.zeros(2, 1).numpy()
        runner._begin_rollout_metrics()

        runner._record_rollout_transition(
            [[[1.0]], [[1.0]]],
            [[[False]], [[False]]],
            [[[False]], [[False]]],
            [[[False]], [[False]]],
        )
        runner._record_rollout_transition(
            [[[2.0]], [[3.0]]],
            [[[False]], [[False]]],
            [[[False]], [[False]]],
            [[[True]], [[False]]],
        )
        infos = {}
        runner._append_rollout_train_infos(infos)

        self.assertEqual(infos['rollout/completed_episode_count'], 1)
        self.assertAlmostEqual(infos['average_episode_rewards'], 3.0)
        self.assertEqual(infos['rollout/timeout_count'], 1)
        self.assertEqual(infos['rollout/bad_done_count'], 0)
        self.assertAlmostEqual(infos['rollout/step_reward_mean'], 1.75)
        self.assertAlmostEqual(float(runner._running_episode_returns[1, 0]), 4.0)

    def test_runner_records_bad_done_reason_and_phase(self):
        runner = F16SimRunner.__new__(F16SimRunner)
        runner.n_rollout_threads = 3
        runner.num_agents = 1
        runner.envs = SimpleNamespace(
            gpu_vec_env=SimpleNamespace(
                task=SimpleNamespace(PHASE_NAMES=(
                    'takeoff', 'rotor_climb', 'transition', 'fixed_wing',
                    'back_transition', 'vertical_landing',
                ))
            )
        )
        runner.all_args = SimpleNamespace(use_cost_constraints=False)
        runner._running_episode_returns = torch.zeros(3, 1).numpy()
        runner._begin_rollout_metrics()

        runner._record_rollout_transition(
            [[[0.0]], [[0.0]], [[0.0]]],
            [[[False]], [[False]], [[False]]],
            [[[True]], [[True]], [[False]]],
            [[[False]], [[False]], [[False]]],
            {
                'extreme_angle': torch.tensor([True, False, True]),
                'mission_off_route': torch.tensor([False, True, False]),
                'mission_phase': torch.tensor([3, 4, 2]),
            },
        )
        infos = {}
        runner._append_rollout_train_infos(infos)

        self.assertEqual(infos['termination/extreme_angle'], 1)
        self.assertEqual(infos['termination/mission_off_route'], 1)
        self.assertEqual(infos['termination/phase_fixed_wing'], 1)
        self.assertEqual(infos['termination/phase_back_transition'], 1)
        self.assertNotIn('termination/phase_transition', infos)

    def test_runner_masks_done_bad_done_and_timeout(self):
        runner = F16SimRunner.__new__(F16SimRunner)
        runner.n_rollout_threads = 4
        runner.num_agents = 1
        runner.all_args = SimpleNamespace(use_cost_constraints=False)

        class BufferCapture:
            def insert(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        runner.buffer = BufferCapture()
        shape = (4, 1, 1)
        data = (
            torch.zeros((4, 1, 45)).numpy(),
            torch.zeros((4, 1, 64)).numpy(),
            torch.zeros((4, 1, 8)).numpy(),
            torch.zeros(shape).numpy(),
            torch.tensor([[[True]], [[False]], [[False]], [[False]]]).numpy(),
            torch.tensor([[[False]], [[True]], [[False]], [[False]]]).numpy(),
            torch.tensor([[[False]], [[False]], [[True]], [[False]]]).numpy(),
            torch.zeros(shape).numpy(),
            torch.zeros(shape).numpy(),
            torch.ones((4, 1, 1, 4)).numpy(),
            torch.ones((4, 1, 1, 4)).numpy(),
            None, None, None,
        )
        runner.insert(data)
        masks = runner.buffer.args[3]
        bad_masks = runner.buffer.args[8]
        np.testing.assert_array_equal(
            masks[:, 0, 0], [0.0, 0.0, 0.0, 1.0]
        )
        np.testing.assert_array_equal(
            bad_masks[:, 0, 0], [1.0, 0.0, 1.0, 1.0]
        )
        self.assertTrue(np.all(data[9][:3] == 0.0))
        self.assertTrue(np.all(data[10][:3] == 0.0))
        self.assertTrue(np.all(data[9][3] == 1.0))
        self.assertTrue(np.all(data[10][3] == 1.0))

    def test_gps_is_sampled_and_held_at_ten_hz(self):
        model = GazeboModel(gps_config(), 4, torch.device('cpu'), 1)
        reset_model(model)
        first_sample = model.get_gps_state()['position'].clone()
        model.s[:, 0] += 10.0

        for _ in range(4):
            model._update_gps()
        torch.testing.assert_close(
            model.get_gps_state()['position'], first_sample
        )
        self.assertAlmostEqual(
            float(model.get_gps_state()['age'][0]), 0.08, places=6
        )

        model._update_gps()
        torch.testing.assert_close(
            model.get_gps_state()['position'], model.s[:, 0:3]
        )
        self.assertEqual(float(model.get_gps_state()['age'][0]), 0.0)

    def _full_mission_env(self, device='cpu', num_envs=1):
        env = ControlEnv(
            num_envs=num_envs,
            config='gazebo_vtol_mission',
            model='GAZEBO',
            random_seed=2,
            device=device,
        )
        env.task.curriculum_enabled = False
        return env

    def _hover_mission_env(self, device='cpu', num_envs=1):
        env = ControlEnv(
            num_envs=num_envs,
            config='gazebo_vtol_hover_mission',
            model='GAZEBO',
            random_seed=2,
            device=device,
        )
        env.task.curriculum_enabled = False
        return env

    def test_hover_mission_inherits_dynamics_and_starts_on_ground(self):
        env = self._hover_mission_env(num_envs=8)
        obs = env.reset()

        self.assertEqual(env.task.terminal_mode, 'hover')
        self.assertEqual(env.task.PHASE_NAMES[-1], 'vertical_hover')
        self.assertEqual(float(env.config.mission_landing_n), 300.0)
        self.assertEqual(float(env.config.mission_landing_hover_altitude), 10.0)
        self.assertEqual(int(env.config.ground_physics_substeps), 5)
        self.assertTrue(bool(env.model.on_ground.all()))
        self.assertEqual(tuple(obs.shape), (8, 45))

    def test_hover_target_uses_airborne_altitude(self):
        env = self._hover_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.VERTICAL_HOVER
        task._refresh_guidance()

        self.assertAlmostEqual(
            float(task.target_altitude[0]), task.landing_hover_altitude, places=6
        )
        clean_obs = task.get_clean_obs(env)
        expected_altitude_error = (
            task.landing_hover_altitude - env.model.s[0, 2]
        ) / task.altitude_norm
        self.assertAlmostEqual(
            float(clean_obs[0, 2]), float(expected_altitude_error), places=6
        )

    def test_hover_success_requires_continuous_hold(self):
        env = self._hover_mission_env()
        env.reset()
        task = env.task
        mask = torch.ones(1, dtype=torch.bool)
        task.phase[:] = task.VERTICAL_HOVER
        task._refresh_guidance()
        env.model.s.zero_()
        env.model.s[:, 0] = task.landing_n
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.sync_reset_state(mask)

        for _ in range(task.hover_hold_steps - 1):
            env.step_count += 1
            task.update_before_observation(env)
        self.assertFalse(bool(task.mission_success(env)[0]))

        env.step_count += 1
        task.update_before_observation(env)
        done, bad_done, timeout, info = task.get_termination(env, {})
        self.assertTrue(bool(done[0]))
        self.assertFalse(bool(bad_done[0]))
        self.assertFalse(bool(timeout[0]))
        self.assertTrue(bool(info['mission_success'][0]))
        self.assertEqual(int(task.hover_stable_count[0]), task.hover_hold_steps)

    def test_hover_controller_keeps_bounded_ppo_residual_authority(self):
        env = self._hover_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.VERTICAL_HOVER
        env.model.s.zero_()
        env.model.s[:, 0] = task.landing_n
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        neutral = task.maybe_override_action(env, torch.zeros((1, 8)))
        policy_action = torch.zeros((1, 8))
        policy_action[:, 0:4] = 0.8
        residual = task.maybe_override_action(env, policy_action)

        self.assertGreater(float(residual[0, :4].mean()), float(neutral[0, :4].mean()))
        self.assertLessEqual(
            float((residual[0, :4] - neutral[0, :4]).abs().max()),
            task.hover_rl_collective_residual_scale + 1e-6,
        )

    def test_ground_contact_is_failure_for_hover_terminal_mode(self):
        env = self._hover_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.VERTICAL_HOVER
        boundary = next(
            condition for condition in task.termination_conditions
            if isinstance(condition, VTOLMissionBoundary)
        )

        bad_done, _, _, info = boundary.get_termination(task, env, {})
        self.assertTrue(bool(bad_done[0]))
        self.assertTrue(bool(info['mission_premature_contact'][0]))

    def test_observation_and_action_contract(self):
        env = self._full_mission_env(num_envs=8)
        obs = env.reset()
        self.assertEqual(tuple(obs.shape), (8, 45))
        self.assertEqual(env.num_actions, 8)
        self.assertTrue(bool(torch.isfinite(obs).all()))
        self.assertTrue(bool((env.task.phase == env.task.TAKEOFF).all()))
        self.assertTrue(env.model.gps_enabled)
        self.assertFalse(bool(getattr(env.config, 'enable_wind', True)))

    def test_default_training_reset_starts_every_agent_on_ground(self):
        env = ControlEnv(
            num_envs=32,
            config='gazebo_vtol_mission',
            model='GAZEBO',
            random_seed=3,
            device='cpu',
        )
        env.reset()
        self.assertFalse(env.task.curriculum_enabled)
        self.assertTrue(bool((env.task.phase == env.task.TAKEOFF).all()))
        self.assertTrue(bool(env.model.on_ground.all()))
        self.assertTrue(bool((env.step_count == 0).all()))

    def test_ground_and_low_speed_rotor_flight_ignore_aero_envelope(self):
        env = self._full_mission_env(num_envs=8)
        env.reset()
        action = torch.zeros((env.n, env.num_actions), device=env.device)
        _, _, _, bad_done, _, info = env.step(action)

        self.assertFalse(bool(bad_done.any()))
        self.assertFalse(bool(info['aero_envelope_active'].any()))
        self.assertFalse(bool(info['extreme_aero_state'].any()))

    def test_vertical_climb_does_not_activate_fixed_wing_aero_envelope(self):
        env = self._full_mission_env()
        env.reset()
        env.task.phase[:] = env.task.TRANSITION
        env.model.s[:, 2] = env.task.cruise_altitude
        env.model.s[:, 6] = 0.0
        env.model.s[:, 8] = -6.0
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        _, bad_done, _, info = env.task.get_termination(env, {})
        self.assertFalse(bool(info['aero_envelope_active'][0]))
        self.assertFalse(bool(info['extreme_aero_state'][0]))
        self.assertFalse(bool(bad_done[0]))

    def test_early_transition_ignores_wing_envelope_until_wingborne(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.TRANSITION
        env.model.s[:, 2] = task.cruise_altitude
        env.model.s[:, 6] = 0.5 * task.transition_speed
        env.model.s[:, 8] = -task.transition_speed
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        _, bad_done, _, info = task.get_termination(env, {})
        self.assertFalse(bool(info['aero_envelope_active'][0]))
        self.assertFalse(bool(info['extreme_aero_state'][0]))
        self.assertFalse(bool(bad_done[0]))

    def test_wingborne_transition_rejects_excessive_alpha(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.TRANSITION
        env.model.s[:, 2] = task.cruise_altitude
        env.model.s[:, 6] = task.transition_speed
        env.model.s[:, 8] = task.transition_speed
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        _, bad_done, _, info = task.get_termination(env, {})
        self.assertTrue(bool(info['aero_envelope_active'][0]))
        self.assertTrue(bool(info['extreme_aero_state'][0]))
        self.assertTrue(bool(bad_done[0]))

    def test_reverse_transition_does_not_use_fixed_wing_aero_termination(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.BACK_TRANSITION
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 6] = task.backtransition_speed
        env.model.s[:, 8] = task.backtransition_speed
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        _, bad_done, _, info = task.get_termination(env, {})
        self.assertFalse(bool(info['aero_envelope_active'][0]))
        self.assertFalse(bool(info['extreme_aero_state'][0]))
        self.assertFalse(bool(bad_done[0]))

    def test_backtransition_altitude_guidance_descends_on_capture_corridor(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.BACK_TRANSITION
        env.model.s[:, 0] = task.landing_n - task.approach_distance
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.cruise_altitude
        task._refresh_guidance()
        self.assertAlmostEqual(
            float(task.target_altitude[0]), task.cruise_altitude, places=5
        )

        env.model.s[:, 0] = task.landing_n - task.descent_capture_radius
        task._refresh_guidance()
        self.assertAlmostEqual(
            float(task.target_altitude[0]), task.landing_hover_altitude, places=5
        )

    def test_backtransition_surface_gate_limits_pitch_commands(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.BACK_TRANSITION
        action = torch.ones((env.n, env.num_actions))
        gated = task.maybe_override_action(env, action)
        self.assertTrue(bool(torch.all(gated[:, 5:8].abs() <=
                                       action[:, 5:8].abs())))
        self.assertLessEqual(float(gated[0, 5].abs()),
                             float(task.backtransition_surface_action_scale))

    def test_backtransition_blends_aerodynamic_loads(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.BACK_TRANSITION
        task.maybe_override_action(env, torch.zeros((env.n, env.num_actions)))
        self.assertAlmostEqual(
            float(env.model.aero_force_scale[0]),
            task.backtransition_aero_scale,
            places=6,
        )

    def test_rotor_climb_disables_fixed_wing_aerodynamic_loads(self):
        env = self._full_mission_env()
        env.reset()
        env.task.phase[:] = env.task.ROTOR_CLIMB
        env.task.maybe_override_action(env, torch.zeros((env.n, env.num_actions)))
        self.assertAlmostEqual(
            float(env.model.aero_force_scale[0]),
            env.task.rotor_climb_aero_scale,
            places=6,
        )

    def test_fixed_wing_gate_stops_lift_rotors(self):
        env = self._full_mission_env()
        env.reset()
        env.task.phase[:] = env.task.FIXED_WING
        action = torch.zeros((env.n, env.num_actions))
        env.model.s[:, 2] = env.task.cruise_altitude
        gated = env.task.maybe_override_action(env, action)
        torch.testing.assert_close(
            gated[:, 0:4],
            torch.full((env.n, 4), env.task.fixed_wing_lift_action),
        )
        env.model.s[:, 2] = env.task.min_cruise_altitude
        low = env.task.maybe_override_action(env, action)
        torch.testing.assert_close(
            low[:, 0:4],
            torch.full((env.n, 4), env.task.fixed_wing_lift_recovery_action),
        )

    def test_transition_pusher_cap_tapers_with_speed(self):
        env = self._full_mission_env(num_envs=2)
        env.reset()
        env.task.phase[:] = env.task.TRANSITION
        action = torch.zeros((env.n, env.num_actions))
        env.model.s[:, 6] = env.task.transition_speed
        low_speed = env.task.maybe_override_action(env, action)
        self.assertAlmostEqual(
            float(low_speed[0, 4]), env.task.transition_pusher_max_action, places=6
        )
        env.model.s[:, 6] = env.task.transition_pusher_speed_limit
        high_speed = env.task.maybe_override_action(env, action)
        self.assertAlmostEqual(
            float(high_speed[0, 4]),
            env.task.transition_pusher_high_speed_action,
            places=6,
        )

    def test_fixed_wing_pusher_cap_limits_cruise_thrust(self):
        env = self._full_mission_env()
        env.reset()
        env.task.phase[:] = env.task.FIXED_WING
        action = torch.zeros((env.n, env.num_actions))
        gated = env.task.maybe_override_action(env, action)
        self.assertAlmostEqual(
            float(gated[0, 4]), env.task.fixed_wing_pusher_max_action, places=6
        )

    def test_fixed_wing_aero_envelope_rejects_excessive_alpha(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.FIXED_WING
        task._refresh_guidance()
        env.model.s[:, 2] = task.cruise_altitude
        env.model.s[:, 6] = task.cruise_speed
        env.model.s[:, 8] = task.cruise_speed
        env.step_count[:] = task.fixed_wing_aero_grace_steps
        task.phase_entry_step[:] = 0
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        _, bad_done, _, info = task.get_termination(env, {})
        self.assertTrue(bool(info['aero_envelope_active'][0]))
        self.assertTrue(bool(info['extreme_aero_state'][0]))
        self.assertTrue(bool(bad_done[0]))

    def test_landing_capture_allows_small_along_track_overshoot(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        boundary = next(
            condition
            for condition in task.termination_conditions
            if isinstance(condition, VTOLMissionBoundary)
        )
        task.phase[:] = task.BACK_TRANSITION
        # 31 m beyond the route endpoint is within the 45 m landing capture
        # radius, so it must not be classified as an early route departure.
        env.model.s[:, 0] = task.landing_n + 31.0 * task.route_unit_n
        env.model.s[:, 1] = task.landing_e + 0.0 * task.route_unit_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        bad_done, done, timeout, info = boundary.get_termination(task, env, {})

        self.assertFalse(bool(bad_done[0]))
        self.assertFalse(bool(done[0]))
        self.assertFalse(bool(timeout[0]))
        self.assertTrue(bool(info['mission_capture_corridor'][0]))
        self.assertFalse(bool(info['mission_off_route'][0]))

    def test_landing_capture_does_not_allow_lateral_route_departure(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        boundary = next(
            condition
            for condition in task.termination_conditions
            if isinstance(condition, VTOLMissionBoundary)
        )
        task.phase[:] = task.BACK_TRANSITION
        # Use a tighter local lateral limit so the point can remain inside the
        # capture circle while still violating the independent cross-track
        # safety boundary.
        boundary.max_cross_track = 20.0
        cross_track = boundary.max_cross_track + 1.0
        along_offset = 10.0
        env.model.s[:, 0] = (
            task.landing_n + along_offset * task.route_unit_n
            - cross_track * task.route_unit_e
        )
        env.model.s[:, 1] = (
            task.landing_e + along_offset * task.route_unit_e
            + cross_track * task.route_unit_n
        )
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        bad_done, _, _, info = boundary.get_termination(task, env, {})

        self.assertTrue(bool(bad_done[0]))
        self.assertTrue(bool(info['mission_capture_corridor'][0]))
        self.assertTrue(bool(info['mission_off_route'][0]))

    def test_ground_action_filter_starts_from_motors_off(self):
        env = self._full_mission_env(num_envs=8)
        env.reset()
        self.assertTrue(bool(env.model.filtered_action_valid.all()))
        self.assertTrue(bool((env.model.filtered_action[:, 0:5] == -1.0).all()))

    def test_vertical_action_gate_preserves_collective_and_limits_differential(self):
        env = self._full_mission_env(num_envs=2)
        env.reset()
        action = torch.tensor([
            [1.0, 0.5, -0.5, -1.0, 1.0, 0.3, -0.2, 0.1],
            [-0.8, -0.2, 0.4, 0.8, 0.5, -0.1, 0.2, -0.3],
        ])
        gated = env.task.maybe_override_action(env, action)

        torch.testing.assert_close(
            gated[:, 0:4].mean(dim=1), action[:, 0:4].mean(dim=1)
        )
        expected_differential = torch.zeros_like(action[:, 0:4])
        torch.testing.assert_close(
            gated[:, 0:4] - gated[:, 0:4].mean(dim=1, keepdim=True),
            expected_differential,
        )
        torch.testing.assert_close(gated[:, 4], torch.full((2,), -1.0))
        torch.testing.assert_close(gated[:, 5:8], torch.zeros((2, 3)))

        env.model.s[:, 2] = env.task.rotor_climb_pusher_enable_altitude + 1.0
        env.task.phase[:] = env.task.ROTOR_CLIMB
        high = env.task.maybe_override_action(env, action)
        expected_high = (
            action[:, 0:4] - action[:, 0:4].mean(dim=1, keepdim=True)
        ) * env.task.rotor_climb_lift_differential_scale
        pusher = torch.full((2,), env.task.rotor_climb_pusher_max_action)
        trim = 0.5 * (pusher + 1.0) * env.task.rotor_climb_pusher_roll_trim_scale
        expected_high += torch.stack((trim, -trim, -trim, trim), dim=1)
        torch.testing.assert_close(
            high[:, 0:4] - high[:, 0:4].mean(dim=1, keepdim=True),
            expected_high,
        )

    def test_rotor_climb_enables_pusher_only_above_configured_altitude(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.ROTOR_CLIMB
        action = torch.zeros((env.n, env.num_actions))
        action[:, 4] = 0.7

        env.model.s[:, 2] = task.rotor_climb_pusher_enable_altitude - 0.1
        low = task.maybe_override_action(env, action)
        self.assertAlmostEqual(float(low[0, 4]), -1.0, places=6)

        env.model.s[:, 2] = task.rotor_climb_pusher_enable_altitude + 0.1
        high = task.maybe_override_action(env, action)
        self.assertAlmostEqual(
            float(high[0, 4]), task.rotor_climb_pusher_max_action, places=6
        )

        action[:, 4] = -1.0
        minimum = task.maybe_override_action(env, action)
        self.assertAlmostEqual(
            float(minimum[0, 4]), task.rotor_climb_pusher_min_action, places=6
        )

        action[:, 4] = 1.0
        maximum = task.maybe_override_action(env, action)
        self.assertAlmostEqual(
            float(maximum[0, 4]), task.rotor_climb_pusher_max_action, places=6
        )
        self.assertGreater(float(maximum[0, 0]), float(maximum[0, 1]))
        self.assertGreater(float(maximum[0, 3]), float(maximum[0, 2]))

    def test_base_env_does_not_apply_vertical_action_gate_twice(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        raw_action = torch.tensor([[1.0, 0.5, -0.5, -1.0, 1.0, 0.3, -0.2, 0.1]])
        expected = task.maybe_override_action(env, raw_action)

        # The runner applies the gate before handing the action to BaseEnv.
        # BaseEnv must consume the marker and execute that action unchanged.
        env._action_preprocessed = True
        with patch.object(env.model, 'update', return_value=None) as update:
            env.step(expected.clone())
        update.assert_called_once()
        applied = update.call_args.args[0]
        torch.testing.assert_close(applied, expected)

    def test_backtransition_pusher_is_tapered_toward_capture_radius(self):
        env = self._full_mission_env(num_envs=3)
        env.reset()
        task = env.task
        task.phase[:] = task.BACK_TRANSITION
        distance = torch.tensor([
            task.approach_distance,
            0.5 * (task.approach_distance + task.descent_capture_radius),
            task.descent_capture_radius,
        ])
        env.model.s[:, 0] = task.landing_n - distance * task.route_unit_n
        env.model.s[:, 1] = task.landing_e - distance * task.route_unit_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 3:12] = 0.0
        env.model.s[:, 6] = task.backtransition_speed

        action = torch.ones((env.n, env.num_actions))
        gated = task.maybe_override_action(env, action)

        max_command_fraction = (
            env.model.motor_omega_max[4] / env.model.motor_cmd_scaling[4]
        )
        torch.testing.assert_close(
            gated[:, 4],
            torch.tensor([
                2.0 * float(max_command_fraction) * task.backtransition_pusher_max_fraction - 1.0,
                float(max_command_fraction) * task.backtransition_pusher_max_fraction - 1.0,
                -1.0,
            ]),
            atol=1e-5,
            rtol=0.0,
        )
        distance_fraction = (
            (distance - task.descent_capture_radius)
            / (task.approach_distance - task.descent_capture_radius)
        ).clamp(0.0, 1.0)
        nominal_altitude = task.landing_hover_altitude + distance_fraction * (
            task.cruise_altitude - task.landing_hover_altitude
        )
        motor_constant = float(env.model.dynamics.motor_constant[0])
        hover_omega = torch.sqrt(
            (
                env.model.mass_curr * env.model.dynamics.g
                / (4.0 * max(motor_constant, 1.0e-8))
            ).clamp_min(0.0)
        )
        rotor_scale = env.model.motor_cmd_scaling[0]
        if env.model.action_is_gazebo_control:
            hover_action = hover_omega / rotor_scale
        else:
            hover_action = 2.0 * hover_omega / rotor_scale - 1.0
        target_sink = -torch.minimum(
            torch.full_like(nominal_altitude, task.backtransition_target_sink_speed),
            0.18 * torch.sqrt(
                (nominal_altitude - task.landing_hover_altitude).clamp_min(0.0)
            ),
        )
        expected_lift = (
            hover_action
            + task.backtransition_altitude_gain
            * (nominal_altitude - task.landing_hover_altitude)
            + task.backtransition_velocity_gain * target_sink
        ).clamp(-1.0, 1.0)
        torch.testing.assert_close(
            gated[:, 0:4], expected_lift.reshape(-1, 1).expand_as(action[:, 0:4])
        )
        torch.testing.assert_close(
            gated[:, 5:8],
            action[:, 5:8] * task.backtransition_surface_action_scale,
        )

    def test_backtransition_pusher_cap_survives_action_filter(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.BACK_TRANSITION
        env.model.s[:, 0] = task.landing_n - task.descent_capture_radius
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.filtered_action_valid[:] = True
        env.model.filtered_action[:, 4] = 1.0

        raw_action = torch.ones((env.n, env.num_actions))
        gated = task.maybe_override_action(env, raw_action)
        omega_ref, _ = env.model._map_action(gated)

        self.assertEqual(float(omega_ref[0, 4]), 0.0)
        self.assertEqual(float(env.model.filtered_action[0, 4]), -1.0)

        # The cap must also survive the motor lag stage.  Checking only the
        # mapped reference would miss a stale high motor state being carried
        # through the actuator filter.
        env.model._update_motor_filter(omega_ref)
        self.assertLessEqual(float(env.model.motor_omega[0, 4]), 1.0e-6)

    def test_backtransition_pusher_cap_limits_filtered_motor_state(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.BACK_TRANSITION
        env.model.s[:, 0] = task.landing_n - task.descent_capture_radius
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.motor_omega[:, 4] = env.model.motor_omega_max[4]
        env.model.filtered_action_valid[:] = True
        env.model.filtered_action[:, 4] = 1.0

        gated = task.maybe_override_action(
            env, torch.ones((env.n, env.num_actions))
        )
        omega_ref, _ = env.model._map_action(gated)
        env.model._update_motor_filter(omega_ref)

        self.assertEqual(float(omega_ref[0, 4]), 0.0)
        self.assertEqual(float(env.model.motor_omega[0, 4]), 0.0)

    def test_backtransition_extra_drag_is_phase_masked(self):
        env = self._full_mission_env(num_envs=3)
        env.reset()
        task = env.task
        task.phase[:] = torch.tensor([
            task.BACK_TRANSITION,
            task.FIXED_WING,
            task.VERTICAL_LANDING,
        ])
        distance = 0.5 * (
            task.backtransition_extra_drag_start_distance
            + task.descent_capture_radius
        )
        env.model.s[:, 0] = task.landing_n - distance
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 3:12] = 0.0
        env.model.sync_reset_state(torch.ones(3, dtype=torch.bool))

        task.maybe_override_action(
            env, torch.zeros((3, env.num_actions), device=env.device)
        )
        coefficient = env.model.dynamics.extra_drag_coefficient
        self.assertTrue(torch.is_tensor(coefficient))
        self.assertGreater(float(coefficient[0]), 0.0)
        self.assertEqual(float(coefficient[1]), 0.0)
        self.assertEqual(float(coefficient[2]), 0.0)

    def test_vertical_landing_attitude_pd_recovers_positive_roll(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.VERTICAL_LANDING
        env.model.s[:, 0] = task.landing_n
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 3:12] = 0.0
        env.model.s[:, 3] = 0.20
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        gated = task.maybe_override_action(
            env, torch.zeros((1, env.num_actions), device=env.device)
        )
        # Positive roll must produce the calibrated [+,-,-,+] differential
        # pattern, which commands a restoring negative roll moment.
        differential = gated[0, 0:4] - gated[0, 0:4].mean()
        self.assertGreater(float(differential[0]), 0.0)
        self.assertLess(float(differential[1]), 0.0)
        self.assertLess(float(differential[2]), 0.0)
        self.assertGreater(float(differential[3]), 0.0)

    def test_vertical_landing_navigation_rotates_world_error_by_yaw(self):
        env = self._full_mission_env(num_envs=2)
        env.reset()
        task = env.task
        task.phase[:] = task.VERTICAL_LANDING
        env.model.s[:, 0] = task.landing_n - 10.0
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 3:12] = 0.0
        env.model.s[1, 5] = math.pi / 2.0
        env.model.sync_reset_state(torch.ones(2, dtype=torch.bool))

        gated = task.maybe_override_action(
            env, torch.zeros((2, env.num_actions), device=env.device)
        )
        diff = gated[:, 0:4] - gated[:, 0:4].mean(dim=1, keepdim=True)
        # At yaw=0, north error maps to pitch differential.  At yaw=90 deg,
        # the same world error maps to body-right and therefore roll.
        self.assertLess(float(diff[0, 0]), 0.0)
        self.assertGreater(float(diff[0, 1]), 0.0)
        self.assertGreater(float(diff[1, 0]), 0.0)
        self.assertLess(float(diff[1, 1]), 0.0)
        self.assertNotEqual(
            tuple(torch.sign(diff[0]).tolist()),
            tuple(torch.sign(diff[1]).tolist()),
        )

    def test_vertical_capture_uses_horizontal_speed_and_requires_hold(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        mask = torch.ones(1, dtype=torch.bool)
        task.phase[:] = task.BACK_TRANSITION
        task.phase_entry_step[:] = 0
        env.step_count[:] = task.phase_min_steps[task.BACK_TRANSITION]
        env.model.s[:, 0] = task.landing_n - 0.5 * task.descent_capture_radius
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 3:12] = 0.0
        # A modest vertical component makes TAS exceed the gate, while
        # horizontal velocity remains a safe approach speed.
        env.model.s[:, 6] = 0.5 * task.backtransition_speed
        env.model.s[:, 8] = 0.5 * task.backtransition_max_sink_speed
        env.model.sync_reset_state(mask)

        for _ in range(task.backtransition_capture_hold_steps - 1):
            env.step_count += 1
            task.update_before_observation(env)
            self.assertEqual(int(task.phase[0]), task.BACK_TRANSITION)
        env.step_count += 1
        task.update_before_observation(env)
        self.assertEqual(int(task.phase[0]), task.VERTICAL_LANDING)

    def test_vertical_capture_does_not_count_duplicate_callback(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.BACK_TRANSITION
        env.step_count[:] = task.phase_min_steps[task.BACK_TRANSITION]
        env.model.s[:, 0] = task.landing_n - 0.5 * task.descent_capture_radius
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 3:12] = 0.0
        env.model.s[:, 6] = 0.5 * task.backtransition_speed
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        # Repeated observation callbacks at the same control timestamp count
        # as one sample, not as multiple hold cycles.
        for _ in range(task.backtransition_capture_hold_steps + 2):
            task.update_before_observation(env)
        self.assertEqual(int(task.phase[0]), task.BACK_TRANSITION)
        self.assertEqual(int(task.backtransition_capture_count[0]), 1)

    def test_vertical_capture_hold_counts_once_per_real_env_step(self):
        """The hold timer advances once through the actual BaseEnv step path."""
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.BACK_TRANSITION
        task.phase_entry_step[:] = 0
        env.step_count[:] = task.phase_min_steps[task.BACK_TRANSITION]
        env.model.s[:, 0] = task.landing_n - 0.5 * task.descent_capture_radius
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 3:12] = 0.0
        env.model.s[:, 6] = 0.5 * task.backtransition_speed
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        # Freeze the state so every callback sees the same valid capture
        # envelope.  BaseEnv.step still increments step_count and invokes the
        # task callback exactly once per control step.
        action = torch.zeros((1, env.num_actions))
        with patch.object(env.model, 'update', return_value=None):
            for expected_count in range(1, task.backtransition_capture_hold_steps):
                env.step(action)
                self.assertEqual(
                    int(task.backtransition_capture_count[0]), expected_count
                )
                self.assertEqual(int(task.phase[0]), task.BACK_TRANSITION)

            env.step(action)

        self.assertEqual(
            int(task.backtransition_capture_count[0]),
            task.backtransition_capture_hold_steps,
        )
        self.assertEqual(int(task.phase[0]), task.VERTICAL_LANDING)

    def test_vertical_capture_rejects_fast_flight_away_from_pad(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        mask = torch.ones(1, dtype=torch.bool)
        task.phase[:] = task.BACK_TRANSITION
        task.phase_entry_step[:] = 0
        env.step_count[:] = task.phase_min_steps[task.BACK_TRANSITION]
        env.model.s[:, 0] = task.landing_n - 0.5 * task.descent_capture_radius
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 3:12] = 0.0
        env.model.s[:, 6] = -0.5 * task.backtransition_speed
        env.model.sync_reset_state(mask)

        for _ in range(task.backtransition_capture_hold_steps + 1):
            env.step_count += 1
            task.update_before_observation(env)
        self.assertEqual(int(task.phase[0]), task.BACK_TRANSITION)
        self.assertEqual(int(task.backtransition_capture_count[0]), 0)

    def test_vertical_capture_rejects_unsafe_sink_rate(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        mask = torch.ones(1, dtype=torch.bool)
        task.phase[:] = task.BACK_TRANSITION
        task.phase_entry_step[:] = 0
        env.step_count[:] = task.phase_min_steps[task.BACK_TRANSITION]
        env.model.s[:, 0] = task.landing_n - 0.5 * task.descent_capture_radius
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 3:12] = 0.0
        env.model.s[:, 6] = 0.5 * task.backtransition_speed
        # Positive body-down W is a negative altitude rate at zero attitude.
        env.model.s[:, 8] = 2.0 * task.backtransition_max_sink_speed
        env.model.sync_reset_state(mask)

        for _ in range(task.backtransition_capture_hold_steps + 1):
            env.step_count += 1
            task.update_before_observation(env)
        self.assertEqual(int(task.phase[0]), task.BACK_TRANSITION)

    def test_vertical_capture_rejects_unsafe_climb_rate(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.BACK_TRANSITION
        task.phase_entry_step[:] = 0
        env.step_count[:] = task.phase_min_steps[task.BACK_TRANSITION]
        env.model.s[:, 0] = task.landing_n - 0.5 * task.descent_capture_radius
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 3:12] = 0.0
        env.model.s[:, 6] = 0.5 * task.backtransition_speed
        # Negative body-down W is a positive altitude rate at zero attitude.
        env.model.s[:, 8] = -2.0 * task.backtransition_max_sink_speed
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        for _ in range(task.backtransition_capture_hold_steps + 1):
            env.step_count += 1
            task.update_before_observation(env)
        self.assertEqual(int(task.phase[0]), task.BACK_TRANSITION)
        self.assertEqual(int(task.backtransition_capture_count[0]), 0)

    def test_vertical_capture_rejects_large_roll_rate(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.BACK_TRANSITION
        task.phase_entry_step[:] = 0
        env.step_count[:] = task.phase_min_steps[task.BACK_TRANSITION]
        env.model.s[:, 0] = task.landing_n - 0.5 * task.descent_capture_radius
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 2] = task.landing_hover_altitude
        env.model.s[:, 3:12] = 0.0
        env.model.s[:, 6] = 0.5 * task.backtransition_speed
        env.model.s[:, 9] = task.descent_capture_max_roll_rate + 0.1
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))

        for _ in range(task.backtransition_capture_hold_steps + 1):
            env.step_count += 1
            task.update_before_observation(env)
        self.assertEqual(int(task.phase[0]), task.BACK_TRANSITION)
        self.assertEqual(int(task.backtransition_capture_count[0]), 0)

    def test_ground_wait_is_worse_than_climbing_after_spool_grace(self):
        env = self._full_mission_env()
        env.reset()
        reward_fn = env.task.reward_functions[0]
        env.step_count[:] = reward_fn.ground_wait_grace_steps + 1
        ground_reward = reward_fn.get_reward(env.task, env).clone()

        env.model.s[:, 2] = 2.0
        env.model.s[:, 8] = -env.task.rotor_climb_speed
        env.model.recent_s[:] = env.model.s
        env.model.sync_reset_state(torch.ones(1, dtype=torch.bool))
        climbing_reward = reward_fn.get_reward(env.task, env)
        self.assertGreater(float(climbing_reward[0]), float(ground_reward[0]))

    @staticmethod
    def _set_airborne_reward_state(env, phase, distance, speed, heading=0.0):
        task = env.task
        task.phase[:] = phase
        task.phase_advanced[:] = False
        task._refresh_guidance()
        env.model.s.zero_()
        env.model.s[:, 0] = task.landing_n - distance * task.route_unit_n
        env.model.s[:, 1] = task.landing_e - distance * task.route_unit_e
        env.model.s[:, 2] = task.target_altitude
        env.model.s[:, 5] = heading
        env.model.s[:, 6] = speed
        env.model.recent_s[:] = env.model.s
        env.model.u.zero_()
        env.model.recent_u.zero_()
        env.model.on_ground[:] = False
        env.model.just_touchdown[:] = False
        env.model.max_ground_penetration.zero_()
        env.model.ground_normal_force.zero_()
        task.previous_waypoint_distance[:] = task._waypoint_distance(env)

    def test_dense_tracking_reward_is_zero_centered(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        reward_fn = task.reward_functions[0]
        self._set_airborne_reward_state(
            env,
            task.FIXED_WING,
            task.approach_distance,
            task.cruise_speed,
            task.route_heading,
        )

        reward = reward_fn.get_reward(task, env)

        # At perfect tracking with no progress, the dense terms must not pay a
        # positive survival bonus. Only the configured time cost remains.
        self.assertAlmostEqual(float(reward[0]), -reward_fn.w_time, places=5)

    def test_backtransition_braking_improvement_has_positive_shaping(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        reward_fn = task.reward_functions[0]
        distance = 0.5 * (
            task.approach_distance + task.descent_capture_radius
        )
        desired_speed = 0.5 * (
            task.cruise_speed + task.approach_speed
        )
        self._set_airborne_reward_state(
            env,
            task.BACK_TRANSITION,
            distance,
            desired_speed,
            task.route_heading,
        )
        reward_fn.acceleration_limit = 1.0e9
        holding_reward = reward_fn.get_reward(task, env).clone()

        env.model.recent_s[:, 6] = task.cruise_speed
        braking_reward = reward_fn.get_reward(task, env)

        self.assertGreater(float(braking_reward[0]), float(holding_reward[0]))

    def test_backtransition_overspeed_is_penalized(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        reward_fn = task.reward_functions[0]
        distance = task.descent_capture_radius
        self._set_airborne_reward_state(
            env,
            task.BACK_TRANSITION,
            distance,
            task.approach_speed,
            task.route_heading,
        )
        reward_fn.w_progress = 0.0
        reward_fn.w_altitude = 0.0
        reward_fn.w_speed = 0.0
        reward_fn.w_heading = 0.0
        reward_fn.w_waypoint = 0.0
        reward_fn.w_time = 0.0
        reward_fn.w_attitude = 0.0
        reward_fn.w_rate = 0.0
        reward_fn.w_aero = 0.0
        reward_fn.w_sink = 0.0
        reward_fn.w_smooth = 0.0
        reward_fn.w_mode = 0.0
        reward_fn.w_backtransition_braking = 0.0
        reward_fn.w_backtransition_receding = 0.0
        reward_fn.w_constraint = 0.0
        reward_fn.w_ground_wait = 0.0
        on_profile_reward = reward_fn.get_reward(task, env).clone()

        env.model.s[:, 6] = task.approach_speed + 4.0
        env.model.recent_s[:] = env.model.s
        overspeed_reward = reward_fn.get_reward(task, env)

        self.assertLess(float(overspeed_reward[0]), float(on_profile_reward[0]))

    def test_backtransition_flying_away_is_penalized(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        reward_fn = task.reward_functions[0]
        distance = 0.5 * (
            task.approach_distance + task.descent_capture_radius
        )
        speed = 0.5 * task.backtransition_speed
        self._set_airborne_reward_state(
            env, task.BACK_TRANSITION, distance, speed, task.route_heading
        )
        reward_fn.w_progress = 0.0
        reward_fn.w_altitude = 0.0
        reward_fn.w_speed = 0.0
        reward_fn.w_heading = 0.0
        reward_fn.w_waypoint = 0.0
        reward_fn.w_time = 0.0
        reward_fn.w_attitude = 0.0
        reward_fn.w_rate = 0.0
        reward_fn.w_aero = 0.0
        reward_fn.w_sink = 0.0
        reward_fn.w_smooth = 0.0
        reward_fn.w_mode = 0.0
        reward_fn.w_backtransition_braking = 0.0
        reward_fn.w_backtransition_overspeed = 0.0
        reward_fn.w_constraint = 0.0
        reward_fn.w_ground_wait = 0.0
        approaching_reward = reward_fn.get_reward(task, env).clone()

        env.model.s[:, 5] = task.route_heading + math.pi
        env.model.recent_s[:] = env.model.s
        receding_reward = reward_fn.get_reward(task, env)

        self.assertLess(float(receding_reward[0]), float(approaching_reward[0]))

    def test_terminal_event_categories_do_not_double_count(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        event_reward = task.reward_functions[1]
        env.bad_done[:] = True
        env.exceed_time_limit[:] = True

        reward = event_reward.get_reward(task, env)

        self.assertAlmostEqual(
            float(reward[0]), -event_reward.failure_penalty, places=5
        )

    def test_small_random_ground_actions_do_not_spike_contact_force(self):
        env = self._full_mission_env(num_envs=128)
        env.reset()
        generator = torch.Generator(device=env.device).manual_seed(11)

        for _ in range(10):
            action = torch.randn(
                (env.n, env.num_actions),
                generator=generator,
                device=env.device,
            ) * 0.2
            _, _, _, bad_done, _, info = env.step(action)
            self.assertFalse(bool(info['ground_crash_force'].any()))
            self.assertFalse(bool(bad_done.any()))

    def test_privileged_critic_observation_contract(self):
        env = self._full_mission_env(num_envs=8)
        actor_obs = env.reset()
        critic_obs = env.critic_obs()
        self.assertEqual(tuple(actor_obs.shape), (8, 45))
        self.assertEqual(tuple(critic_obs.shape), (8, 64))
        self.assertEqual(env.critic_observation_space.shape, (64,))
        self.assertTrue(env.task.use_privileged_critic)
        self.assertTrue(bool(torch.isfinite(critic_obs).all()))
        expected_landing_dn = (
            env.task.landing_n - env.model.s[:, 0]
        ) / env.task.distance_norm
        torch.testing.assert_close(critic_obs[:, 0], expected_landing_dn)

    def test_safety_constraints_are_penalized_in_reward(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        task.phase[:] = task.FIXED_WING
        task._refresh_guidance()
        env.model.s[:, 0] = 0.5 * (task.start_n + task.landing_n)
        env.model.s[:, 1] = 0.5 * (task.start_e + task.landing_e)
        env.model.s[:, 2] = task.cruise_altitude
        env.model.s[:, 3:12] = 0.0
        env.model.recent_s[:] = env.model.s
        task.reward_functions[0].get_reward(task, env)
        baseline = task.constraint_penalty.clone()

        env.model.s[:, 6] = 0.96 * float(env.config.max_velocity)
        env.model.recent_s[:] = env.model.s
        task.reward_functions[0].get_reward(task, env)
        self.assertGreater(float(task.constraint_penalty[0]), float(baseline[0]))

    def test_neutral_elevons_produce_symmetric_wing_loads(self):
        env = self._full_mission_env()
        env.reset()
        model = env.model
        model.s.zero_()
        model.s[:, 2] = env.task.cruise_altitude
        model.s[:, 6] = env.task.cruise_speed
        model.u.zero_()
        state_and_control = torch.hstack((model.s, model.u))
        wind_body = torch.zeros((1, 3), dtype=model.s.dtype)

        left_force, left_moment = model.dynamics._surface_force_moment(
            state_and_control, model.dynamics.surfaces[0], wind_body
        )
        right_force, right_moment = model.dynamics._surface_force_moment(
            state_and_control, model.dynamics.surfaces[1], wind_body
        )

        torch.testing.assert_close(left_force, right_force)
        self.assertAlmostEqual(
            float(left_moment[0, 0] + right_moment[0, 0]), 0.0, places=5
        )
        self.assertAlmostEqual(
            float(left_moment[0, 2] + right_moment[0, 2]), 0.0, places=5
        )

    def test_phase_machine_advances_in_order(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        mask = torch.ones(1, dtype=torch.bool)

        env.model.s[0, 2] = 2.0
        env.model.sync_reset_state(mask)
        env.step_count[:] = task.phase_min_steps[task.TAKEOFF]
        task.update_before_observation(env)
        self.assertEqual(int(task.phase[0]), task.ROTOR_CLIMB)

        env.model.s[0, 2] = task.cruise_altitude
        env.model.sync_reset_state(mask)
        env.step_count[:] = task.phase_entry_step + task.phase_min_steps[task.ROTOR_CLIMB]
        task.update_before_observation(env)
        self.assertEqual(int(task.phase[0]), task.TRANSITION)

        env.model.s[0, 2] = task.cruise_altitude
        env.model.s[0, 6] = task.transition_speed + 1.0
        env.model.sync_reset_state(mask)
        env.step_count[:] = task.phase_entry_step + task.phase_min_steps[task.TRANSITION]
        task.update_before_observation(env)
        self.assertEqual(int(task.phase[0]), task.FIXED_WING)

        env.model.s[0, 0] = task.landing_n - task.route_unit_n * task.approach_distance
        env.model.s[0, 1] = task.landing_e - task.route_unit_e * task.approach_distance
        env.model.sync_reset_state(mask)
        env.step_count[:] = task.phase_entry_step + task.phase_min_steps[task.FIXED_WING]
        task.update_before_observation(env)
        self.assertEqual(int(task.phase[0]), task.BACK_TRANSITION)

        env.model.s[0, 0] = task.landing_n - task.route_unit_n * 10.0
        env.model.s[0, 1] = task.landing_e - task.route_unit_e * 10.0
        env.model.s[0, 2] = task.landing_hover_altitude
        env.model.s[0, 6:12] = 0.0
        env.model.sync_reset_state(mask)
        env.step_count[:] = (
            task.phase_entry_step + task.phase_min_steps[task.BACK_TRANSITION]
        )
        for _ in range(task.backtransition_capture_hold_steps):
            env.step_count += 1
            task.update_before_observation(env)
        self.assertEqual(int(task.phase[0]), task.VERTICAL_LANDING)

    def test_settled_touchdown_at_landing_point_is_success(self):
        env = self._full_mission_env()
        env.reset()
        task = env.task
        mask = torch.ones(1, dtype=torch.bool)
        task.phase[:] = task.VERTICAL_LANDING
        task._refresh_guidance()
        env.model.s[:, 0] = task.landing_n
        env.model.s[:, 1] = task.landing_e
        env.model.s[:, 3:12] = 0.0
        env.model.set_initial_actuators(mask, [0.0, 0.0, 0.0, 0.0, 0.0])
        env.model.sync_reset_state(mask)
        env.model.contact_duration_steps[:] = int(
            getattr(env.config, 'mission_landing_settle_steps', 15)
        )

        done, bad_done, timeout, info = task.get_termination(env, {})
        self.assertTrue(bool(done[0]))
        self.assertFalse(bool(bad_done[0]))
        self.assertFalse(bool(timeout[0]))
        self.assertTrue(bool(info['mission_success'][0]))

    def test_vector_wrapper_returns_reset_observation_after_success(self):
        def make_env():
            return self._full_mission_env()

        vec_env = GPUVecEnv([make_env])
        vec_env.reset()
        base_env = vec_env.gpu_vec_env
        task = base_env.task
        mask = torch.ones(1, dtype=torch.bool)
        task.phase[:] = task.VERTICAL_LANDING
        task._refresh_guidance()
        base_env.model.s[:, 0] = task.landing_n
        base_env.model.s[:, 1] = task.landing_e
        base_env.model.s[:, 3:12] = 0.0
        base_env.model.set_initial_actuators(
            mask, [0.0, 0.0, 0.0, 0.0, 0.0]
        )
        base_env.model.sync_reset_state(mask)
        base_env.model.contact_duration_steps[:] = int(
            getattr(base_env.config, 'mission_landing_settle_steps', 15)
        )

        action = torch.tensor(
            [[[-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0]]]
        )
        obs, _, done, bad, timeout, info = vec_env.step(action)
        self.assertTrue(bool(done.reshape(-1)[0]))
        self.assertFalse(bool(bad.reshape(-1)[0]))
        self.assertFalse(bool(timeout.reshape(-1)[0]))
        self.assertTrue(bool(info['mission_success'][0]))
        self.assertEqual(int(base_env.step_count[0]), 0)
        self.assertEqual(int(base_env.task.phase[0]), task.TAKEOFF)
        self.assertEqual(int(obs[0, 0, 10]), 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_vectorized_task_stays_finite(self):
        env = self._full_mission_env(device='cuda:0', num_envs=1024)
        obs = env.reset()
        actions = torch.zeros((env.n, env.num_actions), device=env.device)
        for _ in range(5):
            obs, reward, _, _, _, _ = env.step(actions)
        torch.cuda.synchronize()
        self.assertEqual(obs.device.type, 'cuda')
        self.assertTrue(bool(torch.isfinite(obs).all()))
        self.assertTrue(bool(torch.isfinite(reward).all()))


if __name__ == '__main__':
    unittest.main()
