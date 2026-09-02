import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'envs'))
sys.path.insert(0, str(REPO_ROOT / 'envs' / 'models'))

from envs.control_env import ControlEnv
from envs.env_wrappers import GPUVecEnv
from models.gazebo_model import GazeboModel


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

    def test_observation_and_action_contract(self):
        env = self._full_mission_env(num_envs=8)
        obs = env.reset()
        self.assertEqual(tuple(obs.shape), (8, 45))
        self.assertEqual(env.num_actions, 8)
        self.assertTrue(bool(torch.isfinite(obs).all()))
        self.assertTrue(bool((env.task.phase == env.task.TAKEOFF).all()))
        self.assertTrue(env.model.gps_enabled)
        self.assertFalse(bool(getattr(env.config, 'enable_wind', True)))

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
