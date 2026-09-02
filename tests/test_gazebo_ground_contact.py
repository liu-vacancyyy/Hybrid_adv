import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'envs'))
sys.path.insert(0, str(REPO_ROOT / 'envs' / 'models'))

from models.Gazebo.ground_contact import StandardVTOLGroundContact
from models.Gazebo.Gazebo_dynamics import GazeboVTOLDynamics
from models.gazebo_model import GazeboModel
from hybrid_termination_conditions.ground_collision import GroundCollision


def make_config(**overrides):
    values = dict(
        num_states=12,
        num_controls=8,
        dt=0.02,
        solver='euler',
        ground_contact_enable=True,
        ground_physics_substeps=5,
        spawn_mode='ground',
        ground_contact_stiffness=10000.0,
        ground_contact_damping=150.0,
        ground_contact_friction=0.7,
        ground_contact_slip_speed=0.05,
        ground_spawn_static_equilibrium=True,
        ground_spawn_clearance=0.0,
        min_altitude=0.5,
        max_altitude=0.5,
        init_roll_range=0.0,
        init_pitch_range=0.0,
        init_yaw_range=0.0,
        init_vel_range=0.0,
        init_omega_range=0.0,
        dr_mass=0.0,
        dr_inertia=0.0,
        ground_crash_max_touchdown_speed=2.5,
        ground_crash_max_penetration=0.05,
        ground_crash_max_roll_deg=45.0,
        ground_crash_max_pitch_deg=45.0,
        ground_crash_force_factor=20.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def reset_model(model):
    flags = torch.ones(model.n, dtype=torch.bool, device=model.device)
    env = SimpleNamespace(is_done=flags, bad_done=flags, exceed_time_limit=flags)
    model.reset(env)


class StandardVTOLGroundContactTest(unittest.TestCase):
    def test_contact_disabled_preserves_airborne_dynamics(self):
        enabled = GazeboVTOLDynamics(make_config())
        disabled = GazeboVTOLDynamics(make_config(ground_contact_enable=False))
        state_and_control = torch.zeros((4, 20))
        state_and_control[:, 2] = 10.0
        state_and_control[:, 3:6] = torch.tensor([0.1, -0.05, 0.2])
        state_and_control[:, 6:12] = torch.tensor([3.0, -0.2, 0.1, 0.02, -0.03, 0.04])
        state_and_control[:, 12:17] = 700.0

        torch.testing.assert_close(
            enabled.nlplant(state_and_control),
            disabled.nlplant(state_and_control),
            rtol=0.0,
            atol=0.0,
        )

    def test_level_static_contact_balances_weight(self):
        model = GazeboModel(make_config(), 8, torch.device('cpu'), 1)
        reset_model(model)

        expected_force = model.mass_curr * model.dynamics.g
        contact = model.get_ground_contact_state()
        self.assertTrue(torch.all(contact['on_ground']))
        self.assertTrue(torch.all(contact['contact_count'] == 4))
        torch.testing.assert_close(
            contact['normal_force'], expected_force, rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(
            contact['moment_body'], torch.zeros_like(contact['moment_body']),
            rtol=0.0, atol=1e-5,
        )

        stopped = torch.full((model.n, 5), -1.0)
        for _ in range(100):
            model.update(stopped)
        self.assertTrue(torch.all(model.on_ground))
        self.assertLess(float(model.s[:, 8].abs().max()), 1e-4)
        self.assertLess(float((model.s[:, 2] - model.s[0, 2]).abs().max()), 1e-5)

    def test_friction_opposes_horizontal_slip(self):
        contact_model = StandardVTOLGroundContact(make_config())
        state = torch.zeros((1, 12))
        static_penetration = 5.0 * 9.807 / (
            contact_model.number_of_points * contact_model.stiffness
        )
        state[:, 2] = contact_model.resting_cg_altitude(
            state, clearance=-static_penetration
        )
        state[:, 6] = 1.0

        force, _, diagnostics = contact_model.compute(state)
        self.assertTrue(bool(diagnostics['on_ground'][0]))
        self.assertLess(float(force[0, 0]), 0.0)
        self.assertAlmostEqual(float(force[0, 1]), 0.0, places=5)

    def test_drop_settles_without_tunneling(self):
        model = GazeboModel(
            make_config(spawn_mode='air'), 1, torch.device('cpu'), 1
        )
        reset_model(model)
        model.motor_omega.zero_()
        model.u.zero_()
        model.on_ground.zero_()

        stopped = torch.full((1, 5), -1.0)
        peak_penetration = 0.0
        peak_touchdown_speed = 0.0
        for _ in range(150):
            model.update(stopped)
            peak_penetration = max(
                peak_penetration, float(model.max_ground_penetration[0])
            )
            peak_touchdown_speed = max(
                peak_touchdown_speed, float(model.touchdown_vertical_speed[0])
            )

        self.assertTrue(torch.isfinite(model.s).all())
        self.assertTrue(bool(model.on_ground[0]))
        self.assertGreater(peak_touchdown_speed, 2.0)
        self.assertLess(peak_penetration, 0.03)
        self.assertLess(abs(float(model.get_climb_rate()[0])), 1e-3)

    def test_lift_rotors_release_contact(self):
        model = GazeboModel(make_config(), 1, torch.device('cpu'), 1)
        reset_model(model)
        takeoff = torch.tensor([[0.3, 0.3, 0.3, 0.3, -1.0]])
        for _ in range(30):
            model.update(takeoff)

        self.assertFalse(bool(model.on_ground[0]))
        self.assertGreater(float(model.s[0, 2]), 0.5)
        self.assertGreater(float(model.get_climb_rate()[0]), 0.5)

    def test_ground_collision_allows_rest_and_rejects_hard_touchdown(self):
        config = make_config()
        model = GazeboModel(config, 2, torch.device('cpu'), 1)
        reset_model(model)
        env = SimpleNamespace(
            model=model,
            n=model.n,
            device=model.device,
        )
        condition = GroundCollision(config)

        bad_done, done, timeout, info = condition.get_termination(None, env)
        self.assertFalse(bool(bad_done.any()))
        self.assertFalse(bool(done.any()))
        self.assertFalse(bool(timeout.any()))
        self.assertIn('ground_contact', info)

        model.just_touchdown[1] = True
        model.touchdown_vertical_speed[1] = 3.0
        bad_done, _, _, info = condition.get_termination(None, env, {})
        self.assertFalse(bool(bad_done[0]))
        self.assertTrue(bool(bad_done[1]))
        self.assertTrue(bool(info['ground_crash_hard_touchdown'][1]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_batch_contact_is_finite(self):
        num_envs = 4096
        model = GazeboModel(make_config(), num_envs, torch.device('cuda:0'), 1)
        reset_model(model)
        stopped = torch.full((num_envs, 5), -1.0, device=model.device)
        for _ in range(10):
            model.update(stopped)
        torch.cuda.synchronize()

        self.assertTrue(bool(torch.isfinite(model.s).all()))
        self.assertTrue(bool(model.on_ground.all()))
        expected_force = model.mass_curr * model.dynamics.g
        torch.testing.assert_close(
            model.ground_normal_force, expected_force, rtol=5e-4, atol=5e-4
        )


if __name__ == '__main__':
    unittest.main()
