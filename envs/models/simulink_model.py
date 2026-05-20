import os
import sys
import torch
from torchdiffeq import odeint_adjoint as odeint
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from model_base import BaseModel
from Simulink.Simulink_dynamics import SimulinkDynamics

class SimulinkModel(BaseModel):
    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)
        self.num_states = 4
        self.num_controls = 1
        self.dt = 0.005
        self.solver = 'rk4'
        self.airspeed = getattr(self.config, 'airspeed', 0)

        self.s = torch.zeros((self.n, self.num_states), device=self.device)  # state
        self.recent_s = torch.zeros((self.n, self.num_states), device=self.device)  # recent state
        self.u = torch.zeros((self.n, self.num_controls), device=self.device) # control
        self.recent_u = torch.zeros((self.n, self.num_controls), device=self.device)  # recent control

        self.dynamics = SimulinkDynamics()
        self.first_control = torch.ones(self.n, dtype=torch.bool, device=self.device)

    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = (done | bad_done) | exceed_time_limit
        size = torch.sum(reset)
        self.s[reset, :] = torch.zeros((size, self.num_states), device=self.device)  # state
        self.u[reset, :] = torch.zeros((size, self.num_controls), device=self.device)
        self.s[reset, 0] = torch.ones_like(self.s[reset, 0]) * 100.
        # theta starts at 0 (was 2.0) — already zeros from line above
        self.recent_s[reset] = self.s[reset]
        self.recent_u[reset] = self.u[reset]
        self.first_control[reset] = True

    def get_extended_state(self):
        x = torch.hstack((self.s, self.u))
        return self.dynamics.nlplant(x)
    
    def update(self, action):
        action = torch.clamp(action, -1, 1)
        self.recent_u = self.u
        # First control input: direct assignment; otherwise: rate-limited
        scaled_action = action * 0.35
        delta_u = torch.clamp(scaled_action - self.u, -0.05, 0.05)
        rate_limited = self.u + delta_u
        self.u = torch.where(self.first_control.unsqueeze(-1), scaled_action, rate_limited)
        self.first_control[:] = False
        self.recent_s = self.s
        self.s = odeint(self.dynamics,
                        torch.hstack((self.s, self.u)),
                        torch.tensor([0., self.dt], device=self.device),
                        method=self.solver)[1, :, :self.num_states]

    # ====== Abstract method implementations ======
    # Simulink model states: [u, w, q, theta], control: [appc]
    # Most flight-dynamics concepts don't apply; return zeros or mapped values.

    def get_state(self):
        return self.s

    def get_control(self):
        return self.u

    def get_position(self):
        zeros = torch.zeros(self.n, device=self.device)
        return zeros, zeros, zeros  # npos, epos, altitude

    def get_posture(self):
        roll = torch.zeros(self.n, device=self.device)
        pitch = self.s[:, 3]   # theta
        yaw = torch.zeros(self.n, device=self.device)
        return roll, pitch, yaw

    def get_velocity(self):
        return self.s[:, 0], torch.zeros(self.n, device=self.device), self.s[:, 1]  # u, v, w

    def get_ground_speed(self):
        return self.s[:, 0], torch.zeros(self.n, device=self.device)  # vx, vy

    def get_climb_rate(self):
        return self.s[:, 1]  # w

    def get_angular_velocity(self):
        zeros = torch.zeros(self.n, device=self.device)
        return zeros, self.s[:, 2], zeros  # P, Q(=q), R

    def get_euler_angular_velocity(self):
        zeros = torch.zeros(self.n, device=self.device)
        return zeros, self.s[:, 2], zeros

    def get_vt(self):
        return self.s[:, 0]  # u as total velocity

    def get_TAS(self):
        return self.s[:, 0]

    def get_AOA(self):
        return torch.zeros(self.n, device=self.device)

    def get_AOS(self):
        return torch.zeros(self.n, device=self.device)

    def get_thrust(self):
        return torch.zeros(self.n, device=self.device)

    def get_control_surface(self):
        zeros = torch.zeros(self.n, device=self.device)
        return self.u[:, 0], zeros, zeros, zeros  # el, ail, rud, lef

    def get_acceleration(self):
        es = self.get_extended_state()
        return es[:, 0], torch.zeros(self.n, device=self.device), es[:, 1]  # ax, ay, az

    def get_accels(self):
        return self.get_acceleration()

    def get_G(self):
        return torch.zeros(self.n, device=self.device)

    def get_EAS2TAS(self):
        return torch.ones(self.n, device=self.device)

        
if __name__ == "__main__":
    sim = SimulinkDynamics()
    state = torch.zeros(1, 4)
    state[:, 0] = 100.
    state[:, 1] = 0.
    state[:, 2] = 0.
    state[:, 3] = 2.
    control = torch.zeros(1, 1)
    for i in range(2000):
        if(i >= 200):
            control = torch.ones(1, 1)
        state = odeint(sim, torch.hstack((state, control)),
                        torch.tensor([0., 0.005], device=torch.device('cuda:0')),
                        method='euler')[1, :, :4]
        # estate = uav.compute_extended_state(torch.hstack((state, control)))
        print("第{:}次的姿态为({:},{:}".format(i, state[:, 2], state[:, 3]))

    