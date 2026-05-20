import math
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from hybrid_model import HybridModel
from Hybrid.Hybrid_dynamics_new import HybridDynamicsNew


class HybridModelNew(HybridModel):
    """Wind-aware HybridModel.

    Differences from ``HybridModel``:
        - removes the old ``get_TAS() = get_vt() + airspeed`` shortcut;
        - computes TAS, alpha, beta and aero sin/cos from relative air velocity;
        - supports optional steady wind / gusts in NED axes.

    Config keys:
        enable_wind: wind is ignored unless this is true
        wind_north, wind_east, wind_down: steady wind in m/s
        gust_north, gust_east, gust_down: extra gust vector in m/s
        gust_speed, gust_direction_deg: horizontal gust convenience form,
            where 0 deg points north and 90 deg points east.
    """

    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)

        # Replace old dynamics with wind-aware dynamics.  Nominal parameters are
        # intentionally identical, so zero-wind output should match HybridModel.
        self.dynamics = HybridDynamicsNew()
        self.mass_curr = torch.ones(self.n, device=self.device) * self.dynamics.nominal_m
        self._Jx_t = torch.ones(self.n, device=self.device) * self.dynamics.nominal_Jx
        self._Jy_t = torch.ones(self.n, device=self.device) * self.dynamics.nominal_Jy
        self._Jz_t = torch.ones(self.n, device=self.device) * self.dynamics.nominal_Jz
        self._Jxz_t = torch.ones(self.n, device=self.device) * self.dynamics.nominal_Jxz
        self.dynamics.set_physics(self.mass_curr, self._Jx_t, self._Jy_t,
                                  self._Jz_t, self._Jxz_t)

        self.wind_enabled = bool(getattr(self.config, 'enable_wind', False))
        self.base_wind_ned = torch.zeros((self.n, 3), device=self.device)
        self.wind_ned = torch.zeros((self.n, 3), device=self.device)
        if self.wind_enabled:
            self.base_wind_ned = torch.stack(self._wind_from_config(), dim=1)
            self.set_wind_ned(
                self.base_wind_ned[:, 0],
                self.base_wind_ned[:, 1],
                self.base_wind_ned[:, 2],
            )
        else:
            self.dynamics.set_wind_ned(None)

    def _wind_from_config(self):
        north = float(getattr(self.config, 'wind_north', 0.0))
        east = float(getattr(self.config, 'wind_east', 0.0))
        down = float(getattr(self.config, 'wind_down', 0.0))

        north += float(getattr(self.config, 'gust_north', 0.0))
        east += float(getattr(self.config, 'gust_east', 0.0))
        down += float(getattr(self.config, 'gust_down', 0.0))

        gust_speed = float(getattr(self.config, 'gust_speed', 0.0))
        if gust_speed != 0.0:
            direction = math.radians(float(getattr(self.config, 'gust_direction_deg', 0.0)))
            north += gust_speed * math.cos(direction)
            east += gust_speed * math.sin(direction)
        return (
            self._expand_component(north),
            self._expand_component(east),
            self._expand_component(down),
        )

    def _expand_component(self, value):
        if torch.is_tensor(value):
            value = value.to(device=self.device, dtype=torch.float32).reshape(-1)
            if value.numel() == 1:
                value = value.expand(self.n)
            return value
        return torch.full((self.n,), float(value), device=self.device)

    def set_wind_ned(self, north=0.0, east=0.0, down=0.0):
        """Set total wind/gust vector in NED/world axes, m/s."""
        if not self.wind_enabled:
            self.wind_ned.zero_()
            self.dynamics.set_wind_ned(None)
            return
        self.wind_ned = torch.stack((
            self._expand_component(north),
            self._expand_component(east),
            self._expand_component(down),
        ), dim=1)
        self.dynamics.set_wind_ned(self.wind_ned)

    def set_gust_ned(self, north=0.0, east=0.0, down=0.0):
        """Alias for setting a one-direction gust vector."""
        self.set_wind_ned(north, east, down)

    def set_wind_gust_ned(self, north=0.0, east=0.0, down=0.0):
        """Apply an environment gust on top of the configured base wind."""
        if not self.wind_enabled:
            self.wind_ned.zero_()
            self.dynamics.set_wind_ned(None)
            return
        gust_ned = torch.stack((
            self._expand_component(north),
            self._expand_component(east),
            self._expand_component(down),
        ), dim=1)
        total = self.base_wind_ned + gust_ned
        self.set_wind_ned(total[:, 0], total[:, 1], total[:, 2])

    def get_wind_ned(self):
        return self.wind_ned[:, 0], self.wind_ned[:, 1], self.wind_ned[:, 2]

    def _wind_body(self):
        roll, pitch, yaw = self.get_posture()
        return HybridDynamicsNew._wind_body_from_ned(self.wind_ned, roll, pitch, yaw)

    def get_wind_body(self):
        return self._wind_body()

    def get_air_relative_velocity_body(self):
        U, V, W = self.s[:, 6], self.s[:, 7], self.s[:, 8]
        wx_b, wy_b, wz_b = self._wind_body()
        return U - wx_b, V - wy_b, W - wz_b

    def get_vt(self):
        U, V, W = self.get_air_relative_velocity_body()
        return torch.sqrt((U * U + V * V + W * W).clamp_min(0.0))

    def get_TAS(self):
        return self.get_vt()

    def get_EAS(self):
        return self.get_TAS() / self.get_EAS2TAS()

    def get_AOA(self):
        U, _V, W = self.get_air_relative_velocity_body()
        vxz2 = U * U + W * W
        vt2 = vxz2 + _V * _V
        alpha = torch.atan2(W, U)
        return torch.where(vt2 > 1e-4, alpha, torch.zeros_like(alpha))

    def get_AOS(self):
        U, V, W = self.get_air_relative_velocity_body()
        vxz = torch.sqrt((U * U + W * W).clamp_min(0.0))
        vt2 = vxz * vxz + V * V
        beta = torch.atan2(V, vxz)
        return torch.where(vt2 > 1e-4, beta, torch.zeros_like(beta))

    def get_aero_sincos(self):
        U, V, W = self.get_air_relative_velocity_body()
        vxz2 = U * U + W * W
        vt2 = vxz2 + V * V
        vxz = torch.sqrt(vxz2.clamp(min=0))
        inv_vxz = torch.rsqrt(vxz2.clamp(min=1e-6))
        inv_vt = torch.rsqrt(vt2.clamp(min=1e-6))
        aero_on = vt2 > 0.25
        sa = torch.where(aero_on, W * inv_vxz, torch.zeros_like(W))
        ca = torch.where(aero_on, U * inv_vxz, torch.zeros_like(U))
        sb = torch.where(aero_on, V * inv_vt, torch.zeros_like(V))
        cb = torch.where(aero_on, vxz * inv_vt, torch.zeros_like(vxz))
        return sa, ca, sb, cb
