import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from termination_condition_base import BaseTerminationCondition
import torch
from utils.utils import wrap_PI

class ExtremeYaw(BaseTerminationCondition):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is on an extreme state.

        Args:
            env: environment instance

        Returns:
            (tuple): (bad_done, done, exceed_time_limit, info)
        """
        npos, epos, altitude = env.model.get_position()

        delta_x = task.target_npos - npos
        delta_y = task.target_epos - epos
        target_yaw = torch.atan2(delta_y, delta_x)
        
        # 计算Pitch角（绕X轴的旋转）
        horizontal_distance = torch.sqrt(delta_x**2 + delta_y**2)
        delta_z = task.target_altitude - altitude
        target_pitch = torch.atan2(delta_z, horizontal_distance)

        roll, pitch, heading = env.model.get_posture()
        bad_done = (torch.abs(wrap_PI(heading - target_yaw)) > torch.pi / 2) 
        done = torch.zeros_like(bad_done)
        exceed_time_limit = torch.zeros_like(bad_done)
        if torch.any(bad_done):
            self.log(f'extreme yaw!')
            print(torch.sum(bad_done), 'extreme yaw!')
        return bad_done, done, exceed_time_limit, info