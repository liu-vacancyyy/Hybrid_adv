import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class PositionReward(BaseRewardFunction):
    """
    Measure the difference between the current position and the target position
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env):
        """
        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        npos, epos, altitude = env.model.get_position()

        distance = torch.sqrt((npos - task.target_npos) ** 2 + (epos - task.target_epos) ** 2 + (altitude - task.target_altitude) ** 2)
        last_distance = torch.sqrt((task.last_delta_npos) ** 2 + (task.last_delta_epos) ** 2 + (task.last_delta_altitude) ** 2)

        delta_npos = (torch.abs(npos - task.target_npos) - torch.abs(task.last_delta_npos))
        delta_epos = (torch.abs(epos - task.target_epos) - torch.abs(task.last_delta_epos))
        delta_altitude = (torch.abs(altitude - task.target_altitude) - torch.abs(task.last_delta_altitude))
        reward_npos = - delta_npos / (task.max_distance/100)
        reward_epos = - delta_epos / (task.max_distance/100) 
        reward_altitude = - delta_altitude / (task.max_distance/100)
        # reward_target = reward_npos + reward_epos + reward_altitude

        delta_x = task.target_npos - npos
        delta_y = task.target_epos - epos
        target_yaw = torch.atan2(delta_y, delta_x)
        
        # 计算Pitch角（绕X轴的旋转）
        horizontal_distance = torch.sqrt(delta_x**2 + delta_y**2)
        delta_z = task.target_altitude - altitude
        target_pitch = torch.atan2(delta_z, horizontal_distance)

        roll, pitch, heading = env.model.get_posture()
        delta_pitch = wrap_PI(pitch - target_pitch)
        delta_heading = wrap_PI(heading - target_yaw)
        reward_pitch = -(torch.abs(delta_pitch)-torch.abs(task.last_delta_pitch)) * 180 / torch.pi * torch.abs(altitude - task.target_altitude) / (torch.abs(task.delta_altitude).clamp_min(100))
        reward_heading = -(torch.abs(delta_heading)-torch.abs(task.last_delta_heading)) * 180 / torch.pi * torch.abs(epos - task.target_epos) / (torch.abs(task.target_epos).clamp_min(100))

        # reward_target = (last_distance - distance) / (task.max_distance/100) - 0.01 + reward_heading + 0.2 * reward_altitude / (task.max_distance/100)
        reward_target = (last_distance - distance) / (task.max_distance/100) - 0.01 + reward_heading + reward_pitch

        print('reward npos=',reward_target[:10])
        print('reward heading=',reward_heading[:10])
        print('reward pitch=',reward_pitch[:10])
        # print('delta heading=',delta_heading[:10])
        # print('delta heading=',task.last_delta_heading[:10])
        # print('delta pitch=',delta_pitch[:10])
        # print('delta pitch=',task.last_delta_pitch[:10])

        # print('reward npos=',reward_npos[0],',reward_epos=',reward_epos[0],',reward_alt=',reward_altitude[0])

        task.last_delta_npos = npos - task.target_npos
        task.last_delta_epos = epos - task.target_epos
        task.last_delta_altitude = altitude - task.target_altitude
        task.last_delta_pitch = delta_pitch 
        task.last_delta_heading = delta_heading

        print('last npos=',task.last_delta_npos[0],',last_epos=',task.last_delta_epos[0],',last_alt=',task.last_delta_altitude[0])
        print('                           ')
        return reward_target
