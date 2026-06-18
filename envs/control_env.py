import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from env_base import BaseEnv
from models.F16_model import F16Model
from models.UAV_model import UAVModel
from models.hybrid_model import HybridModel
from models.hybrid_model_new import HybridModelNew, HybridModelNewNoForward
from models.simulink_model import SimulinkModel
from tasks.heading_task import HeadingTask
from tasks.control_task import ControlTask
from tasks.tracking_task import TrackingTask
from tasks.rc_task import RCTask
from tasks.rc_human_task import RCHumanTask
from tasks.rpy_throttle_human_task import RPYThrottleHumanTask
from tasks.rpy_throttle_reach_task import RPYThrottleReachTask
from tasks.simulink_task import SimulinkTask
from tasks.hover_task import HoverTask
from tasks.circle_task import CircleTask

class ControlEnv(BaseEnv):
    """
    ControlEnv is a fly-control env for single agent to do tracking task.
    """
    def __init__(self, num_envs=1, config='heading', model='F16', random_seed=None, device="cuda:0"):
        super().__init__(num_envs, config, model, random_seed, device)
    
    def load(self, random_seed, config, model):
        if random_seed is not None:
            self.seed(random_seed)
        if model == 'F16':
            self.model = F16Model(self.config, self.n, self.device, random_seed)
        elif model == 'UAV':
            self.model = UAVModel(self.config, self.n, self.device, random_seed)
        elif model == 'HYBRID':
            self.model = HybridModel(self.config, self.n, self.device, random_seed)
        elif model in ('HYBRID_NEW', 'HYBRID_WIND'):
            self.model = HybridModelNew(self.config, self.n, self.device, random_seed)
        elif model == 'HYBRID_NEW_NO_FORWARD':
            self.model = HybridModelNewNoForward(self.config, self.n, self.device, random_seed)
        elif model == 'Simulink':
            self.model = SimulinkModel(self.config, self.n, self.device, random_seed)
        else:
            raise NotImplementedError
        # print(self.config)
        task_name = getattr(self.config, 'task_name', config)
        if task_name == 'heading':
            self.task = HeadingTask(self.config, self.n, self.device, random_seed)
        elif task_name == 'control':
            self.task = ControlTask(self.config, self.n, self.device, random_seed)
        elif task_name == 'tracking':
            self.task = TrackingTask(self.config, self.n, self.device, random_seed)
        elif task_name == 'rc':
            self.task = RCTask(self.config, self.n, self.device, random_seed)
        elif task_name == 'rc_human':
            self.task = RCHumanTask(self.config, self.n, self.device, random_seed)
        elif task_name == 'rpy_throttle_human':
            self.task = RPYThrottleHumanTask(self.config, self.n, self.device, random_seed)
        elif task_name == 'rpy_throttle_reach':
            self.task = RPYThrottleReachTask(self.config, self.n, self.device, random_seed)
        elif task_name == 'simulink':
            self.task = SimulinkTask(self.config, self.n, self.device, random_seed)
        elif task_name == 'hover':
            self.task = HoverTask(self.config, self.n, self.device, random_seed)
        elif task_name == 'circle':
            self.task = CircleTask(self.config, self.n, self.device, random_seed)
        else:
            raise NotImplementedError
    
