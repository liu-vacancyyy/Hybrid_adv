import torch
import torch.nn as nn

class SimulinkDynamics(nn.Module):
    def __init__(self):
        super().__init__()
    
    def compute_extended_state(self, x):
        return self.nlplant(x)

    def forward(self, t, x):
        es = self.compute_extended_state(x)
        return es
    
    def nlplant(self, x):
        """
        model state(dim 4):
            0. u      
            1. v      
            2. q      
            3. tet    

        model control(dim 1)
            0. ego_appc
        """
        A = [-0.015, 0.002, 0.001, -0.003, -0.02, 0.005, -0.05, 1.0]
        B = [10.0, 2.0]
        gain_q = -0.5
        gain2 = 1.0
        u = x[:, 0]
        w = x[:, 1]
        q = x[:, 2]
        tet = x[:, 3]
        appc = x[:, 4]

        xdot = torch.zeros_like(x)

        internal_gain2 = (gain_q * q + appc)*gain2
        xdot[:, 0] = A[0]*u + A[1]*w + A[2]*q
        xdot[:, 1] = A[3]*u + A[4]*w + A[5]*q + B[0]*internal_gain2
        xdot[:, 2] = A[6]*q + B[1]*internal_gain2
        xdot[:, 3] = A[7]*q

        return xdot