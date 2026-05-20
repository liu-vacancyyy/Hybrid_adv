import torch
import torch.nn as nn
import torch.optim as optim
from ..utils.utils import check
from ..utils.flatten import build_flattener

class Adversary(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(Adversary, self).__init__()
        self.device = device  # 配置硬件
        self.gamma = args.adv_gamma
        self.value_loss_coef = args.adv_value_loss_coef
        self.max_grad_norm = args.adv_max_grad_norm
        self.num_learning_epochs = args.adv_num_learning_epochs
        self.num_mini_batches = args.adv_num_mini_batch
        self.tpdv = dict(dtype=torch.float32, device=device)

        activation = nn.ELU()
        advq_hidden_dims = [128, 128, 64]
        obs_dim = build_flattener(obs_space).size
        act_dim = build_flattener(act_space).size
        mlp_input_dim_q = obs_dim + act_dim
        # Qvalue function
        q_layers = []
        q_layers.append(nn.Linear(mlp_input_dim_q, advq_hidden_dims[0]))
        q_layers.append(activation)
        for l in range(len(advq_hidden_dims)):
            if l == len(advq_hidden_dims) - 1:
                q_layers.append(nn.Linear(advq_hidden_dims[l], 1))
            else:
                q_layers.append(nn.Linear(advq_hidden_dims[l], advq_hidden_dims[l + 1]))
                q_layers.append(activation)
        self.adv_q = nn.Sequential(*q_layers)
        # print('adv para',self.parameters())
        self.optimizer = optim.Adam(self.parameters(), lr=args.adv_lr)  # 优化器配置

        self.to(device)
        
    def evaluate_forward(self, obs, action):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        obs_action = torch.cat([obs, action], dim=-1)
        q = self.adv_q(obs_action).detach()
        return q

    def evaluate_backrward(self, obs, action):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        obs_action = torch.cat([obs, action], dim=-1)
        q = self.adv_q(obs_action)
        return q
    
