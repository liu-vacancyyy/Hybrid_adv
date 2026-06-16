import torch
import torch.nn as nn

from ..utils.flatten import build_flattener
from ..utils.utils import check
from .ppo_actor import _activation, _parse_hidden_size


class AdversarialPPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.obs_flattener = build_flattener(obs_space)
        input_dim = int(self.obs_flattener.size)
        hidden_dims = _parse_hidden_size(args.hidden_size)
        activation_cls = _activation(args.activation_id)

        self.use_feature_normalization = bool(getattr(args, "use_feature_normalization", False))
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)

        self.hidden_layers = nn.ModuleList()
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim
        self.activation = activation_cls()
        self.value_out = nn.Linear(last_dim, 1)

        self.to(device)

    def forward(self, obs, rnn_states, masks):
        x = check(obs).to(**self.tpdv).reshape(obs.shape[0], -1)
        if self.use_feature_normalization:
            x = self.feature_norm(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        values = self.value_out(x)
        return values, rnn_states
