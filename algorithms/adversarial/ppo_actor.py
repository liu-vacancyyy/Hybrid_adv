import numpy as np
import torch
import torch.nn as nn
import gym.spaces

from ..utils.distributions import FixedNormal
from ..utils.flatten import build_flattener
from ..utils.utils import check


def _parse_hidden_size(hidden_size):
    if isinstance(hidden_size, str):
        return [int(x) for x in hidden_size.split() if x]
    return [int(x) for x in hidden_size]


def _activation(activation_id):
    return [nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.ELU][activation_id]


class AdversarialPPOActor(nn.Module):
    """MLP adversary policy.

    The paper adversary uses a fully-connected MLP policy.  The default caller
    config uses three hidden layers and ReLU activations, matching the main
    adversary setting while keeping the PPO interface compatible with the
    existing rollout buffer.
    """

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__()
        if not isinstance(act_space, gym.spaces.Box):
            raise NotImplementedError("AdversarialPPOActor only supports Box action spaces.")

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.obs_flattener = build_flattener(obs_space)
        input_dim = int(self.obs_flattener.size)
        hidden_dims = _parse_hidden_size(args.hidden_size)
        action_dim = int(np.prod(act_space.shape))
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
        self.mean_out = nn.Linear(last_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.action_bound = float(getattr(args, "adv_action_bound", 1.0))

        self.to(device)

    def _features(self, obs):
        obs = check(obs).to(**self.tpdv).reshape(obs.shape[0], -1)
        if self.use_feature_normalization:
            obs = self.feature_norm(obs)
        x = obs
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return x

    def _distribution(self, obs):
        features = self._features(obs)
        mean = torch.tanh(self.mean_out(features)) * self.action_bound
        std = self.log_std.exp().expand_as(mean)
        return FixedNormal(mean, std)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        dist = self._distribution(obs)
        actions = dist.mode() if deterministic else dist.sample()
        actions = torch.clamp(actions, -self.action_bound, self.action_bound)
        action_log_probs = dist.log_probs(actions)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        action = check(action).to(**self.tpdv)
        action = torch.clamp(action, -self.action_bound, self.action_bound)
        dist = self._distribution(obs)
        action_log_probs = dist.log_probs(action)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
            dist_entropy = (dist.entropy() * active_masks) / active_masks.sum()
        else:
            dist_entropy = dist.entropy() / action_log_probs.size(0)
        return action_log_probs, dist_entropy

    def lipschitz_product(self):
        norms = []
        for layer in list(self.hidden_layers) + [self.mean_out]:
            weight = layer.weight
            norms.append(torch.max(torch.sum(torch.abs(weight), dim=1)))
        if not norms:
            return torch.zeros((), **self.tpdv)
        return torch.prod(torch.stack(norms))
