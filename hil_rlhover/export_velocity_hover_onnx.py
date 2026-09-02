#!/usr/bin/env python3
"""Export a recurrent PPO actor checkpoint for the Gazebo velocity-hover policy.

The hover policy trained with ``envs/configs/gazebo_velocity_hover.yaml`` uses:

* observation: 27 floats
* action: 5 continuous motor actions in [-1, 1]
* recurrent state: [batch, 1, 128]

The exported ONNX has inputs ``obs``, ``rnn_states``, ``masks`` and outputs
``actions``, ``rnn_states_out``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _state_dict_from_checkpoint(path: Path):
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "actor", "policy", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def _make_onnx_wrapper(actor):
    import torch.nn as nn

    class PPOActorONNXWrapper(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.base = actor.base
            self.use_recurrent_policy = actor.use_recurrent_policy
            self.rnn = actor.rnn if actor.use_recurrent_policy else None
            self.act_mlp = actor.act.mlp if actor.act._mlp_actlayer else None
            self.mu_net = actor.act.action_out.mu_net

        def forward(self, obs, rnn_states, masks):
            x = self.base(obs)
            if self.use_recurrent_policy:
                x, rnn_states = self.rnn(x, rnn_states, masks)
            if self.act_mlp is not None:
                x = self.act_mlp(x)
            actions = self.mu_net(x)
            return actions, rnn_states

    return PPOActorONNXWrapper(actor)


def build_actor(config: dict[str, Any], device: str):
    import torch

    try:
        import gym
    except ImportError:  # pragma: no cover - depends on local env
        import gymnasium as gym

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from algorithms.ppo.ppo_actor import PPOActor

    obs_dim = int(config.get("num_observation", 27))
    action_dim = int(config.get("num_actions", 5))
    args = SimpleNamespace(
        gain=0.01,
        hidden_size="128 128",
        act_hidden_size="128 128",
        activation_id=1,
        use_feature_normalization=True,
        use_recurrent_policy=True,
        recurrent_hidden_size=128,
        recurrent_hidden_layers=1,
        use_prior=False,
        use_safety_aux=False,
        tpdv={"dtype": torch.float32, "device": torch.device(device)},
    )
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return PPOActor(args, obs_space, act_space, device=torch.device(device))


def export_onnx(
    ckpt: Path,
    onnx_path: Path,
    config_path: Path,
    device: str,
    opset: int,
) -> None:
    import torch

    config = _load_yaml(config_path)
    obs_dim = int(config.get("num_observation", 27))
    action_dim = int(config.get("num_actions", 5))

    actor = build_actor(config, device)
    state_dict = _state_dict_from_checkpoint(ckpt)
    missing, unexpected = actor.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")

    wrapper = _make_onnx_wrapper(actor).to(device)
    wrapper.eval()

    batch = 1
    rnn_layers = int(actor.recurrent_hidden_layers)
    rnn_hidden = int(actor.recurrent_hidden_size)
    dummy_obs = torch.zeros(batch, obs_dim, dtype=torch.float32, device=device)
    dummy_rnn = torch.zeros(batch, rnn_layers, rnn_hidden, dtype=torch.float32, device=device)
    dummy_masks = torch.ones(batch, 1, dtype=torch.float32, device=device)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_obs, dummy_rnn, dummy_masks),
            str(onnx_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["obs", "rnn_states", "masks"],
            output_names=["actions", "rnn_states_out"],
            dynamic_axes={
                "obs": {0: "batch"},
                "rnn_states": {0: "batch"},
                "masks": {0: "batch"},
                "actions": {0: "batch"},
                "rnn_states_out": {0: "batch"},
            },
        )

    print(f"exported: {onnx_path}")
    print(f"interface: obs={obs_dim}, actions={action_dim}, rnn=[batch,{rnn_layers},{rnn_hidden}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path, help="episode_720 actor checkpoint")
    parser.add_argument("--onnx", required=True, type=Path, help="output ONNX path")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "envs/configs/gazebo_velocity_hover.yaml",
        help="training config used for obs/action dimensions",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--opset", default=17, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_onnx(args.ckpt, args.onnx, args.config, args.device, args.opset)


if __name__ == "__main__":
    main()
