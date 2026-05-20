import os
import sys
import glob
import numpy as np
import gym

import torch
import torch.nn as nn
import onnx

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from algorithms.ppo.ppo_actor import PPOActor


# ======================================================
# F16 Tracking 任务的 PPO 网络参数 (与 train_tracking.sh 一致)
# ======================================================
# tracking.yaml: num_observation=22, num_actions=4
NUM_OBS = 22
NUM_ACTIONS = 4
HIDDEN_SIZE = '128 128'
ACT_HIDDEN_SIZE = '128 128'
RECURRENT_HIDDEN_SIZE = 128
RECURRENT_HIDDEN_LAYERS = 1


class Args:
    """与 train_tracking.sh 中的超参数保持一致"""
    def __init__(self, device="cpu"):
        self.gain = 0.01
        self.hidden_size = HIDDEN_SIZE
        self.act_hidden_size = ACT_HIDDEN_SIZE
        self.activation_id = 1                    # ReLU
        self.use_feature_normalization = True
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = RECURRENT_HIDDEN_SIZE
        self.recurrent_hidden_layers = RECURRENT_HIDDEN_LAYERS
        self.tpdv = dict(dtype=torch.float32, device=torch.device(device))
        self.use_prior = False


# ==========================================================
# 1. 构建 PPO Actor 模型 (F16 Tracking)
# ==========================================================
def build_model(device="cpu"):
    """
    构建 PPO Actor 网络，使用 F16 Tracking 任务参数。
    obs_space: Box(22,), act_space: Box(4,)
    """
    args = Args(device=device)
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_OBS,))
    act_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_ACTIONS,))
    model = PPOActor(args, obs_space, act_space, device=torch.device(device))
    return model


# ==========================================================
# 2. ONNX 导出用的包装器
#    PPOActor.forward 内部有分布采样，无法直接导出 ONNX。
#    此包装器只保留确定性推理路径:
#      输入: obs, rnn_states, masks
#      输出: actions (mean of Gaussian), rnn_states_out
# ==========================================================
class PPOActorONNXWrapper(nn.Module):
    """
    将 PPOActor 的确定性推理路径封装为纯张量运算，以便导出 ONNX。

    输入:
      obs:        [N, num_obs]
      rnn_states: [N, recurrent_hidden_layers, recurrent_hidden_size]
      masks:      [N, 1]
    输出:
      actions:        [N, num_actions]   (Gaussian 均值，经过 Tanh 映射到 [-1,1])
      rnn_states_out: [N, recurrent_hidden_layers, recurrent_hidden_size]
    """
    def __init__(self, actor: PPOActor):
        super().__init__()
        self.base = actor.base
        self.use_recurrent_policy = actor.use_recurrent_policy
        if self.use_recurrent_policy:
            self.rnn = actor.rnn
        # act 模块中可能有 mlp 层 + DiagGaussian
        self.act_mlp = actor.act.mlp if actor.act._mlp_actlayer else None
        # DiagGaussian 的 mu_net：Linear + Tanh，直接输出确定性动作
        self.mu_net = actor.act.action_out.mu_net

    def forward(self, obs, rnn_states, masks):
        x = self.base(obs)
        if self.use_recurrent_policy:
            x, rnn_states = self.rnn(x, rnn_states, masks)
        if self.act_mlp is not None:
            x = self.act_mlp(x)
        actions = self.mu_net(x)  # Tanh 已包含在 mu_net 中
        return actions, rnn_states


# ==========================================================
# 3. 加载 ckpt 权重
# ==========================================================
def load_ckpt_weights(model, ckpt_path, device="cpu"):
    """加载 PPOActor 的 ckpt 权重（由 torch.save(state_dict) 保存）"""
    state_dict = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"权重加载完成: {ckpt_path}")
    if missing:
        print(f"  [Warning] missing_keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"  [Warning] unexpected_keys ({len(unexpected)}): {unexpected}")
    return model


# ==========================================================
# 4. 导出 ONNX
# ==========================================================
def export_to_onnx(model, onnx_path, device="cpu", opset_version=17):
    """
    将 PPOActorONNXWrapper 导出为 ONNX。
    输入: obs [N, 22], rnn_states [N, 1, 128], masks [N, 1]
    输出: actions [N, 4], rnn_states_out [N, 1, 128]
    """
    model.to(device)
    model.eval()

    batch = 1
    dummy_obs = torch.randn(batch, NUM_OBS, device=device)
    dummy_rnn = torch.zeros(batch, RECURRENT_HIDDEN_LAYERS, RECURRENT_HIDDEN_SIZE, device=device)
    dummy_masks = torch.ones(batch, 1, device=device)

    input_names = ["obs", "rnn_states", "masks"]
    output_names = ["actions", "rnn_states_out"]

    dynamic_axes = {
        "obs": {0: "batch"},
        "rnn_states": {0: "batch"},
        "masks": {0: "batch"},
        "actions": {0: "batch"},
        "rnn_states_out": {0: "batch"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_obs, dummy_rnn, dummy_masks),
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    print(f"ONNX 已导出到: {onnx_path}")


# ==========================================================
# 5. 检查 ONNX
# ==========================================================
def check_onnx(onnx_path):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX 模型检查通过。")
    print(f"  inputs:  {[inp.name for inp in model.graph.input]}")
    print(f"  outputs: {[out.name for out in model.graph.output]}")


# ==========================================================
# 6. ONNXRuntime 推理测试
# ==========================================================
def test_onnxruntime(onnx_path, device="cpu"):
    if not HAS_ONNXRUNTIME:
        print("未安装 onnxruntime，跳过推理测试。pip install onnxruntime")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    batch = 1
    dummy_obs = np.random.randn(batch, NUM_OBS).astype(np.float32)
    dummy_rnn = np.zeros((batch, RECURRENT_HIDDEN_LAYERS, RECURRENT_HIDDEN_SIZE), dtype=np.float32)
    dummy_masks = np.ones((batch, 1), dtype=np.float32)

    outputs = session.run(None, {
        "obs": dummy_obs,
        "rnn_states": dummy_rnn,
        "masks": dummy_masks,
    })

    print("ONNXRuntime 推理成功:")
    print(f"  actions       shape: {outputs[0].shape}, values: {outputs[0]}")
    print(f"  rnn_states_out shape: {outputs[1].shape}")


# ==========================================================
# 7. PyTorch vs ONNX 一致性验证
# ==========================================================
def verify_consistency(actor_model, onnx_path, device="cpu"):
    """对比 PyTorch 原始模型与 ONNX 的输出是否一致"""
    if not HAS_ONNXRUNTIME:
        print("未安装 onnxruntime，跳过一致性验证。")
        return

    actor_model.to(device)
    actor_model.eval()

    batch = 2
    obs = torch.randn(batch, NUM_OBS, device=device)
    rnn_states = torch.zeros(batch, RECURRENT_HIDDEN_LAYERS, RECURRENT_HIDDEN_SIZE, device=device)
    masks = torch.ones(batch, 1, device=device)

    # PyTorch 推理
    with torch.no_grad():
        pt_actions, _, pt_rnn_out = actor_model(obs, rnn_states, masks, deterministic=True)
    pt_actions = pt_actions.cpu().numpy()

    # ONNX 推理
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {
        "obs": obs.cpu().numpy(),
        "rnn_states": rnn_states.cpu().numpy(),
        "masks": masks.cpu().numpy(),
    })
    onnx_actions = onnx_outputs[0]

    diff = np.abs(pt_actions - onnx_actions).max()
    print(f"PyTorch vs ONNX 最大误差: {diff:.8f}")
    if diff < 1e-5:
        print("✅ 一致性验证通过!")
    else:
        print("⚠️  存在较大误差，请检查模型结构。")
        print(f"  PyTorch actions:  {pt_actions}")
        print(f"  ONNX actions:     {onnx_actions}")


# ==========================================================
# 8. 自动查找最新的 tracking 训练目录
# ==========================================================
def find_latest_ckpt(runs_dir, scenario="tracking", model_name="F16"):
    """
    在 runs 目录中找到最新的 tracking 训练目录，并返回其中最大 episode 的 actor_latest.ckpt
    """
    pattern = os.path.join(runs_dir, f"*{scenario}*{model_name}*ppo*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        return None
    latest_run = dirs[-1]
    # 查找最大 episode 编号
    episode_dirs = glob.glob(os.path.join(latest_run, "episode_*"))
    if not episode_dirs:
        return None
    episode_dirs.sort(key=lambda x: int(x.split("episode_")[-1]))
    latest_episode = episode_dirs[-1]
    ckpt_path = os.path.join(latest_episode, "actor_latest.ckpt")
    if os.path.exists(ckpt_path):
        return ckpt_path
    return None


# ==========================================================
# 主函数
# ==========================================================
def main():
    # ====== 配置 ======
    device = "cpu"   # 建议用 cpu 导出 ONNX
    onnx_path = "ppo_actor_tracking_f16.onnx"

    # ====== 查找 ckpt ======
    runs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scripts", "runs")
    ckpt_path = "/home/a/demo/NeuralPlane_stable_V1/scripts/runs/2026-03-09_21-31-43_Control_tracking_F16_ppo_v1/episode_250/actor_latest.ckpt"

    # if ckpt_path is None:
    #     # 手动指定路径
    #     ckpt_path = input("未找到 tracking 训练目录，请手动输入 actor_latest.ckpt 路径: ").strip()
    # print(f"使用 ckpt: {ckpt_path}")

    # 1) 构建 PPO Actor 模型
    actor = build_model(device=device)
    print(f"模型结构: obs_dim={NUM_OBS}, act_dim={NUM_ACTIONS}, "
          f"hidden={HIDDEN_SIZE}, rnn={RECURRENT_HIDDEN_SIZE}x{RECURRENT_HIDDEN_LAYERS}")

    # 2) 加载权重
    actor = load_ckpt_weights(actor, ckpt_path, device=device)

    # 3) 包装为 ONNX 可导出模型
    wrapper = PPOActorONNXWrapper(actor)
    wrapper.to(device)
    wrapper.eval()

    # 4) 导出 ONNX
    export_to_onnx(wrapper, onnx_path, device=device, opset_version=17)

    # 5) 检查 ONNX
    check_onnx(onnx_path)

    # 6) ONNXRuntime 推理测试
    test_onnxruntime(onnx_path, device=device)

    # 7) 一致性验证 (PyTorch vs ONNX)
    verify_consistency(actor, onnx_path, device=device)


if __name__ == "__main__":
    main()