"""Evaluate a PID baseline on the RC OU command task.

This uses the real ``RCTask`` inside ``ControlEnv``, so the targets are the
same OU process used during RC RL training/evaluation.

本脚本用于评估 RC 任务上的 PID baseline。环境仍然使用真实的
``RCTask``，因此目标 ``target_vx / target_vz / target_heading`` 与
RL 训练、评估时看到的 OU 随机控制过程一致。

Usage:
    python renders/render_pid_rc.py --device cuda:0 --steps 1200 --episodes 3
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

import matplotlib
os.environ.setdefault('DISPLAY', ':1')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

# 这里依赖项目内的环境、工具函数和 PID 控制器。
from envs.control_env import ControlEnv                  # noqa: E402
from envs.utils.utils import _t2n                         # noqa: E402
from algorithms.pid.rc_pid import RCPIDController         # noqa: E402


def wrap_pi_np(x):
    """把角度误差包到 [-pi, pi]，避免 heading 跨越 pi/-pi 时误差突变。"""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def scalar(x):
    """把 torch tensor 转成 Python float，便于存入列表和 CSV。"""
    return float(_t2n(x).reshape(-1)[0])


def seed_everything(seed):
    """固定 numpy / random / torch 随机种子，使同一 seed 下的 OU 目标可复现。"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_buf():
    """为单个 episode 创建日志缓存；每个 key 对应一条待绘制或待统计的曲线。"""
    return dict(
        t=[],
        vx=[], vx_tgt=[],
        vz=[], vz_tgt=[],
        heading=[], heading_tgt=[],
        roll=[], pitch=[],
        reward=[],
        f_head=[], f_lift_mean=[], f_lift_spread=[],
    )


def push(buf, env, reward=None):
    """从环境中读取当前状态，并追加到日志缓存。

    注意：这里绘制的 vx 是 local/body forward velocity，而不是世界系
    North 方向速度。它表示飞机沿自身机头方向的前向速度，和
    LearningToFly 中的 local velocity 定义一致。
    """
    vn, ve = env.model.get_ground_speed()
    vz = env.model.get_climb_rate()
    roll, pitch, heading = env.model.get_posture()
    u = env.model.get_control()
    # 将世界系水平速度 (vn, ve) 旋转到机体系，得到机头方向速度 local vx。
    vx = torch.cos(heading) * vn + torch.sin(heading) * ve

    buf['vx'].append(scalar(vx))
    buf['vx_tgt'].append(scalar(env.task.target_vx))
    buf['vz'].append(scalar(vz))
    buf['vz_tgt'].append(scalar(env.task.target_vz))
    buf['heading'].append(scalar(heading))
    buf['heading_tgt'].append(scalar(env.task.target_heading))
    buf['roll'].append(scalar(roll))
    buf['pitch'].append(scalar(pitch))
    buf['reward'].append(0.0 if reward is None else scalar(reward))

    # env.model.get_control() 返回的是动力学最终实际使用的电机推力，单位 N。
    # index 0 是 head/pusher motor，index 1:5 是四个升力电机。
    un = _t2n(u).reshape(-1)
    lift = un[1:]
    buf['f_head'].append(float(un[0]))
    buf['f_lift_mean'].append(float(np.mean(lift)))
    buf['f_lift_spread'].append(float(np.max(lift) - np.min(lift)))


def run_episode(args, device, seed):
    """运行一个 episode，并返回该 episode 的状态、目标和电机输出日志。"""
    env = ControlEnv(num_envs=1, config='rc', model='HYBRID',
                     random_seed=seed, device=device)
    obs = env.reset()
    _ = obs
    pid = RCPIDController(
        n=env.n,
        device=device,
        dt=env.model.dt,
        max_thrust_per_motor=env.model.max_F,
    )
    pid.reset()

    buf = make_buf()
    buf['t'].append(0.0)
    push(buf, env)

    for step in range(1, args.steps + 1):
        # PID 读取 env.task 的目标和 env.model 的当前状态，输出归一化动作 [-1, 1]。
        action = pid.compute_action(env)
        _obs, reward, done, bad_done, exceed, _info = env.step(action)
        reset_mask = done | bad_done | exceed
        if torch.any(reset_mask):
            pid.reset(mask=reset_mask)

        # 横坐标使用真实仿真时间：t = step * dt，而不是纯 step 编号。
        buf['t'].append(step * env.model.dt)
        push(buf, env, reward)

        # 任意终止条件触发后结束当前 episode，例如 extreme state / overload 等。
        if torch.any(reset_mask):
            break

    env.close()
    return buf


def metrics(buf):
    """计算当前 episode 的跟踪指标。

    MAE 表示平均绝对误差；RMSE 对较大的瞬时误差更敏感。
    heading 误差使用 wrap_pi_np 处理，避免角度跨界导致虚假的大误差。
    """
    vx = np.asarray(buf['vx'])
    vz = np.asarray(buf['vz'])
    hdg = np.asarray(buf['heading'])
    vx_t = np.asarray(buf['vx_tgt'])
    vz_t = np.asarray(buf['vz_tgt'])
    hdg_t = np.asarray(buf['heading_tgt'])
    err_h = wrap_pi_np(hdg - hdg_t)
    return {
        'length': len(vx) - 1,
        'return': float(np.sum(buf['reward'])),
        'mae_vx': float(np.mean(np.abs(vx - vx_t))),
        'mae_vz': float(np.mean(np.abs(vz - vz_t))),
        'mae_heading_rad': float(np.mean(np.abs(err_h))),
        'rmse_vx': float(np.sqrt(np.mean((vx - vx_t) ** 2))),
        'rmse_vz': float(np.sqrt(np.mean((vz - vz_t) ** 2))),
        'rmse_heading_rad': float(np.sqrt(np.mean(err_h ** 2))),
    }


def plot(buffers, out_png):
    """把多个 episode 的跟踪曲线和最终电机推力画到同一张图中。"""
    n = len(buffers)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4.0 * n), squeeze=False)

    for i, buf in enumerate(buffers):
        t = np.asarray(buf['t'])

        # 第一列：机体系前向速度 local vx 与目标 vx。
        ax = axes[i, 0]
        ax.plot(t, buf['vx'], label='vx')
        ax.plot(t, buf['vx_tgt'], '--', label='vx target')
        ax.set_title(f'Episode {i} - local vx')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # 第二列：垂直速度 vz 与目标 vz。
        ax = axes[i, 1]
        ax.plot(t, buf['vz'], label='vz')
        ax.plot(t, buf['vz_tgt'], '--', label='vz target')
        ax.set_title(f'Episode {i} - vz')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # 第三列：航向角。np.unwrap 用于让曲线跨越 pi/-pi 时保持连续。
        ax = axes[i, 2]
        ax.plot(t, np.unwrap(buf['heading']), label='heading')
        ax.plot(t, np.unwrap(buf['heading_tgt']), '--', label='heading target')
        ax.set_title(f'Episode {i} - heading')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # 第四列：动力学最终使用的电机推力，而不是 PID 原始动作。
        ax = axes[i, 3]
        ax.plot(t, buf['f_head'], label='F_head')
        ax.plot(t, buf['f_lift_mean'], label='F_1-4 mean')
        ax.plot(t, buf['f_lift_spread'], label='F_1-4 spread')
        ax.set_title(f'Episode {i} - final motor force')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f'[plot] saved: {out_png}')


def save_csv(rows, out_csv):
    """把每个 episode 的指标保存成 CSV，方便后续和 RL / DAgger 做表格比较。"""
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    """命令行参数。

    steps 是每个 episode 最多运行多少个仿真 step；真实时间长度为
    ``steps * env.model.dt``。
    """
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--out-dir', type=str,
                   default=os.path.join(ROOT, 'renders', 'result', 'rc_pid'))
    p.add_argument('--no-show', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    # 如果用户指定 cuda 但当前机器没有可用 CUDA，则自动回退到 CPU。
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buffers = []
    rows = []
    for ep in range(args.episodes):
        # 每个 episode 使用不同 seed，让 OU 目标过程不同。
        buf = run_episode(args, device, args.seed + ep)
        m = metrics(buf)
        m['episode'] = ep
        rows.append(m)
        buffers.append(buf)
        print(
            f"[pid-rc] ep={ep} len={m['length']} return={m['return']:.1f} "
            f"mae_vx={m['mae_vx']:.3f} mae_vz={m['mae_vz']:.3f} "
            f"mae_hdg={m['mae_heading_rad']:.3f}"
        )

    # 输出指标表和曲线图。
    save_csv(rows, out_dir / 'pid_rc_metrics.csv')
    plot(buffers, out_dir / 'pid_rc_tracking.png')

    print('[pid-rc] average:')
    for key in ('return', 'mae_vx', 'mae_vz', 'mae_heading_rad',
                'rmse_vx', 'rmse_vz', 'rmse_heading_rad'):
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        print(f'  {key}: {vals.mean():.4f} +/- {vals.std():.4f}')

    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
