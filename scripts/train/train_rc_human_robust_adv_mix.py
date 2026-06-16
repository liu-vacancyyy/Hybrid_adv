#!/usr/bin/env python
import datetime
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from algorithms.adversarial.rc_human_adv_mix_env import RCHumanAdvMixTrainEnv  # noqa: E402
from config import get_config                                                   # noqa: E402
from envs.control_env import ControlEnv                                         # noqa: E402
from envs.env_wrappers import GPUVecEnv                                         # noqa: E402
from runner.F16sim_runner import F16SimRunner                                   # noqa: E402


def add_adv_mix_args(parser):
    group = parser.add_argument_group("RC human robust adversarial mix")
    group.add_argument("--adv-ckpt", type=str, required=True,
                       help="Frozen adversary actor checkpoint.")
    group.add_argument("--adv-mix-frac", type=float, default=0.10,
                       help="Fraction of vectorized envs driven by adversarial command/obs/wind.")
    group.add_argument("--uniform-curriculum-levels", type=int, default=120,
                       help="Number of curriculum levels sampled uniformly by the non-adversarial envs.")
    group.add_argument("--stochastic-adv", action="store_true", default=False,
                       help="Sample adversary actions instead of using deterministic means.")

    group.add_argument("--adv-hidden-size", type=str, default="128 128 128")
    group.add_argument("--adv-activation-id", type=int, default=1)
    group.add_argument("--adv-command-frac", type=float, default=1.0)
    group.add_argument("--adv-obs-frac", type=float, default=1.0)
    group.add_argument("--adv-wind-frac", type=float, default=1.0)
    group.add_argument("--adv-command-alpha", type=float, default=1.0)
    group.add_argument("--adv-obs-alpha", type=float, default=1.0)
    group.add_argument("--adv-wind-alpha", type=float, default=1.0)
    group.add_argument("--adv-command-rate-limit-frac", type=float, default=0.0)
    group.add_argument("--adv-obs-rate-limit-frac", type=float, default=0.1)
    group.add_argument("--adv-wind-rate-limit-frac", type=float, default=0.1)
    group.add_argument("--adv-obs-default-scale", type=float, default=0.02)
    group.add_argument("--adv-obs-max-scale", type=float, default=0.10)
    return parser


def parse_args(args):
    parser = add_adv_mix_args(get_config())
    group = parser.add_argument_group("F16Sim Env parameters")
    group.add_argument("--env-name", type=str, default="Control")
    group.add_argument("--scenario-name", type=str, default="rc_human")
    group.add_argument("--model-name", type=str, default="HYBRID_NEW")
    return parser.parse_known_args(args)[0]


def make_train_env(all_args, device):
    def get_env_fn():
        def init_env():
            if all_args.env_name != "Control":
                raise NotImplementedError("robust adv-mix training currently supports Control only")
            base_env = ControlEnv(
                num_envs=all_args.n_rollout_threads,
                config=all_args.scenario_name,
                model=all_args.model_name,
                random_seed=all_args.seed,
                device=device,
            )
            return RCHumanAdvMixTrainEnv(base_env, all_args, device)
        return init_env
    return GPUVecEnv([get_env_fn()])


def make_eval_env(all_args, device):
    def get_env_fn():
        def init_env():
            return ControlEnv(
                num_envs=all_args.n_eval_rollout_threads,
                config=all_args.scenario_name,
                model=all_args.model_name,
                random_seed=all_args.seed * 50000,
                device=device,
            )
        return init_env
    return GPUVecEnv([get_env_fn()])


def main(args):
    all_args = parse_args(args)
    print(all_args)

    os.environ.setdefault("RC_HUMAN_MODE_ORDER", "0 1 2 5 3 4")
    os.environ.setdefault("RC_HUMAN_MAX_MODE_SLOTS", "6")

    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device(all_args.device)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        all_args.device = "cpu"
        all_args.cuda = False

    run_dir = (
        ROOT / "scripts" / "runs"
        / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
          f"{all_args.env_name}_{all_args.scenario_name}_{all_args.model_name}_"
          f"{all_args.algorithm_name}_{all_args.experiment_name}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = tb.SummaryWriter(run_dir)

    envs = make_train_env(all_args, device)
    eval_envs = make_eval_env(all_args, device) if all_args.use_eval else None
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
    }
    runner = F16SimRunner(config, writer)
    try:
        runner.run()
    except BaseException:
        traceback.print_exc()
    finally:
        envs.close()
        if eval_envs is not None:
            eval_envs.close()
        writer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
