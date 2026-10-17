# SPDX-License-Identifier: Apache-2.0
#
# OMX Cube-To-Cup PPO 학습 (RSL-RL). 카메라가 없으므로 headless 로 돌리는 것이 기본이다.
#
#   python source/sim_to_real_so101/scripts/omx_rl_train.py --headless --num_envs 512
#   python source/sim_to_real_so101/scripts/omx_rl_train.py --headless --max_iterations 20 --num_envs 64   # 동작 확인
#   (이어서 학습)  ... --resume outputs/rl/omx_cube_to_cup/<run>/model_500.pt
#
# 로그/체크포인트: outputs/rl/omx_cube_to_cup/<시각>/   (tensorboard --logdir outputs/rl)
import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train OMX Cube-To-Cup with RSL-RL PPO.")
parser.add_argument("--task", type=str, default="Lerobot-Omx-Cube-To-Cup-RL")
parser.add_argument("--num_envs", type=int, default=None, help="기본 512")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_iterations", type=int, default=None, help="기본 1500")
parser.add_argument("--resume", type=str, default=None, help="이어서 학습할 model_*.pt 경로")
parser.add_argument("--run_name", type=str, default="")
parser.add_argument(
    "--noise_std", type=float, default=None,
    help="--resume 후 행동 잡음 표준편차를 이 값으로 다시 설정해 탐색을 늘린다 (예: 0.5)",
)
parser.add_argument("--log_root", type=str, default="outputs/rl")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.tasks.omx_rl_agent_cfg import OmxCubeToCupPPORunnerCfg


def set_noise_std(runner, value: float):
    """정책의 행동 잡음 파라미터(std 또는 log_std)를 value 로 덮어쓴다. rsl_rl 버전마다 이름이 달라 이름으로 찾는다."""
    import math

    import torch

    policy = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor_critic", None)
    found = []
    for name, p in policy.named_parameters():
        leaf = name.split(".")[-1]
        if leaf == "log_std" or leaf == "log_std_param":
            with torch.no_grad():
                p.fill_(math.log(value))
            found.append(name)
        elif leaf in ("std", "std_param"):
            with torch.no_grad():
                p.fill_(value)
            found.append(name)
    if not found:
        print("[train][WARN] 잡음 파라미터를 찾지 못했다. 파라미터 이름:", [n for n, _ in policy.named_parameters()])
    else:
        print(f"[train] 행동 잡음 std -> {value}  (설정한 파라미터: {found})")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    agent_cfg = OmxCubeToCupPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args_cli.run_name}" if args_cli.run_name else "")
    log_dir = os.path.abspath(os.path.join(args_cli.log_root, agent_cfg.experiment_name, stamp))
    print(f"[train] log_dir: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if args_cli.resume:
        print(f"[train] resume: {args_cli.resume}")
        runner.load(args_cli.resume)
    if args_cli.noise_std is not None:
        set_noise_std(runner, args_cli.noise_std)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
