# SPDX-License-Identifier: Apache-2.0
#
# OMX Cube-To-Cup RL 환경 점검 (학습 전에 한 번 실행). 환경이 만들어지는지, 관찰/액션 차원, 액션 방향과 처짐,
# 랜덤 액션에서 보상/종료가 어떻게 나오는지 출력한다.
#
#   python source/sim_to_real_so101/scripts/omx_rl_check.py --headless --num_envs 8
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OMX Cube-To-Cup RL env check.")
parser.add_argument("--task", type=str, default="Lerobot-Omx-Cube-To-Cup-RL")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.assets.omx import OMX_ARM_JOINTS


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    ids = robot.find_joints(OMX_ARM_JOINTS, preserve_order=True)[0]
    n, dev = u.num_envs, u.device

    obs, _ = env.reset()
    print(f"[check] num_envs={n}  obs={tuple(obs['policy'].shape)}  action_space={env.action_space.shape}")
    print(f"[check] step_dt={u.step_dt:.4f} s  max_episode_length={u.max_episode_length} steps")
    print(f"[check] 보상 항: {list(u.reward_manager.active_terms)}")
    print(f"[check] 종료 항: {list(u.termination_manager.active_terms)}")
    print(f"[check] 관절 한계(soft) gripper_joint_1: {robot.data.soft_joint_pos_limits[0, ids[-1]].tolist()}")

    def jp():
        return robot.data.joint_pos[0, ids].tolist()

    def fmt(v):
        return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"

    zero = torch.zeros(n, 6, device=dev)
    p0 = jp()
    for _ in range(30):
        env.step(zero)
    print(f"[check] 액션 0 으로 30 스텝: 관절 변화(처짐) {fmt([b - a for a, b in zip(p0, jp())])}")

    g = zero.clone()
    g[:, 5] = 1.0
    for _ in range(30):
        env.step(g)
    print(f"[check] gripper 액션 +1 x30: gripper_joint_1 = {jp()[5]:+.3f} rad")
    g[:, 5] = -1.0
    for _ in range(60):
        env.step(g)
    print(f"[check] gripper 액션 -1 x60: gripper_joint_1 = {jp()[5]:+.3f} rad")

    env.reset()
    tot = torch.zeros(n, device=dev)
    n_term = n_trunc = 0
    steps = 200
    for _ in range(steps):
        a = 2 * torch.rand(n, 6, device=dev) - 1
        _, r, term, trunc, _ = env.step(a)
        tot += r
        n_term += int(term.sum())
        n_trunc += int(trunc.sum())
    print(f"[check] 랜덤 액션 {steps} 스텝: 환경당 누적 보상 평균 {tot.mean():.2f}, terminated {n_term}, truncated {n_trunc}")
    cube = u.scene["cube"].data.root_pos_w[0] - u.scene.env_origins[0]
    ee = u.scene["ee_frame"].data.target_pos_w[0, 0] - u.scene.env_origins[0]
    print(f"[check] env0 cube={fmt(cube.tolist())} ee={fmt(ee.tolist())}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
