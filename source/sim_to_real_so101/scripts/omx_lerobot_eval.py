# SPDX-License-Identifier: Apache-2.0
#
# OMX-F 평가 클라이언트 (워크숍 scripts/lerobot_eval.py 의 OMX 대응 신규 파일).
# Isaac Lab 환경(Lerobot-Omx-Teleop-*-Eval)을 돌리고, 원격 GR00T 추론 서버에 관측을 보내 액션을 받아 실행한다.
#
# 예)
#   python source/sim_to_real_so101/scripts/omx_lerobot_eval.py \
#       --task Lerobot-Omx-Teleop-Vials-To-Rack-Eval --policy_host localhost --policy_port 5555 \
#       --rename_map '{"ego": "front", "external_D455": "top"}' --num_episodes 10
#
# 주의: --zero_json / --gripper_*_norm 은 데이터를 녹화(omx_lerobot_agent.py)할 때와 같은 값을 써야 한다.
#       (정책이 내놓는 정규화 값 -> sim 각도 변환이 녹화 때의 역변환과 정확히 대응해야 하기 때문)
import argparse
import json
import random

from tqdm import tqdm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab OMX-F Eval Client (remote GR00T inference server).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=1984)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--rename_map", type=str, default=None, help='sim 카메라 이름 -> 정책 카메라 키 JSON')
parser.add_argument("--policy_host", type=str, default="localhost")
parser.add_argument("--policy_port", type=int, default=5555)
parser.add_argument("--action_horizon", type=int, default=16)
parser.add_argument("--lang_instruction", type=str, default="Pick up the vial and place it in the rack")
parser.add_argument("--rerun", action="store_true", default=False)
parser.add_argument("--zero_json", type=str, default="/nonexistent/zero.json", help="녹화 때 쓴 zero json (기본: zero=0)")
parser.add_argument("--gripper_closed_norm", type=float, default=None)
parser.add_argument("--gripper_open_norm", type=float, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.lerobot_interface import GR00TRemotePolicy
from sim_to_real_so101.utils.omx_interface import OmxInterface


def main():
    keyboard_control = KeyboardControl()

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.cuda.manual_seed_all(args_cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print("[INFO]: Click 'R' to reset the world")

    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": camera_cfg.height, "width": camera_cfg.width}
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")
    if len(cameras) == 0:
        print("[Info]: No cameras found - videos will not be recorded")

    rename_map = json.loads(args_cli.rename_map) if args_cli.rename_map else None
    robot_iface = OmxInterface(
        device=env.unwrapped.device,
        port=None,
        id="omx_leader_arm",
        cameras=cameras,
        fps=30,
        kind="follower",
        rename_map=rename_map,
        zero_json=args_cli.zero_json,
        gripper_closed_norm=args_cli.gripper_closed_norm,
        gripper_open_norm=args_cli.gripper_open_norm,
    )
    robot_iface.init_device(visualize=args_cli.rerun)

    policy = GR00TRemotePolicy(
        robot_iface=robot_iface,
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        action_horizon=args_cli.action_horizon,
        lang_instruction=args_cli.lang_instruction,
    )
    policy.connect()

    obs, _ = env.reset()
    policy.reset()

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    # OMX 의 기본(zero) 자세. SO-101 워크숍은 비-zero 초기 자세 상수를 썼지만 OMX 는 zero 가 초기 자세다.
    initial_action = torch.zeros(env.action_space.shape[-1], device=env.unwrapped.device)

    step = 0
    num_episodes = 0
    num_successes = 0
    success_rate = 0.0
    pbar = None

    while simulation_app.is_running():
        with torch.inference_mode():
            if step == 0:
                pbar = tqdm(
                    total=env.unwrapped.max_episode_length,
                    desc=f"Rollout (ep {num_episodes + 1}, success: {success_rate:.1f}%)",
                    unit="step",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                )

            if step < 10:  # warmup
                actions[:] = initial_action
            else:
                joint_positions = obs["policy"]["joint_pos_obs"][0].clone()  # (6,) rad
                actions[:] = policy.get_action(joint_positions, obs["visual"], log=True)

            obs, rewards, terminated, truncated, info = env.step(actions)
            step += 1
            if pbar is not None:
                pbar.update(1)

            is_terminated = terminated.item() if terminated.numel() == 1 else terminated.any().item()
            is_truncated = truncated.item() if truncated.numel() == 1 else truncated.any().item()

            if is_terminated or is_truncated:
                if pbar is not None:
                    pbar.close()
                    pbar = None
                num_episodes += 1
                if is_terminated and not is_truncated:
                    num_successes += 1
                success_rate = (num_successes / num_episodes) * 100
                obs, _ = env.reset()
                policy.reset()
                step = 0
                continue

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                if pbar is not None:
                    pbar.close()
                    pbar = None
                print(f"[MANUAL RESET] Episode interrupted at step {step}")
                obs, _ = env.reset()
                policy.reset()
                step = 0
                continue

            if num_episodes >= args_cli.num_episodes:
                if pbar is not None:
                    pbar.close()
                    pbar = None
                print(f"[INFO]: Evaluated {args_cli.num_episodes} episodes")
                print(f"[INFO]: Success Rate: {num_successes}/{args_cli.num_episodes} ({success_rate:.1f}%)")
                env.close()
                simulation_app.close()

    env.close()


if __name__ == "__main__":
    main()

    while True:
        simulation_app.update()

    simulation_app.close()
