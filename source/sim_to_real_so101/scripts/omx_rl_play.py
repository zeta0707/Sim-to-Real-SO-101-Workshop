# SPDX-License-Identifier: Apache-2.0
#
# 학습된 PPO 정책을 시뮬레이터에서 돌려 성공률을 센다.
#
#   python source/sim_to_real_so101/scripts/omx_rl_play.py --headless --checkpoint outputs/rl/omx_cube_to_cup/<run>/model_1499.pt --num_episodes 100
#   (화면으로 보기)  ... --num_envs 4 --num_episodes 8      (--headless 없이)
#
# 성공 = 'success'(cube_in_cup) 종료. 모방학습 평가(Cube-To-Cup-Eval)와 같은 판정이다.
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play/evaluate an OMX Cube-To-Cup PPO policy.")
parser.add_argument("--task", type=str, default="Lerobot-Omx-Cube-To-Cup-RL")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=64)
parser.add_argument("--seed", type=int, default=1984)
parser.add_argument(
    "--force_open_at", type=int, default=None,
    help="이 스텝부터 그리퍼 액션을 +1(열기)로 강제한다. 열면 큐브가 떨어지는지 확인하는 실험용 (--probe 와 같이 사용)",
)
parser.add_argument(
    "--auto_release", action="store_true",
    help="규칙 추가: 큐브가 컵 위(xy 2 cm 안, z>0.075)에서 거의 멈춘 채 --release_dwell 스텝이 지나면 그리퍼를 열고 에피소드 끝까지 유지",
)
parser.add_argument("--release_speed", type=float, default=100.0, help="--auto_release: 이 속도(m/s) 미만일 때만 발동 (기본값은 사실상 비활성: 큐브가 떨려서 속도가 항상 0.3~0.4)")
parser.add_argument("--release_tol", type=float, default=0.03, help="--auto_release: 큐브가 컵 중심 xy 에서 이 거리(m) 안이면 컵 위로 본다")
parser.add_argument("--release_dwell", type=int, default=15, help="--auto_release: 멈춘 상태를 유지해야 하는 스텝 수 (30 Hz 기준 15 = 0.5 s)")
parser.add_argument("--probe", action="store_true", help="env 0 의 그리퍼 각도/큐브 높이/컵까지 거리를 10 스텝마다 출력 (보상 임계값 확인용)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from sim_to_real_so101.assets.omx import OMX_ARM_JOINTS
from sim_to_real_so101.tasks.omx_cube_to_cup_env_cfg import CUP_POS

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.tasks.omx_rl_agent_cfg import OmxCubeToCupPPORunnerCfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = OmxCubeToCupPPORunnerCfg()
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    tm = env.unwrapped.termination_manager
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    episodes = successes = lost = 0
    robot = env.unwrapped.scene["robot"]
    grip_id = robot.find_joints(OMX_ARM_JOINTS, preserve_order=True)[0][-1]
    finger_ids = robot.find_bodies(["link6", "link7"], preserve_order=True)[0]
    cube = env.unwrapped.scene["cube"]
    origins = env.unwrapped.scene.env_origins
    step = 0
    release_events = 0
    print(
        f"[play] auto_release={args_cli.auto_release} release_dwell={args_cli.release_dwell} "
        f"release_speed={args_cli.release_speed} release_tol={args_cli.release_tol} force_open_at={args_cli.force_open_at}"
    )
    n_env = env.unwrapped.num_envs
    dev = env.unwrapped.device
    dwell = torch.zeros(n_env, dtype=torch.long, device=dev)
    released = torch.zeros(n_env, dtype=torch.bool, device=dev)
    cup_t = torch.tensor(CUP_POS, device=dev)
    step_dt = env.unwrapped.step_dt
    prev_cpos = (cube.data.root_pos_w - origins).clone()
    with torch.inference_mode():
        while simulation_app.is_running() and episodes < args_cli.num_episodes:
            actions = policy(obs)
            if args_cli.force_open_at is not None and step >= args_cli.force_open_at:
                actions[:, 5] = 1.0
            cpos = cube.data.root_pos_w - origins
            cd = torch.norm(cpos[:, :2] - cup_t, dim=1)
            # 시뮬레이터가 보고하는 root_lin_vel_w 는 끼인 접촉에서 고정값(0.32 m/s)으로 남는 경우가 있어서 쓰지 않고,
            # 스텝 사이 위치 변화로 속도를 직접 계산한다.
            speed = torch.norm(cpos - prev_cpos, dim=1) / step_dt
            prev_cpos = cpos.clone()
            if args_cli.auto_release:
                ok = (cpos[:, 2] > 0.075) & (cd < args_cli.release_tol) & (speed < args_cli.release_speed)
                dwell = torch.where(ok, dwell + 1, torch.zeros_like(dwell))
                newly = (dwell >= args_cli.release_dwell) & ~released
                release_events += int(newly.sum())
                released |= newly
                actions[released, 5] = 1.0
            obs, _, dones, _ = env.step(actions)
            if args_cli.auto_release and dones.any():
                d_mask = dones.bool()
                dwell[d_mask] = 0
                released[d_mask] = False
            step += 1
            if args_cli.probe and step % 10 == 0:
                pos = cube.data.root_pos_w[0] - origins[0]
                d = ((pos[0] - CUP_POS[0]) ** 2 + (pos[1] - CUP_POS[1]) ** 2) ** 0.5
                f6, f7 = robot.data.body_pos_w[0, finger_ids[0]], robot.data.body_pos_w[0, finger_ids[1]]
                gap = float(torch.norm(f6 - f7))
                mid = 0.5 * (f6 + f7)
                c_mid = float(torch.norm(cube.data.root_pos_w[0] - mid))
                print(
                    f"[probe] step={step:5d} gripper={float(robot.data.joint_pos[0, grip_id]):.3f} rad "
                    f"cube_z={float(pos[2]):.3f} cup_dist_xy={float(d):.3f} "
                    f"finger_gap={gap:.3f} cube_to_finger_mid={c_mid:.3f} "
                    f"speed={float(speed[0]):.3f} dwell={int(dwell[0])} released={bool(released[0])} done={bool(dones[0])}"
                )
            if dones.any():
                succ = tm.get_term("success")
                out = tm.get_term("cube_out")
                for i in torch.nonzero(dones).flatten().tolist():
                    episodes += 1
                    successes += int(succ[i])
                    lost += int(out[i])
    print(f"[play] checkpoint: {args_cli.checkpoint}")
    print(f"[play] 규칙(auto_release)이 발동한 횟수: {release_events}")
    print(f"[play] Success Rate: {successes}/{episodes} ({100.0 * successes / max(episodes, 1):.1f}%)  cube_lost={lost}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
