# SPDX-License-Identifier: Apache-2.0
#
# OMX-F 텔레옵 + 데이터 녹화 에이전트 (워크숍 scripts/lerobot_agent.py 의 OMX 대응 신규 파일).
# 실물 OMX-L 리더 암으로 Isaac Lab 환경(Lerobot-Omx-Teleop-*) 안의 omx_f 를 조종하고,
# 카메라 영상(ego/external) + 관절 상태/액션을 LeRobotDataset 으로 녹화한다.
#
# 예)
#   python source/sim_to_real_so101/scripts/omx_lerobot_agent.py \
#       --task Lerobot-Omx-Teleop-Vials-To-Rack-DR --port /dev/ttyACM0 --robot_id omx_leader_arm \
#       --repo_id <hf-user>/omx_vials --repo_root datasets/omx_vials --task_name "Pick up the vial and place it in the rack"
#
# 키: R = world reset, S = 녹화 시작/종료  (워크숍과 동일)
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab OMX-F Teleop agent.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None, help="예: Lerobot-Omx-Teleop-Vials-To-Rack-DR")
parser.add_argument("--port", type=str, default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"))
parser.add_argument("--robot_id", type=str, default=os.getenv("TELEOP_ID", "omx_leader_arm"))
parser.add_argument("--leader_type", type=str, default="omx", choices=["omx", "so101"])
parser.add_argument(
    "--zero_mode",
    type=str,
    default="json",
    choices=["json", "capture", "none"],
    help="json: 저장된 zero 정렬 json 사용(없으면 0 기준) / capture: 시작 시 리더 현재 자세를 zero 로 / none: zero=0",
)
parser.add_argument("--zero_json", type=str, default=None, help="zero json 경로(기본: HF_LEROBOT_HOME/calibration/...)")
parser.add_argument("--gripper_closed_norm", type=float, default=None, help="트리거 꽉 닫힘 normalized 값(리더 개체별 실측)")
parser.add_argument("--gripper_open_norm", type=float, default=None, help="트리거 끝까지 연 normalized 값(리더 개체별 실측)")
parser.add_argument("--repo_id", type=str, default=None)
parser.add_argument("--repo_root", type=str, default=None)
parser.add_argument("--save_mp4", action="store_true", default=False)
parser.add_argument("--depth", action="store_true", default=False)
parser.add_argument("--instance_id_seg", action="store_true", default=False)
parser.add_argument("--task_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=101)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401  (Lerobot-Omx-* 등록)
from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.omx_recorder import OmxLeRobotRecorder as LeRobotRecorder
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
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print("[INFO]: Click 'R' to reset the world")
    print("[INFO]: Click 'S' to start/stop recording; 'R' will also stop recording")

    env.reset()

    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": camera_cfg.height, "width": camera_cfg.width}
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")
    if len(cameras) == 0:
        print("[Info]: No cameras found - videos will not be recorded")

    robot_iface = OmxInterface(
        device=env.unwrapped.device,
        port=args_cli.port,
        id=args_cli.robot_id,
        cameras=cameras,
        fps=30,
        kind="leader",
        leader_type=args_cli.leader_type,
        zero_json=args_cli.zero_json if args_cli.zero_mode == "json" else "/nonexistent/zero.json",
        gripper_closed_norm=args_cli.gripper_closed_norm,
        gripper_open_norm=args_cli.gripper_open_norm,
    )
    robot_iface.init_device()
    robot_iface.connect()

    if args_cli.zero_mode == "capture":
        print("[OMX] 리더를 화면 속 로봇의 초기 자세와 똑같이 맞춘 상태여야 한다. 캡처 중...")
        first = robot_iface.robot.get_action()
        robot_iface.set_zero_norm(robot_iface.get_raw_actions_tensor(first).cpu().tolist())
        print("[OMX][WARN] capture 모드의 action 은 실물 팔로워 공간과 다를 수 있다. 가능하면 json/none 사용 권장.")

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    recording_mode = all([args_cli.repo_id, args_cli.repo_root, args_cli.task_name])
    if recording_mode:
        recorder = LeRobotRecorder(
            task_name=args_cli.task_name,
            repo_id=args_cli.repo_id,
            dataset_root=args_cli.repo_root,
            fps=30,
            device=env.unwrapped.device,
            cameras=cameras,
            save_mp4=args_cli.save_mp4,
            depth=args_cli.depth,
            instance_id_seg=args_cli.instance_id_seg,
        )
        try:
            recorder.init_dataset()
        except ValueError:
            print("[ERROR]: Failed to initialize dataset. folder already exists")
            env.close()
            simulation_app.close()

    while simulation_app.is_running():
        with torch.inference_mode():
            real_action = robot_iface.robot.get_action()
            real_action, mapped_action = robot_iface.real_to_sim_obs_processor(real_action)
            actions[:] = mapped_action

            obs, _, _, _, _ = env.step(actions)

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                env.reset()
                continue

            if recording_mode and keyboard_control.recording:
                visual_obs = obs.get("visual", None)
                if visual_obs is None:
                    print("[WARNING]: No 'visual' observation group - recording requires a task with cameras")
                    keyboard_control.recording = False
                    continue
                joint_pos_obs = obs["policy"]["joint_pos_obs"][0]  # (6,) rad, joint1~5 + gripper_joint_1
                real_obs, visual_buffers, depth_buffers, instance_id_seg_buffers = (
                    robot_iface.sim_to_real_dataset_processor(joint_pos_obs, visual_obs)
                )
                recorder.push_frame_to_buffer(
                    real_action, real_obs, visual_buffers, depth_buffers, instance_id_seg_buffers
                )

    env.close()


if __name__ == "__main__":
    main()

    while True:
        simulation_app.update()

    simulation_app.close()
