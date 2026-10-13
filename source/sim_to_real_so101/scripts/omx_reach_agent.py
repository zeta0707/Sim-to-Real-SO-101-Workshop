# SPDX-License-Identifier: Apache-2.0
#
# OMX-F 도달 테스트 (리더 없이 실행). 평면 IK 로 바이알 위치까지 팔을 움직여 닿는지 확인한다.
#
# 예)
#   python source/sim_to_real_so101/scripts/omx_reach_agent.py --task Lerobot-Omx-Teleop-Vials-To-Rack --target vial_2
#   python source/sim_to_real_so101/scripts/omx_reach_agent.py --task Lerobot-Omx-Teleop-Vials-To-Rack --target vial_3 --pitch 45
#
# 주의: omx_f 의 그리퍼 축(link5 z)은 롤 0 에서 월드 y 라서 핑거가 위/아래로(피치 평면 안에서) 열린다.
#       그래서 --roll 90 으로 손목을 돌려야 핑거가 좌우로 열려 바닥의 바이알을 옆에서 잡을 수 있다.
# 동작: 영 자세 -> 바이알 위 -> 바이알 높이로 하강(그리퍼 열림) -> 닫기 -> 들어올리기 -> 랙 위로 이동 -> 열기.
# 각 단계에서 IK 가 관절 한계 안에서 풀렸는지, 실제 ee 위치와 목표의 오차가 몇 mm 인지 출력한다.
# 관절 기하(USD 에서 읽은 값)는 아래 상수와 같아야 한다.
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OMX-F reach test (planar IK).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Lerobot-Omx-Teleop-Vials-To-Rack")
parser.add_argument("--target", type=str, default="vial_2", help="vial_1 | vial_2 | vial_3")
parser.add_argument("--rack", type=str, default="rack_left")
parser.add_argument("--pitch", type=float, default=20.0, help="그리퍼 접근 각도(deg, 90=수직 아래, 0=수평)")
parser.add_argument(
    "--roll", type=float, default=90.0,
    help="손목 롤 joint5 (deg). omx_f 는 롤 0 에서 핑거가 세로(피치 평면)로 열린다. 바닥에 누운 바이알을 옆에서 집으려면 90",
)
parser.add_argument("--open_deg", type=float, default=45.0, help="그리퍼 열림 각도 (deg, gripper_joint_1). 0=닫힘, 100=최대")
parser.add_argument("--tip_floor", type=float, default=None, help="(선택) 손가락 끝이 내려갈 수 있는 최저 높이 (m) 로 파지 높이를 제한. 기본은 제한 없음. 매트 윗면은 약 0.035")
parser.add_argument(
    "--seat", type=float, default=0.01,
    help="ee 를 바이알 중심보다 접근 방향으로 더 보내는 거리 (m). 손가락 패드 중앙이 ee 보다 약 0.025 m 뒤쪽이라 바이알이 패드 중앙에 오도록 한다",
)
parser.add_argument("--hover_dz", type=float, default=0.08, help="바이알 위 대기 높이 (m)")
parser.add_argument("--grasp_dz", type=float, default=0.0, help="바이알 중심 대비 그리퍼 높이 오프셋 (m)")
parser.add_argument("--place", action="store_true", default=False, help="들어올린 뒤 랙 위로 옮겨 놓는 단계까지 실행 (기본: 들어올리기까지만)")
parser.add_argument("--axis_offset", type=float, default=0.0413, help="바이알 원점에서 몸통 중심까지 축 방향 거리 (m). Vial_opaque.usda 의 bbox 중심")
parser.add_argument("--seg_sec", type=float, default=2.0, help="단계별 이동 시간 (s)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401

# omx_f_isaaclab.usda 에서 읽은 기하 (m)
SHOULDER_DX, SHOULDER_Z = -0.0113, 0.0975  # joint1 축의 베이스 기준 x 오프셋, joint2 높이(0.034+0.0635)
A_X, A_Z = 0.0415, 0.1132  # joint2 -> joint3
L2 = 0.162  # joint3 -> joint4
L3 = 0.0287 + 0.092  # joint4 -> ee (joint5 + end_effector_link)
L1 = math.hypot(A_X, A_Z)
ALPHA = math.atan2(A_Z, A_X)
LIMITS_DEG = {  # q2, q3, q4
    "q2": (-120.0, 90.0),
    "q3": (-120.0, 90.0),
    "q4": (-100.0, 100.0),
}


def planar_fk(q2, q3, q4):
    """(수평거리 r, 높이 z[어깨 기준 +SHOULDER_Z 포함 안 함], 그리퍼 pitch) -- 어깨 축 기준."""
    e1 = ALPHA - q2
    e2 = -(q2 + q3)
    e3 = -(q2 + q3 + q4)
    r = L1 * math.cos(e1) + L2 * math.cos(e2) + L3 * math.cos(e3)
    z = L1 * math.sin(e1) + L2 * math.sin(e2) + L3 * math.sin(e3)
    return r, z


def planar_ik(r, z, pitch):
    """어깨 축 기준 (r, z) 에 ee 를 pitch(아래 방향 +, rad)로 두는 q2,q3,q4 (elbow-up). 실패하면 None."""
    wr = r - L3 * math.cos(pitch)
    wz = z + L3 * math.sin(pitch)
    d2 = wr * wr + wz * wz
    c = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c) > 1.0:
        return None
    g = -math.acos(c)  # elbow-up
    e1 = math.atan2(wz, wr) - math.atan2(L2 * math.sin(g), L1 + L2 * math.cos(g))
    e2 = e1 + g
    q2 = ALPHA - e1
    q3 = -e2 - q2
    q4 = pitch - (q2 + q3)
    return q2, q3, q4


def solve_pose(target_w, shoulder_w, pitch_deg, grip_rad):
    """월드 좌표 target -> 6축 관절 목표(rad). 반환: (q 리스트, 한계 위반 문자열)"""
    dx, dy = target_w[0] - shoulder_w[0], target_w[1] - shoulder_w[1]
    q1 = math.atan2(dy, dx)
    r = math.hypot(dx, dy)
    z = target_w[2] - shoulder_w[2]
    sol = planar_ik(r, z, math.radians(pitch_deg))
    if sol is None:
        return None, f"도달 불가 (r={r:.3f} m, z={z:.3f} m)"
    q2, q3, q4 = sol
    bad = [
        f"{n}={math.degrees(v):.0f}deg(한계 {lo:.0f}..{hi:.0f})"
        for n, v in zip(("q2", "q3", "q4"), sol)
        for lo, hi in [LIMITS_DEG[n]]
        if not (lo <= math.degrees(v) <= hi)
    ]
    return [q1, q2, q3, q4, math.radians(args_cli.roll), grip_rad], ("; ".join(bad) if bad else "")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    uenv = env.unwrapped
    dev = uenv.device
    scene = uenv.scene
    origin = scene.env_origins[0]

    robot_root = (scene["robot"].data.root_pos_w[0] - origin).cpu().tolist()
    shoulder = [robot_root[0] + SHOULDER_DX, robot_root[1], robot_root[2] + SHOULDER_Z]
    vial_root = (scene[args_cli.target].data.root_pos_w[0] - origin).cpu().tolist()
    # Vial_opaque.usda 의 원점은 원뿔 끝 쪽에 있고 로컬 +z 가 몸통 축(길이 -0.017..0.0995)이다.
    # 원점을 그대로 집으면 원뿔 끝을 잡게 되므로, 루트 자세로 축 방향을 구해 몸통 중심으로 옮긴다.
    w, x, y, z = scene[args_cli.target].data.root_quat_w[0].cpu().tolist()
    axis = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]  # R * (0,0,1)
    vial = [vial_root[i] + args_cli.axis_offset * axis[i] for i in range(3)]
    print(f"[INFO] 바이알 축 방향(월드)={[round(v,3) for v in axis]}, 몸통 중심={[round(v,3) for v in vial]}")
    rack = (scene[args_cli.rack].data.root_pos_w[0] - origin).cpu().tolist()
    print(f"[INFO] robot root={robot_root}, {args_cli.target}(원점)={vial_root}, {args_cli.rack}={rack}")
    print(f"[INFO] 어깨~바이알 수평거리 = {math.hypot(vial[0]-shoulder[0], vial[1]-shoulder[1]):.3f} m")

    OPEN, CLOSED = math.radians(args_cli.open_deg), 0.0
    # 손가락 끝은 ee 보다 접근 방향으로 0.021 m 더 나가 있다(omx_f 핑거 메시 범위로 계산). 끝이 매트(약 0.035 m)에
    # 파묻히지 않도록 ee 높이의 하한을 둔다.
    min_ee_z = (
        args_cli.tip_floor + 0.021 * math.sin(math.radians(args_cli.pitch)) if args_cli.tip_floor is not None else -1e9
    )
    # 접근 방향(수평 방위 + pitch)으로 --seat 만큼 더 보내 바이알이 손가락 패드 중앙(ee 보다 뒤쪽)에 오게 한다.
    hx, hy = vial[0] - shoulder[0], vial[1] - shoulder[1]
    hn = math.hypot(hx, hy)
    hx, hy = hx / hn, hy / hn
    cp, sp = math.cos(math.radians(args_cli.pitch)), math.sin(math.radians(args_cli.pitch))
    gx = vial[0] + args_cli.seat * cp * hx
    gy = vial[1] + args_cli.seat * cp * hy
    gz = vial[2] + args_cli.grasp_dz - args_cli.seat * sp
    grasp_z = max(gz, min_ee_z)
    if grasp_z > gz:
        print(f"[INFO] 손가락 끝이 매트에 닿지 않도록 파지 높이를 {gz:.3f} -> {grasp_z:.3f} m 로 올림")
    hover = [gx, gy, grasp_z + args_cli.hover_dz]
    grasp = [gx, gy, grasp_z]
    over_rack = [rack[0], rack[1], rack[2] + 0.12]
    plan = [
        ("영 자세", None, None),
        ("바이알 위", hover, OPEN),
        ("하강", grasp, OPEN),
        ("파지(닫기)", grasp, CLOSED),
        ("들어올리기", hover, CLOSED),
    ]
    if args_cli.place:
        plan += [("랙 위로", over_rack, CLOSED), ("놓기(열기)", over_rack, OPEN)]
    joint_plan = []
    for name, tgt, grip in plan:
        if tgt is None:
            joint_plan.append((name, [0.0] * 6, tgt))
            continue
        q, warn = solve_pose(tgt, shoulder, args_cli.pitch, grip)
        if q is None:
            print(f"[FAIL] {name}: {warn}")
            q = joint_plan[-1][1]
        elif warn:
            print(f"[WARN] {name}: 관절 한계 초과 -> {warn}")
        else:
            print(
                f"[OK]   {name}: q(deg)=" + ", ".join(f"{math.degrees(v):.0f}" for v in q)
            )
        joint_plan.append((name, q, tgt))

    grip_ids = scene["robot"].find_joints(["gripper_joint_1", "gripper_joint_2"])[0]
    seg_steps = max(1, int(args_cli.seg_sec / uenv.step_dt))
    current = torch.zeros(6, device=dev)
    ee_idx_report = True
    for name, q, tgt in joint_plan:
        goal = torch.tensor(q, device=dev)
        start = current.clone()
        for i in range(seg_steps):
            if not simulation_app.is_running():
                env.close()
                return
            a = start + (goal - start) * (i + 1) / seg_steps
            with torch.inference_mode():
                env.step(a.unsqueeze(0))
            if name.startswith("파지") and (i % 10 == 0 or i == seg_steps - 1):
                # 닫는 동안 무엇이 손을 밀어 올리는지 보기 위한 진단 출력
                robot = scene["robot"]
                gj = robot.data.joint_pos[0, grip_ids].mul(180.0 / math.pi).tolist()
                ee_now = (scene["ee_frame"].data.target_pos_w[0, 0] - origin).tolist()
                vial_now = (scene[args_cli.target].data.root_pos_w[0] - origin).tolist()
                forces = scene["contact_grasp"].data.net_forces_w[0].norm(dim=-1).tolist()
                print(
                    f"[CLOSE {i:3d}/{seg_steps}] ee_z={ee_now[2]:.3f} ee_x={ee_now[0]:.3f} "
                    f"vial=({vial_now[0]:.3f},{vial_now[1]:.3f},{vial_now[2]:.3f}) "
                    f"grip_joint(deg)={[round(v) for v in gj]} finger_force(N)={[round(f, 1) for f in forces]}"
                )
        current = goal
        if tgt is not None and ee_idx_report:
            ee = (scene["ee_frame"].data.target_pos_w[0, 0] - origin).cpu().tolist()
            err = math.dist(ee, tgt) * 1000.0
            print(f"[CHECK] {name}: ee={[round(v,3) for v in ee]} 목표={[round(v,3) for v in tgt]} 오차={err:.0f} mm")

    print("[INFO] 시퀀스 완료. 마지막 자세를 유지합니다 (창을 닫으면 종료).")
    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(current.unsqueeze(0))
    env.close()


if __name__ == "__main__":
    main()
