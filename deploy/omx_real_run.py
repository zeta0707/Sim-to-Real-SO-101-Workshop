# SPDX-License-Identifier: Apache-2.0
#
# 시뮬레이션에서 학습한 OMX Cube-To-Cup PPO 정책(.npz)을 실제 OMX-F 에서 실행하는 제어 루프 (30 Hz).
# Isaac Sim 없이 numpy 만으로 돈다. 큐브/컵 위치는 "고정 좌표"로 가정한다 (계획서의 1번 방식).
#
#   [로봇 없이 점검]   python deploy/omx_real_run.py --selftest outputs/rl/ref_r10.npz
#   [A: 관측/행동만 보기, 명령 안 보냄]  python deploy/omx_real_run.py --io lerobot --port /dev/ttyUSB0 \
#                       --gripper_open_pos <값> --gripper_closed_pos <값> --probe
#                       python deploy/omx_real_run.py --io lerobot ... --max_steps 300         (--send 가 없으면 명령을 안 보낸다)
#   [B~E: 실제 구동]   ... --send --action_scale 0.3   (처음엔 느리게, 단계마다 올린다)
#
# 안전: 기본은 "읽기만" 이다. 관절 명령은 --send 를 줘야 나간다. Ctrl-C 로 멈추면 토크를 유지한 채 끝난다
#       (팔이 툭 떨어지지 않게). 토크를 끄려면 --torque_off_on_exit.
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "source", "sim_to_real_so101", "scripts"))  # 저장소 배치
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))  # 이 작업 폴더 배치
from omx_kinematics import fk, rotmat_to_quat_wxyz  # noqa: E402
from omx_policy_np import OmxPolicy  # noqa: E402

# ---- 시뮬레이션과 같은 상수 (tasks/omx_cube_to_cup_env_cfg.py, mdp/omx_rl.py) ------------------------------------
CUBE_XY = np.array([0.22, -0.04])          # CUBE_START
CUBE_Z = 0.035 + 0.03 / 2 + 0.002          # 매트 위 큐브 중심 높이
CUP_XY = np.array([0.17, 0.10])            # CUP_POS
# 시뮬레이션 정책이 큐브를 들고 있을 때 (큐브 - ee) 를 ee 좌표계로 본 중앙값 (표준편차 약 4 mm).
# 쥔 순간 ee 가 이미 움직여 있어도 큐브는 손에 있으므로, 고정 좌표(cube0) 기준이 아니라 이 값으로 큐브를 ee 에 붙인다.
HELD_REL_LOCAL = np.array([-0.0038, -0.0149, -0.0017])
CUP_CUBE_Z = 0.05                          # 컵 안에 들어간 큐브의 대략적 높이
DELTA_SCALE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.08])  # rad / step (gripper 포함)
CONTROL_HZ = 30.0
GRIPPER_MAX_RAD = np.deg2rad(100.0)        # sim gripper_joint_1 범위 0 ~ 100 deg (닫힘 0, 열림 100)
# 관절 한계 (USD 한계에서 5도 안쪽). joint1~5, gripper
LIMITS_DEG = np.array([[-100, 100], [-115, 85], [-115, 85], [-95, 95], [-180, 180], [0, 100]], dtype=float)
LO = np.deg2rad(LIMITS_DEG[:, 0])
HI = np.deg2rad(LIMITS_DEG[:, 1])
EE_Z_MIN = 0.05                            # ee 가 이 높이(m) 아래로 내려가는 목표는 거부 (매트 윗면 0.035 + 여유)

GRIP_OPENED = 0.30   # rad. 이 이상이면 "열었다"
GRIP_CLOSED = 0.15   # rad. 열었다가 이 아래로 닫히면 "큐브를 쥐었다"


class ObsBuilder:
    """정책 입력 33 차원을 만든다. 큐브는 고정 좌표 -> 쥔 뒤에는 ee 에 붙어 다닌다고 가정한다."""

    def __init__(self, cube_xy=CUBE_XY, cup_xy=CUP_XY):
        self.cube0 = np.array([cube_xy[0], cube_xy[1], CUBE_Z])
        self.cup_xy = np.asarray(cup_xy)
        self.attached = False
        self.seen_open = False
        self.released = False
        self.rel_local = None
        self.R_rel = None
        self.release_steps = 0

    def cube_pose(self, ee, R):
        if self.released:
            return np.array([self.cup_xy[0], self.cup_xy[1], CUP_CUBE_Z]), np.array([1.0, 0, 0, 0])
        if self.attached:
            return ee + R @ self.rel_local, rotmat_to_quat_wxyz(R @ self.R_rel)
        return self.cube0.copy(), np.array([1.0, 0, 0, 0])

    def build(self, q, qd, last_action):
        M = fk(q)
        ee, R = M[:3, 3], M[:3, :3]
        g = q[5]
        if not self.attached and not self.released:
            if g > GRIP_OPENED:
                self.seen_open = True
            elif self.seen_open and g < GRIP_CLOSED:
                self.attached = True  # 이 순간 큐브가 시작 위치에 있고 ee 가 그것을 쥐었다고 본다
                self.rel_local = HELD_REL_LOCAL.copy()  # 쥔 큐브는 손 안에 있다 (고정 좌표가 아니라 시뮬레이션의 쥔 상태 값 사용)
                self.R_rel = R.T @ np.eye(3)
        elif self.attached and not self.released and g > GRIP_OPENED:
            near = np.linalg.norm(ee[:2] - self.cup_xy) < 0.06
            if near:
                self.released = True
            else:
                raise RuntimeError("컵에서 먼 곳에서 그리퍼가 열렸다 (큐브를 놓침). 중단.")
        if self.released:
            self.release_steps += 1
        cube, cq = self.cube_pose(ee, R)
        return np.concatenate([q, qd, ee, cube, cq, cube - ee, cube[:2] - self.cup_xy, last_action])

    @property
    def phase(self):
        return "released" if self.released else ("attached" if self.attached else ("open" if self.seen_open else "approach"))


# ---- 로봇 입출력 ------------------------------------------------------------------------------------------------
class DummyIO:
    """로봇 없이 루프를 점검하기 위한 가짜 로봇 (목표를 lag 만큼 따라간다)."""

    def __init__(self, gain=1.0):
        self.q = np.zeros(6)
        self.gain = gain

    def read(self):
        return self.q.copy()

    def write(self, target):
        self.q = self.q + self.gain * (target - self.q)

    def close(self):
        pass


class LerobotOmxIO:
    """LeRobot OmxFollower 연결. API/단위는 설치된 lerobot 버전에서 확인할 것 (--probe 로 먼저 값을 본다)."""

    MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def __init__(self, port, robot_id, units, gripper_closed_pos, gripper_open_pos, send, max_relative_target, torque_off_on_exit,
                 joint_offset_deg=None, gripper_closed_cmd_pos=None, grip_gain=1.0, joint_sign=None):
        from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig

        kw = dict(id=robot_id, port=port, disable_torque_on_disconnect=torque_off_on_exit)
        if units == "degrees":
            kw["use_degrees"] = True
        if max_relative_target:
            kw["max_relative_target"] = max_relative_target
        self.robot = OmxFollower(OmxFollowerConfig(**kw))
        self.robot.connect(calibrate=True)  # 공장 기본 보정(offset 0) 적용
        self.units = units
        self.go_hw = gripper_open_pos  # 하드웨어 최대 열림 (명령은 이 위로 보내지 않는다)
        # grip_gain>1: 정책의 같은 rad 에 대해 실제 그리퍼를 gain 배 더 벌린다 (읽기도 같은 비율로 되돌림)
        gripper_open_pos = gripper_closed_pos + (gripper_open_pos - gripper_closed_pos) * grip_gain
        self.gc, self.go = gripper_closed_pos, gripper_open_pos  # 읽기: 큐브를 쥔 값 -> 0 rad, 완전 열림 -> 1.745 rad
        # 명령: 0 rad 를 "완전히 닫힘(빈 손)" 위치로 보낸다. 쥔 값 그대로 명령하면 큐브를 누르지 못한다 (전류 제한 제어라 꽉 쥐어도 안전)
        self.gcmd = gripper_closed_cmd_pos if gripper_closed_cmd_pos is not None else gripper_closed_pos
        self.sign = np.ones(5) if joint_sign is None else np.asarray(joint_sign, dtype=float)  # 관절별 회전 방향 (+1/-1)
        self.off = np.zeros(5) if joint_offset_deg is None else np.asarray(joint_offset_deg, dtype=float)  # deg, 영점 보정
        self.send = send
        if not send:  # 읽기 전용: 손으로 움직여 볼 수 있게 토크를 끈다
            try:
                self.robot.bus.disable_torque()
                print("[io] 읽기 전용 모드: 토크를 껐다 (손으로 움직일 수 있다)")
            except Exception as e:  # noqa: BLE001
                print(f"[io][경고] 토크를 끄지 못했다: {e}")

    def _to_rad_arm(self, v):
        deg = self.sign * ((v if self.units == "degrees" else v * 1.8) - self.off)
        return np.deg2rad(deg)

    def _from_rad_arm(self, r):
        deg = self.sign * np.rad2deg(r) + self.off
        return deg if self.units == "degrees" else deg / 1.8

    def read(self):
        obs = self.robot.get_observation()
        v = np.array([obs[f"{m}.pos"] for m in self.MOTORS], dtype=float)
        q = np.empty(6)
        q[:5] = self._to_rad_arm(v[:5])
        q[5] = np.clip((v[5] - self.gc) / (self.go - self.gc) * GRIPPER_MAX_RAD, -0.6, GRIPPER_MAX_RAD)  # 음수 = 빈 손으로 더 닫힘
        return q

    def write(self, target):
        if not self.send:
            return
        vals = self._from_rad_arm(np.asarray(target[:5], dtype=float))  # 5 관절 한꺼번에 (오프셋이 5-벡터)
        a = {f"{m}.pos": float(vals[i]) for i, m in enumerate(self.MOTORS[:5])}
        # 읽기와 같은 직선 매핑 + 닫힘 근처(0~0.3 rad)에서만 "더 쥐도록" 최대 (gc-gcmd) 만큼 더 닫는다
        squeeze = (self.gc - self.gcmd) * max(0.0, 1.0 - target[5] / 0.3)
        gcmd_v = self.gc + target[5] / GRIPPER_MAX_RAD * (self.go - self.gc) - squeeze
        a["gripper.pos"] = float(min(gcmd_v, self.go_hw))
        self.robot.send_action(a)

    def close(self):
        self.robot.disconnect()


# ---- 제어 루프 ------------------------------------------------------------------------------------------------------
def _avg_pos(robot, seconds=1.0):
    """seconds 동안의 평균 관절 값 (로봇은 읽기 전용 상태)."""
    rows = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        obs = robot.get_observation()
        rows.append([float(obs[f"{m}.pos"]) for m in LerobotOmxIO.MOTORS])
        time.sleep(0.05)
    return np.mean(rows, axis=0)


def calibrate(io, path, units):
    """대화형 보정: 영점 자세, 그리퍼 열림/닫힘/쥠 값을 재서 JSON 으로 저장한다 (토크 꺼진 상태, 손으로 움직인다)."""
    r = io.robot
    print("\n[보정] 토크가 꺼져 있다. 안내에 따라 손으로 자세를 잡고 Enter 를 누른다.\n")
    input("1/4  팔을 '0 자세'로: 위팔 수직, 아래팔 수평(앞쪽), 손목 일직선, 손목 롤 0.  잡고 Enter > ")
    zero = _avg_pos(r)
    print(f"      읽힌 값: {np.round(zero, 2)}")
    input("2/4  그리퍼를 '완전히 연' 상태로 두고 Enter > ")
    g_open = float(_avg_pos(r)[5])
    print(f"      gripper.pos = {g_open:.2f}")
    input("3/4  그리퍼를 '완전히 닫은(빈 손)' 상태로 두고 Enter > ")
    g_closed_empty = float(_avg_pos(r)[5])
    print(f"      gripper.pos = {g_closed_empty:.2f}")
    input("4/4  3 cm 큐브를 '쥔' 상태로 두고 Enter (손으로 살짝 쥐어 큐브가 잡힌 상태) > ")
    g_hold = float(_avg_pos(r)[5])
    print(f"      gripper.pos = {g_hold:.2f}")
    out = {
        "units": units,
        "joint_offset_deg": [round(float(v), 2) for v in zero[:5]],  # 0 자세에서 읽힌 값 = 영점 오프셋
        "gripper_open_pos": round(g_open, 2),
        "gripper_closed_pos": round(g_hold, 2),          # 쥠 값 -> sim 0 rad
        "gripper_closed_cmd_pos": round(g_closed_empty, 2),  # 닫기 명령 위치
    }
    if abs(g_open - g_hold) < 3:
        print("[보정][경고] 열림 값과 쥠 값이 3 이내로 가깝다. 그리퍼를 제대로 열고 닫았는지 확인할 것.")
    if np.max(np.abs(zero[:5])) > 15:
        print("[보정][경고] 0 자세에서 15° 넘게 읽힌 관절이 있다. 팔을 0 자세로 정확히 놓았는지 확인할 것 "
              "(이 값이 오프셋으로 저장된다).")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[보정] 저장: {path}\n{json.dumps(out, indent=2)}\n")
    print("[보정] 다음부터는 --calib 만 주면 이 값을 쓴다.")


def lift_ee_z(t, dz):
    """팔끝 x,y,자세 방향은 되도록 두고 z 만 dz(m) 올리는 관절 변화량 (관절 1~4, 수치 야코비안 의사역행렬)."""
    base = fk(t)[:3, 3]
    J = np.zeros((3, 4))
    for j in range(4):
        tj = t.copy()
        tj[j] += 1e-4
        J[:, j] = (fk(tj)[:3, 3] - base) / 1e-4
    want = np.array([0.0, 0.0, dz])
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), want)
    return np.clip(dq, -0.05, 0.05)


def ee_z(q):
    return fk(q)[2, 3]


def smooth_move(io, q_from, q_to, seconds, send):
    n = max(1, int(seconds * CONTROL_HZ))
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * i / n)
        if send:
            io.write(q_from + s * (q_to - q_from))
        time.sleep(1.0 / CONTROL_HZ)


def trim(io, args):
    """토크를 켠 채 0 자세를 유지하고, 눈으로 보며 관절별 영점 오프셋을 미세 조정한 뒤 calib.json 에 저장한다."""
    if not args.calib:
        sys.exit("--trim 은 --calib deploy/calib.json 과 --send 가 필요하다")
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    q = io.read()
    if args.trim_q is not None:
        tq = np.asarray(args.trim_q, dtype=float)
        q_hold = np.concatenate([np.deg2rad(tq[:5]), [tq[5] if len(tq) > 5 else 0.5]])  # 팔 5개는 deg, 그리퍼는 rad
        e = fk(q_hold)[:3, 3]
        print(f"[trim] 작업 자세(시뮬레이션 관절각 {np.round(tq, 1)})로 이동. 계산상 팔끝 = ({e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}) m")
        print("       큐브를 그 자리(팔끝 계산 위치)에 놓아 두고, 그리퍼가 큐브를 좌우로 벌린 채 가운데에 오도록 맞춘다.")
    else:
        q_hold = np.zeros(6)
        print("[trim] 0 자세로 이동 (토크 켬)")
        print("       기준 0 자세: 아래팔 수평, 손목 일직선, 손가락 좌우로 벌어짐. 팔끝 목표 (x 0.313, y 0, z 0.212) m")
    smooth_move(io, q, q_hold, args.home_secs, True)

    def hold(secs=1.0):
        for _ in range(int(secs * CONTROL_HZ)):
            io.write(q_hold)
            time.sleep(1.0 / CONTROL_HZ)
        qr = io.read()
        e = fk(qr)[:3, 3]
        print(f"       읽힌 관절(deg) {np.round(np.rad2deg(qr[:5]), 1)} / 명령 {np.round(np.rad2deg(q_hold[:5]), 1)}  "
              f"팔끝(읽힌 값 기준) ({e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f})")

    hold(1.5)

    print("[trim] 명령:  '<관절번호 1~5> <각도>' 예) '2 5' (shoulder_lift 를 +5° 이동),  p=현재 오프셋,  s=저장 후 종료,  q=저장 없이 종료")
    print("       번호: 1 shoulder_pan, 2 shoulder_lift, 3 elbow_flex, 4 wrist_flex, 5 wrist_roll. 각도는 실제 모터 방향(+/-)이다.")
    while True:
        try:
            line = input("trim> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[trim] 저장 없이 종료")
            return
        if line in ("q", "quit"):
            print("[trim] 저장 없이 종료")
            return
        if line in ("p", ""):
            print("       오프셋(deg):", {n: round(float(v), 2) for n, v in zip(names, io.off)})
            continue
        if line in ("s", "save"):
            with open(args.calib) as f:
                c = json.load(f)
            c["joint_offset_deg"] = [round(float(v), 2) for v in io.off]
            with open(args.calib, "w") as f:
                json.dump(c, f, indent=2)
            print(f"[trim] 저장: {args.calib}  오프셋 {c['joint_offset_deg']}")
            return
        try:
            j, d = line.split()
            j, d = int(j), float(d)
            assert 1 <= j <= 5 and abs(d) <= 15
        except Exception:  # noqa: BLE001
            print("       형식: '관절번호(1~5) 각도(|각도|<=15)'  예) 3 -4")
            continue
        io.off[j - 1] += d
        hold(1.5)
        print(f"       {names[j - 1]} 오프셋 -> {io.off[j - 1]:.2f}")


def retreat(io, args, target):
    """놓은 뒤 컵에서 안전하게 빠진다: (1) 그리퍼를 넓게 연다 -> (2) 팔끝 x,y 를 유지한 채 수직으로 올린다. 마지막 목표를 돌려준다."""
    tgt = target.copy()
    if args.retreat_open is not None:
        tgt[5] = args.retreat_open  # 지정했을 때만 그리퍼를 더 연다 (기본: 지금 열림 그대로)
    print(f"[run] 후퇴: 그리퍼 {tgt[5]:.2f} rad 유지, z {args.retreat_z:.2f} m 까지 수직 상승")
    for _ in range(int(0.8 * CONTROL_HZ)):
        io.write(tgt)
        time.sleep(1.0 / CONTROL_HZ)
    for _ in range(int(6.0 * CONTROL_HZ)):
        z = ee_z(tgt)
        if z >= args.retreat_z:
            break
        tgt[:4] += lift_ee_z(tgt, min(0.004, args.retreat_z - z))
        tgt = np.clip(tgt, LO, HI)
        io.write(tgt)
        time.sleep(1.0 / CONTROL_HZ)
    print(f"[run] 후퇴 완료: 팔끝 z={ee_z(tgt):.3f}")
    return tgt


def set_park(io, args):
    """토크를 끈 채 손으로 팔을 주차 자세에 놓고 Enter -> 1초 평균을 calib.json 에 저장한다."""
    if not args.calib:
        sys.exit("--set_park 은 --calib deploy/calib.json 이 필요하다 (저장 위치)")
    print("[park] 토크가 꺼져 있다. 팔을 주차 자세로 손으로 놓고 Enter 를 누른다 (그리퍼도 원하는 상태로).")
    input("       준비되면 Enter > ")
    qs = []
    t_end = time.time() + 1.0
    while time.time() < t_end:
        qs.append(io.read())
        time.sleep(0.03)
    qm = np.mean(qs, axis=0)
    e = fk(qm)[:3, 3]
    print(f"[park] 읽힌 값(시뮬레이션 기준 deg) {np.round(np.rad2deg(qm[:5]), 1)}  그리퍼 {qm[5]:.2f} rad  팔끝 ({e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f})")
    with open(args.calib) as f:
        c = json.load(f)
    c["park_q"] = [round(float(v), 4) for v in qm]  # rad, 시뮬레이션 관절 순서 (관절1~5, 그리퍼)
    with open(args.calib, "w") as f:
        json.dump(c, f, indent=2)
    print(f"[park] 저장: {args.calib}  (park_q). 실행 때 --park 을 붙이면 끝난 뒤 이 자세로 간다.")


def run(args, io):
    policy = OmxPolicy(args.policy)
    builder = ObsBuilder(args.cube_xy or CUBE_XY, args.cup_xy or CUP_XY)
    print(f"[run] 큐브 xy={np.round(builder.cube0[:2], 3)}  컵 xy={np.round(builder.cup_xy, 3)}")
    dt = 1.0 / CONTROL_HZ

    q = io.read()
    print(f"[run] 시작 관절(rad): {np.round(q, 3)}  gripper={q[5]:.3f} rad")
    if args.goto_home:
        print(f"[run] 시뮬레이션 시작 자세(관절 0, 그리퍼 닫힘)로 {args.home_secs}s 동안 이동 (--send 일 때만 실제로 움직임)")
        smooth_move(io, q, np.zeros(6), args.home_secs, args.send)
        q = io.read()
    target = q.copy()
    prev_q = q.copy()
    qd = np.zeros(6)
    last_action = np.zeros(6)
    log = {k: [] for k in ("q", "qd", "obs", "action", "target", "ee", "phase")}
    t_next = time.perf_counter()
    attach_step, n_refuse, n_empty, n_in_cup, aborted = None, 0, 0, 0, None
    interrupted = False
    ee_hist, releasing = [], False
    try:
        for step in range(args.max_steps):
            q = io.read()
            qd = args.vel_alpha * qd + (1 - args.vel_alpha) * (q - prev_q) / dt
            prev_q = q
            obs = builder.build(q, qd, last_action)
            if builder.attached and attach_step is None:
                attach_step = step
            a = policy(obs)  # [-1,1] 로 clip 됨
            if args.auto_release and builder.attached and not builder.released:
                # 규칙형 놓기: (1) 팔끝이 컵 중심 release_tol 안에 release_dwell 스텝 머물거나
                # (2) 정책이 컵 근처(stall_radius 안)에서 멈춰 서면(stall_steps 동안 팔끝 이동 < 1.5 mm) 그리퍼를 연다
                rule_cup = np.asarray(args.rule_cup_xy if args.rule_cup_xy else builder.cup_xy)
                ee_obs, cube_z = obs[12:15], obs[17]
                d_cup = np.linalg.norm(ee_obs[:2] - rule_cup)
                carried = cube_z > 0.075 and (attach_step is not None and step >= attach_step + args.grasp_dwell)
                inside = d_cup < args.release_tol and cube_z > 0.075
                n_in_cup = n_in_cup + 1 if inside else 0
                ee_hist.append(ee_obs[:2].copy())
                ee_hist[:] = ee_hist[-args.stall_steps:]
                stalled = (len(ee_hist) >= args.stall_steps and carried and d_cup < args.stall_radius
                           and np.max(np.linalg.norm(np.asarray(ee_hist) - ee_hist[-1], axis=1)) < 0.0015 * 4)
                if n_in_cup >= args.release_dwell or stalled:
                    if not releasing:
                        why = "컵 반경 안" if n_in_cup >= args.release_dwell else "컵 근처에서 정지"
                        print(f"[run] 놓기 시작 ({why}): 팔끝 xy=({ee_obs[0]:.3f},{ee_obs[1]:.3f}) 컵 중심까지 {d_cup*100:.1f} cm")
                    releasing = True
            if releasing and not builder.released:
                a[5] = 1.0  # 그리퍼 열기 (정책 출력 무시)
            if args.lock_roll is not None:
                a[4] = 0.0  # 손목 롤 고정: 정책의 롤 동작은 무시하고, 정책이 보는 last_action 에도 0 으로 넣는다
            last_action = a.copy()  # 시뮬레이션의 last_action 과 같은 의미 (clip 된 정책 출력)
            new_t = np.clip(target + DELTA_SCALE * args.action_scale * a, LO, HI)
            if args.lock_roll is not None:
                new_t[4] = np.deg2rad(args.lock_roll)
            if args.carry_z_min > 0 and builder.attached and not builder.released and attach_step is not None \
                    and step >= attach_step + args.grasp_dwell:
                z_now = ee_z(new_t)
                if z_now < args.carry_z_min:  # 컵 림을 넘도록 운반 높이를 보장한다 (정책이 낮게 움직여도 팔끝을 올린다)
                    new_t[:4] += lift_ee_z(new_t, min(0.006, args.carry_z_min - z_now))
                    new_t = np.clip(new_t, LO, HI)
            if ee_z(new_t) < EE_Z_MIN and ee_z(new_t) < ee_z(target):
                new_t[:5] = target[:5]  # 바닥 쪽으로 더 내려가는 팔 동작은 거부 (그리퍼는 허용)
                n_refuse += 1
                if step % 10 == 0:
                    print(f"[run][안전] step {step}: ee z 하한({EE_Z_MIN} m) 때문에 팔 목표 갱신 거부")
            else:
                n_refuse = 0
            if releasing and not builder.released:
                new_t[:5] = target[:5]  # 놓는 동안 팔은 정지
            if attach_step is not None and step < attach_step + args.grasp_dwell:
                new_t[:5] = target[:5]  # 쥔 직후 grasp_dwell 스텝 동안은 팔을 멈춰 두고 그리퍼만 더 조인다 (들다가 떨어짐 방지)
            target = new_t
            io.write(target)
            for k, v in (("q", q), ("qd", qd), ("obs", obs), ("action", a), ("target", target), ("ee", fk(q)[:3, 3])):
                log[k].append(np.array(v))
            log["phase"].append(builder.phase)
            if step % 10 == 0:
                e = fk(q)[:3, 3]
                print(f"[run] step {step:4d} phase={builder.phase:8s} ee=({e[0]:.3f},{e[1]:.3f},{e[2]:.3f}) "
                      f"grip={q[5]:.2f} a={np.round(a, 2)}")
            if builder.attached and attach_step is None:
                attach_step = step
            if getattr(args, "send", True) and attach_step is not None and not builder.released and step >= attach_step + 8:
                n_empty = n_empty + 1 if q[5] < -0.10 else 0
                if n_empty >= 6:
                    aborted = f"빈 손: 닫은 뒤 그리퍼가 계속 {q[5]:.2f} rad (큐브를 쥐면 약 0). 큐브를 못 쥐었다"
            if getattr(args, "send", True) and n_refuse >= 45:
                aborted = "팔 목표 갱신이 1.5초 넘게 계속 거부됨 (바닥/큐브에 걸린 것으로 보임)"
            if aborted:
                print(f"[run][중단] {aborted}. 토크를 유지한 채 멈춘다. 손으로 받치고 Ctrl-C 없이 종료된다.")
                break
            if builder.release_steps > (min(args.after_release_steps, 2) if args.park else args.after_release_steps):
                print("[run] 놓기 완료 -> 종료")
                break
            t_next += dt
            slp = t_next - time.perf_counter()
            if slp > 0:
                time.sleep(slp)
            else:
                t_next = time.perf_counter()  # 주기를 못 맞췄다 -> 따라잡지 않고 리셋
    except KeyboardInterrupt:
        interrupted = True
        print("\n[run] 중단 (Ctrl-C). 마지막 목표를 유지한 채 종료한다.")
    finally:
        if args.log:
            np.savez(args.log, **{k: np.array(v) for k, v in log.items()})
            print(f"[run] 로그 저장: {args.log}")
    if args.park and args.send and not interrupted and not aborted:
        if args.park_q is None:
            print("[run][경고] 주차 자세가 없다. 먼저 --set_park 으로 저장할 것 (--calib 필요)")
        else:
            if not builder.released:
                print("[run][경고] 놓지 못했다 (큐브를 쥔 채). 주차하지 않고 현재 자세를 유지한다. 손으로 받치고 확인할 것")
                return builder.phase
            retreat(io, args, target)
            qn = io.read()
            print(f"[run] 주차 자세로 이동 ({args.park_secs}s)")
            smooth_move(io, qn, np.asarray(args.park_q, dtype=float), args.park_secs, True)
            print("[run] 주차 완료 (토크 유지)")
    return builder.phase


def selftest(ref_path, policy_path):
    ref = np.load(ref_path)
    obs, act = ref["obs"], ref["act"]
    # 1) FK 가 시뮬레이션 ee 와 일치하는가 (에피소드 첫 스텝은 시뮬레이션이 ee 를 갱신하지 않아 제외)
    err = np.array([np.linalg.norm(fk(o[:6])[:3, 3] - o[12:15]) for o in obs])
    good = err < 0.01
    print(f"[selftest] FK: {good.sum()}/{len(err)} 샘플 일치 (최대 {err[good].max():.1e} m). 제외된 첫 스텝: {(~good).sum()} 개")
    # 2) 정책이 같은 관측에서 같은 행동을 내는가
    pol = OmxPolicy(policy_path)
    perr = np.abs(np.stack([pol.raw(o) for o in obs]) - act).max()
    print(f"[selftest] 정책 출력 오차 최대 {perr:.1e}")
    # 3) 가짜 로봇으로 루프 한 번 돌려 보기 (큐브 고정좌표 + 쥔 뒤 ee 에 부착 가정)
    class A:  # noqa: D106
        policy = policy_path
        goto_home = False
        cube_xy = None
        rule_cup_xy = None
        retreat_z = 0.24
        retreat_open = None
        park = False
        park_q = None
        park_secs = 6.0
        auto_release = False
        release_tol = 0.03
        release_dwell = 10
        stall_steps = 30
        stall_radius = 0.045
        carry_z_min = 0.115
        lock_roll = None
        grasp_dwell = 20
        trim_q = None
        joint_sign = None
        cup_xy = None
        home_secs = 0
        send = True
        action_scale = 1.0
        max_steps = 400
        vel_alpha = 0.5
        after_release_steps = 15
        log = None

    io = DummyIO()
    phase = run(A, io)
    print(f"[selftest] 가짜 로봇 루프 종료 phase={phase}, 최종 관절 {np.round(io.q, 2)}")
    return bool(good.sum() > 0.95 * len(err) and perr < 1e-3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=os.path.join(HERE, "omx_policy_r10.npz"))
    p.add_argument("--selftest", default=None, help="ref_*.npz 경로. 로봇 없이 FK/정책/루프를 점검한다")
    p.add_argument("--io", choices=["dummy", "lerobot"], default="dummy")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--robot_id", default="omx_follower")
    p.add_argument("--units", choices=["degrees", "norm100"], default="degrees", help="팔로워 관절 값의 단위 (lerobot use_degrees)")
    p.add_argument("--gripper_closed_pos", type=float, default=None, help="큐브를 쥐었을 때(또는 완전히 닫았을 때) gripper.pos 값 -> sim 0 rad")
    p.add_argument("--gripper_open_pos", type=float, default=None, help="완전히 열었을 때 gripper.pos 값 -> sim 1.745 rad")
    p.add_argument("--send", action="store_true", help="이 옵션이 있어야 관절 명령을 실제로 보낸다 (없으면 읽기 전용)")
    p.add_argument("--action_scale", type=float, default=0.3, help="행동 스케일 (1.0 = 시뮬레이션과 같은 속도). 처음엔 0.3")
    p.add_argument("--max_relative_target", type=float, default=8.0, help="lerobot 의 스텝당 최대 이동량 (단위는 --units 와 같음)")
    p.add_argument("--goto_home", action="store_true", help="시작 시 관절 0 자세로 천천히 이동 (시뮬레이션 시작 자세)")
    p.add_argument("--home_secs", type=float, default=4.0)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--vel_alpha", type=float, default=0.5, help="관절 속도 추정의 저역통과 계수 (유한 차분)")
    p.add_argument("--after_release_steps", type=int, default=30)
    p.add_argument("--torque_off_on_exit", action="store_true")
    p.add_argument("--grip_gain", type=float, default=1.0,
                   help="그리퍼 열림 배율. 1.5 면 정책의 같은 열림 각도에서 실제로 1.5배 더 벌어진다 (하드웨어 최대는 넘지 않음)")
    p.add_argument("--cube_xy", type=float, nargs=2, default=None, help="실제 큐브 중심 xy (m, 베이스 기준). 기본 0.22 -0.04")
    p.add_argument("--cup_xy", type=float, nargs=2, default=None, help="실제 컵 중심 xy (m). 기본 0.17 0.10")
    p.add_argument("--joint_sign", type=float, nargs=5, default=None,
                   help="관절 1~5 의 회전 방향 (+1 또는 -1). 예: 손목 롤 방향이 시뮬레이션과 반대면 1 1 1 1 -1")
    p.add_argument("--trim", action="store_true", help="토크를 켠 0 자세에서 영점 오프셋을 눈으로 보며 미세 조정 (--send --calib 필요)")
    p.add_argument("--trim_q", type=float, nargs="+", default=None,
                   help="--trim 에서 0 자세 대신 유지할 시뮬레이션 관절각: 관절1~5 (deg) [+ 그리퍼 rad]")
    p.add_argument("--grasp_dwell", type=int, default=20, help="쥔 것으로 판단한 뒤 팔을 멈춰 두는 스텝 수 (30 스텝 = 1초)")
    p.add_argument("--lock_roll", type=float, default=None,
                   help="손목 롤을 이 각도(deg, 시뮬레이션 기준)로 고정한다. 예) 0. 정책의 롤 동작은 무시한다")
    p.add_argument("--carry_z_min", type=float, default=0.115,
                   help="큐브를 쥔 뒤 컵에 놓기 전까지 팔끝(ee) 최소 높이 (m, 로봇 바닥 기준). 컵 림 높이 + 약 2 cm. 0 이면 끔")
    p.add_argument("--auto_release", action="store_true",
                   help="큐브(추정)가 컵 xy 반경 안에 --release_dwell 스텝 머물면 정책 대신 그리퍼를 연다 (정책이 컵 위에서 망설일 때)")
    p.add_argument("--release_tol", type=float, default=0.03, help="auto_release 반경 (m, 팔끝 xy 와 규칙용 컵 중심의 거리)")
    p.add_argument("--rule_cup_xy", type=float, nargs=2, default=None,
                   help="auto_release 가 쓰는 실제 컵 중심 (기본: --cup_xy). 정책에는 --cup_xy 로 다른 값을 줄 수 있다")
    p.add_argument("--retreat_z", type=float, default=0.24, help="놓은 뒤 후퇴 높이 (팔끝 z, m). 운반 높이보다 충분히 높게")
    p.add_argument("--retreat_open", type=float, default=None, help="후퇴 때 그리퍼 열림 (rad). 지정하지 않으면 더 열지 않는다")
    p.add_argument("--stall_steps", type=int, default=30, help="정지 판정 스텝 수 (auto_release)")
    p.add_argument("--stall_radius", type=float, default=0.045, help="이 반경(컵 중심) 안에서 정지하면 놓는다 (m)")
    p.add_argument("--release_dwell", type=int, default=10, help="auto_release 유지 스텝")
    p.add_argument("--park", action="store_true", help="정상 종료(놓기 완료/최대 스텝) 뒤 저장된 주차 자세로 이동 (--send, --calib 의 park_q 필요)")
    p.add_argument("--park_secs", type=float, default=6.0)
    p.set_defaults(park_q=None)
    p.add_argument("--set_park", action="store_true", help="손으로 놓은 현재 자세를 주차 자세로 calib.json 에 저장 (읽기 전용)")
    p.add_argument("--calibrate", action="store_true", help="대화형 보정 (영점 자세 + 그리퍼 열림/닫힘/쥠) 후 --calib_path 에 저장")
    p.add_argument("--calib", default=None, help="보정 JSON 경로 (--calibrate 로 만든 파일). 있으면 그리퍼/영점 인자를 대신한다")
    p.add_argument("--calib_path", default=os.path.join(HERE, "calib.json"))
    p.add_argument("--probe", action="store_true", help="관절/그리퍼 값을 계속 출력만 한다 (그리퍼 보정값 측정용, Ctrl-C 로 끝)")
    p.add_argument("--log", default=None, help="실행 기록(.npz) 저장 경로")
    args = p.parse_args()

    if args.selftest:
        ok = selftest(args.selftest, args.policy)
        print("[selftest]", "통과" if ok else "실패")
        sys.exit(0 if ok else 2)

    joint_off, gcmd = None, None
    if args.calib:
        with open(args.calib) as f:
            c = json.load(f)
        args.units = c.get("units", args.units)
        args.gripper_open_pos = c["gripper_open_pos"]
        args.gripper_closed_pos = c["gripper_closed_pos"]
        gcmd = c.get("gripper_closed_cmd_pos")
        joint_off = c.get("joint_offset_deg")
        args.park_q = c.get("park_q")
        print(f"[run] 보정 로드: {args.calib}")
    if args.io == "dummy":
        io = DummyIO()
    else:
        if args.gripper_closed_pos is None or args.gripper_open_pos is None:
            if not (args.probe or args.calibrate):
                sys.exit("--calib 또는 --gripper_closed_pos / --gripper_open_pos 가 필요하다. 먼저 --calibrate 를 실행할 것.")
            args.gripper_closed_pos, args.gripper_open_pos = 0.0, 100.0
        io = LerobotOmxIO(args.port, args.robot_id, args.units, args.gripper_closed_pos, args.gripper_open_pos,
                          args.send and not args.calibrate, args.max_relative_target, args.torque_off_on_exit,
                          joint_offset_deg=joint_off, gripper_closed_cmd_pos=gcmd, grip_gain=args.grip_gain, joint_sign=args.joint_sign)
    try:
        if args.calibrate:
            calibrate(io, args.calib_path, args.units)
            return
        if args.trim:
            if not args.send:
                sys.exit("--trim 은 --send 가 필요하다")
            trim(io, args)
            return
        if args.set_park:
            set_park(io, args)
            return
        if args.probe:
            print("[probe] 손으로 움직여 값을 읽는다 (읽기 전용). Ctrl-C 로 종료하면 관절별 최소/최대를 보여 준다.")
            lo, hi = {}, {}
            last = {}
            n = 0
            try:
                while True:
                    obs = io.robot.get_observation() if hasattr(io, "robot") else {}
                    last = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
                    for k, v in last.items():
                        lo[k] = min(lo.get(k, v), v)
                        hi[k] = max(hi.get(k, v), v)
                    n += 1
                    if n % 5 == 0:  # 0.5 초마다 한 줄
                        short = {k.replace(".pos", ""): round(v, 1) for k, v in last.items()}
                        print(f"[probe] {short}", flush=True)
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n[probe] 관절별 관측 범위 (min ~ max)  /  마지막 값:")
                for k in lo:
                    print(f"   {k:20s} {lo[k]:8.2f} ~ {hi[k]:8.2f}   last {last.get(k, float('nan')):8.2f}")
            return
        run(args, io)
    finally:
        io.close()


if __name__ == "__main__":
    main()
