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
                self.rel_local = R.T @ (self.cube0 - ee)
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

    def __init__(self, port, robot_id, units, gripper_closed_pos, gripper_open_pos, send, max_relative_target, torque_off_on_exit):
        from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig

        kw = dict(id=robot_id, port=port, disable_torque_on_disconnect=torque_off_on_exit)
        if units == "degrees":
            kw["use_degrees"] = True
        if max_relative_target:
            kw["max_relative_target"] = max_relative_target
        self.robot = OmxFollower(OmxFollowerConfig(**kw))
        self.robot.connect(calibrate=True)  # 공장 기본 보정(offset 0) 적용
        self.units = units
        self.gc, self.go = gripper_closed_pos, gripper_open_pos
        self.send = send
        if not send:  # 읽기 전용: 손으로 움직여 볼 수 있게 토크를 끈다
            try:
                self.robot.bus.disable_torque()
                print("[io] 읽기 전용 모드: 토크를 껐다 (손으로 움직일 수 있다)")
            except Exception as e:  # noqa: BLE001
                print(f"[io][경고] 토크를 끄지 못했다: {e}")

    def _to_rad_arm(self, v):
        deg = v if self.units == "degrees" else v * 1.8
        return np.deg2rad(deg)

    def _from_rad_arm(self, r):
        deg = np.rad2deg(r)
        return deg if self.units == "degrees" else deg / 1.8

    def read(self):
        obs = self.robot.get_observation()
        v = np.array([obs[f"{m}.pos"] for m in self.MOTORS], dtype=float)
        q = np.empty(6)
        q[:5] = self._to_rad_arm(v[:5])
        q[5] = (v[5] - self.gc) / (self.go - self.gc) * GRIPPER_MAX_RAD
        return q

    def write(self, target):
        if not self.send:
            return
        a = {f"{m}.pos": float(self._from_rad_arm(target[i])) for i, m in enumerate(self.MOTORS[:5])}
        a["gripper.pos"] = float(self.gc + target[5] / GRIPPER_MAX_RAD * (self.go - self.gc))
        self.robot.send_action(a)

    def close(self):
        self.robot.disconnect()


# ---- 제어 루프 ------------------------------------------------------------------------------------------------------
def ee_z(q):
    return fk(q)[2, 3]


def smooth_move(io, q_from, q_to, seconds, send):
    n = max(1, int(seconds * CONTROL_HZ))
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * i / n)
        if send:
            io.write(q_from + s * (q_to - q_from))
        time.sleep(1.0 / CONTROL_HZ)


def run(args, io):
    policy = OmxPolicy(args.policy)
    builder = ObsBuilder()
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
    try:
        for step in range(args.max_steps):
            q = io.read()
            qd = args.vel_alpha * qd + (1 - args.vel_alpha) * (q - prev_q) / dt
            prev_q = q
            obs = builder.build(q, qd, last_action)
            a = policy(obs)  # [-1,1] 로 clip 됨
            last_action = a.copy()  # 시뮬레이션의 last_action 과 같은 의미 (clip 된 정책 출력)
            new_t = np.clip(target + DELTA_SCALE * args.action_scale * a, LO, HI)
            if ee_z(new_t) < EE_Z_MIN and ee_z(new_t) < ee_z(target):
                new_t[:5] = target[:5]  # 바닥 쪽으로 더 내려가는 팔 동작은 거부 (그리퍼는 허용)
                if step % 10 == 0:
                    print(f"[run][안전] step {step}: ee z 하한({EE_Z_MIN} m) 때문에 팔 목표 갱신 거부")
            target = new_t
            io.write(target)
            for k, v in (("q", q), ("qd", qd), ("obs", obs), ("action", a), ("target", target), ("ee", fk(q)[:3, 3])):
                log[k].append(np.array(v))
            log["phase"].append(builder.phase)
            if step % 10 == 0:
                e = fk(q)[:3, 3]
                print(f"[run] step {step:4d} phase={builder.phase:8s} ee=({e[0]:.3f},{e[1]:.3f},{e[2]:.3f}) "
                      f"grip={q[5]:.2f} a={np.round(a, 2)}")
            if builder.release_steps > args.after_release_steps:
                print("[run] 놓기 완료 -> 종료")
                break
            t_next += dt
            slp = t_next - time.perf_counter()
            if slp > 0:
                time.sleep(slp)
            else:
                t_next = time.perf_counter()  # 주기를 못 맞췄다 -> 따라잡지 않고 리셋
    except KeyboardInterrupt:
        print("\n[run] 중단 (Ctrl-C). 마지막 목표를 유지한 채 종료한다.")
    finally:
        if args.log:
            np.savez(args.log, **{k: np.array(v) for k, v in log.items()})
            print(f"[run] 로그 저장: {args.log}")
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
    p.add_argument("--probe", action="store_true", help="관절/그리퍼 값을 계속 출력만 한다 (그리퍼 보정값 측정용, Ctrl-C 로 끝)")
    p.add_argument("--log", default=None, help="실행 기록(.npz) 저장 경로")
    args = p.parse_args()

    if args.selftest:
        ok = selftest(args.selftest, args.policy)
        print("[selftest]", "통과" if ok else "실패")
        sys.exit(0 if ok else 2)

    if args.io == "dummy":
        io = DummyIO()
    else:
        if args.gripper_closed_pos is None or args.gripper_open_pos is None:
            if not args.probe:
                sys.exit("--gripper_closed_pos / --gripper_open_pos 가 필요하다. 먼저 --probe 로 값을 측정할 것.")
            args.gripper_closed_pos, args.gripper_open_pos = 0.0, 100.0
        io = LerobotOmxIO(args.port, args.robot_id, args.units, args.gripper_closed_pos, args.gripper_open_pos,
                          args.send, args.max_relative_target, args.torque_off_on_exit)
    try:
        if args.probe:
            print("[probe] 손으로 움직여 값을 읽는다. Ctrl-C 로 종료. (gripper 는 보정 전이라 0~100 선형 가정)")
            try:
                while True:
                    obs = io.robot.get_observation() if hasattr(io, "robot") else {}
                    print({k: round(float(v), 2) for k, v in obs.items() if k.endswith(".pos")}, end="\r", flush=True)
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print()
            return
        run(args, io)
    finally:
        io.close()


if __name__ == "__main__":
    main()
