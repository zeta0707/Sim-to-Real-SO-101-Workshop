# SPDX-License-Identifier: Apache-2.0
#
# OMX-L 리더 값 확인용 (Isaac Sim 없이 실행). 시뮬레이션과 연결하기 전에 리더가 읽히는지,
# 값의 범위가 맞는지, 그리퍼 열림/닫힘 값이 얼마인지 잰다.
#
#   python source/sim_to_real_so101/scripts/omx_leader_probe.py --port /dev/ttyACM0 --id omx_leader_arm
#
# 하는 일: connect(calibrate=False) 로 연결 -> 0.2 s 마다 정규화 값과 각도(deg = 값 x 1.8)를 출력
#          -> 그리퍼 열림/닫힘 값을 평균내어 --gripper_open_norm / --gripper_closed_norm 후보를 출력.
# 주의: 보정(calibrate)은 호출하지 않는다. 보정 값이 모터와 다르면 경고만 하고 종료한다.
#       connect() 가 내부적으로 configure() 를 불러 모터 동작 모드를 쓰고 그리퍼 토크를 켠다(그리퍼가 열림 위치로 간다).
import argparse
import time

from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig
from lerobot.teleoperators.omx_leader.omx_leader import OmxLeader

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="/dev/ttyACM0")
parser.add_argument("--id", type=str, default="omx_leader_arm")
parser.add_argument("--seconds", type=float, default=10.0, help="자유 관찰 시간")
parser.add_argument("--phase_sec", type=float, default=5.0, help="그리퍼 열림/닫힘 측정 시간")
args = parser.parse_args()

leader = OmxLeader(OmxLeaderConfig(port=args.port, id=args.id))
leader.connect(calibrate=False)
try:
    if not leader.is_calibrated:
        print("[중단] 모터의 보정 값과 보정 파일이 다릅니다. 보정은 하지 않고 종료합니다. 출력을 알려주세요.")
        raise SystemExit(1)

    def read():
        a = leader.get_action()
        return {k.replace(".pos", ""): v for k, v in a.items()}

    print(f"[1/3] {args.seconds:.0f}초 동안 값을 출력합니다. 리더를 천천히 움직여 보세요. (정규화 값 / 각도)")
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        a = read()
        print(
            " ".join(f"{k[:8]}={v:7.1f}({v * 1.8:6.1f}deg)" if k != "gripper" else f"gripper={v:6.2f}" for k, v in a.items())
        )
        time.sleep(0.2)

    def measure(label):
        vals = []
        t0 = time.time()
        while time.time() - t0 < args.phase_sec:
            vals.append(read()["gripper"])
            time.sleep(0.05)
        return sum(vals) / len(vals), min(vals), max(vals)

    print(f"[2/3] {args.phase_sec:.0f}초: 그리퍼 트리거를 놓은 상태(열림)로 가만히 두세요.")
    time.sleep(2.0)
    open_mean, open_min, open_max = measure("open")
    print(f"[3/3] {args.phase_sec:.0f}초: 그리퍼 트리거를 끝까지 꽉 쥐고(닫힘) 유지하세요.")
    time.sleep(2.0)
    closed_mean, closed_min, closed_max = measure("closed")

    print()
    print(f"열림: 평균 {open_mean:.2f} (min {open_min:.2f}, max {open_max:.2f})")
    print(f"닫힘: 평균 {closed_mean:.2f} (min {closed_min:.2f}, max {closed_max:.2f})")
    print(f"=> --gripper_open_norm {open_mean:.2f} --gripper_closed_norm {closed_mean:.2f}")
finally:
    leader.disconnect()
