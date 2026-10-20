# OMX-F 실제 로봇 배포 (RL 정책 r10)

시뮬레이션에서 학습한 정책을 실제 OMX-F 에서 돌리는 제어 루프다. Isaac Sim 없이 numpy 만 쓴다.
큐브/컵 위치는 **고정 좌표**로 가정한다 (큐브 `(0.22, -0.04)`, 컵 `(0.17, 0.10)`, 로봇 베이스 기준 m).

| 파일 | 역할 |
|---|---|
| `omx_policy_r10.npz` | 내보낸 정책 (actor 가중치 + 관측 정규화 통계) |
| `omx_kinematics.py` | OMX-F 순기구학 (시뮬레이션 ee 와 5e-7 m 일치) |
| `omx_real_run.py` | 30 Hz 제어 루프, 관측 33차원 생성, 안전장치 |

## 0. 로봇 없이 점검
```bash
python deploy/omx_real_run.py --selftest outputs/rl/ref_r10.npz
```
FK / 정책 출력 / 가짜 로봇 루프를 확인한다 (`통과` 가 나와야 한다).

## 1. 준비물과 배치
- 로봇 베이스 원점이 시뮬레이션의 원점이다. 큐브는 베이스에서 앞 0.22 m, 옆(+y 가 왼쪽) -0.04 m (오른쪽 4 cm), 컵은 앞 0.17 m, 왼쪽 0.10 m 에 둔다.
  큐브는 시작 위치에서 ±3 cm, 회전 ±0.5 rad 안에 놓는다 (시뮬레이션이 이 범위로 학습했다).
- 관절 0 자세 = 공장 보정(offset 0) 자세다. 위팔은 수직, 아래팔은 수평으로 앞을 향한다. 시작 시 `--goto_home` 이 이 자세로 천천히 옮긴다.
- LeRobot 에 `lerobot.robots.omx_follower` 가 있어야 한다 (`pip show lerobot`).

## 2. 그리퍼 보정 (필수)
`gripper.pos` 의 의미(0 이 열림인지 닫힘인지)와 범위는 개체/버전마다 다를 수 있다. 읽기 전용으로 값을 본다:
```bash
python deploy/omx_real_run.py --io lerobot --port /dev/ttyUSB0 --probe
```
손으로 그리퍼를 (a) 완전히 열었을 때, (b) **큐브를 쥐었을 때** 값을 적는다.
- `--gripper_open_pos` = (a) 값 -> 시뮬레이션 100° (1.745 rad)
- `--gripper_closed_pos` = (b) 값 -> 시뮬레이션 0 rad. 시뮬레이션에서는 큐브를 쥔 채 그리퍼 각도가 0 으로 읽힌다. 정책이 "쥐었다"를 0 으로 기대하므로 큐브를 쥔 값을 0 으로 맞추는 것이 중요하다.

## 3. 단계별 시험 (앞 단계 통과 후 다음으로)
| 단계 | 명령 (공통: `--io lerobot --port ... --gripper_open_pos A --gripper_closed_pos B`) | 확인 |
|---|---|---|
| A | `--max_steps 300 --log run_a.npz`  (`--send` 없음: 토크를 끄고 읽기만. 손으로 팔을 움직여 본다) | 관측/행동이 튀지 않음, ee 좌표가 눈으로 맞음 |
| B | `--send --goto_home --action_scale 0.3` (큐브 없음) | 관절 한계/책상 충돌 없음, 방향이 시뮬레이션과 같음 |
| C | 큐브를 놓고 같은 명령 | 큐브를 쥠 |
| D | `--action_scale 1.0` | 컵 위로 옮김 |
| E | 반복 10회, `--log` 기록 | 성공률 |

- 처음에는 손이 비상정지(전원)에 닿는 곳에서 한다. Ctrl-C 는 토크를 유지한 채 끝난다 (`--torque_off_on_exit` 로 끌 수 있다).
- 기본값은 **읽기 전용**이다. `--send` 를 줘야 명령이 나간다.
- 안전장치: 관절 한계(USD 한계 -5°), ee 높이 0.05 m 미만으로의 팔 동작 거부, 스텝당 최대 이동량(`--max_relative_target`).

## 4. 알아 둘 한계 (시뮬레이션과 다른 점)
- 큐브 위치는 상수다. 쥔 뒤에는 "쥔 순간의 상대 위치로 ee 에 붙어 있다"고 가정한다 (`ObsBuilder`). 큐브가 미끄러져도 정책은 모른다.
- 관절 속도는 유한 차분(저역통과)으로 추정한다. 시뮬레이션의 속도와 분포가 다르다.
- 모터 PD/지연/마찰이 시뮬레이션과 다르다. `--log` 기록을 시뮬레이션 재생과 비교해서 어긋남을 본다.
- 이 기준은 시뮬레이션에서 큐브가 시작 위치에 정확히 있을 때 100% 다. 실제 오차가 ±3 cm 를 넘으면 실패할 수 있다.

## 실제 로봇 검증된 실행 (Cube-To-Cup, OMX-F)

고정 좌표: 큐브 (0.232, -0.050), 컵 중심 (0.156, 0.098), 컵 높이 7 cm / 내측 반지름 4 cm. 큐브와 컵은 같은 받침대(약 1.7 cm) 위에 둔다.
손목 롤은 고정(`--lock_roll 0`)한다. 정책이 롤을 계속 돌리기 때문.

```
python deploy/omx_real_run.py --io lerobot --port /dev/ttyACM0 --calib deploy/calib.json \
  --send --goto_home --home_secs 4 --action_scale 0.3 --grip_gain 1.5 \
  --cube_xy 0.232 -0.050 --cup_xy 0.179 0.110 --rule_cup_xy 0.156 0.100 \
  --lock_roll 0 --carry_z_min 0.165 --auto_release --release_tol 0.02 \
  --park --max_steps 400 --log run.npz
```

결과: action_scale 0.15 / 0.25 / 0.3 에서 각각 5/5 (큐브가 컵 안, 컵 충돌 없음).

### 옵션 요약
- `--calibrate`: 손으로 관절 오프셋·그리퍼 열림/닫힘 값을 `calib.json` 에 저장. `--set_park`: 주차 자세 저장.
- `--grasp_dwell N`: 쥔 직후 N 스텝 동안 팔을 멈추고 그리퍼만 조인다 (들다 떨어짐 방지).
- `--carry_z_min`: 운반 중 팔끝 최소 높이 (컵 림 위로 넘기기, 0.165 권장).
- `--auto_release`: 컵 중심 `--release_tol` 안에 `--release_dwell` 스텝 머물거나, 컵 `--stall_radius` 안에서 `--stall_steps` 동안 정지하면 그리퍼를 열고 팔은 멈춘다.
- `--rule_cup_xy`: 놓기 규칙이 쓰는 실제 컵 중심. 정책이 보는 `--cup_xy` 와 다르게 줄 수 있다.
- `--park`: 놓은 뒤 그리퍼를 더 열지 않고 `--retreat_z`(0.24)까지 수직 후퇴 → 저장된 주차 자세. 놓지 못했으면 주차하지 않고 현재 자세를 유지한다.
- 안전: ee z 하한 0.05, 빈 손/목표 갱신 거부 중단(`--send` 일 때), Ctrl-C 는 토크 유지.
