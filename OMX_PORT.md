# OMX-F 포팅 (Sim-to-Real-SO-101-Workshop)

SO-101 워크숍의 Isaac Lab 환경을 ROBOTIS OMX-F(+OMX-L 리더)로 옮긴 **추가 파일** 모음이다.
기존 워크숍 파일은 `source/sim_to_real_so101/tasks/__init__.py` 끝에 `from . import omx_register` 한 줄을 추가한 것 외에는 수정하지 않았다. 로봇 에셋은 Pseudo-Lab/NVIDIA-RFM-Lab 의 `omx_f` 를 가져와 썼다.

## 추가된 파일 (`source/sim_to_real_so101/` 기준)

| 파일 | 역할 | SO-101 대응 파일 |
|---|---|---|
| `assets/omx.py` | `OMX_F_CFG`, `OMX_F_CONTACT_GRASP_CFG` (ArticulationCfg) | `assets/so101.py` |
| `assets/usd/omx_f/omx_f.usd` + `configuration/*.usd` | NVIDIA-RFM-Lab 에서 복사한 원본 에셋 (수정 없음) | `assets/usd/SO-ARM101-USD.usd` |
| `assets/usd/omx_f/omx_f_isaaclab.usda` | 원본 위에 얹는 오버라이드 레이어 (아래 설명) | - |
| `mdp/omx_actions.py` | 6차원 액션 + `gripper_joint_2` 를 -1 로 미러하는 액션 term | `JointPositionActionCfg` |
| `tasks/omx_env_cfg.py` | 베이스 teleop 환경 | `tasks/so101_env_cfg.py` |
| `tasks/omx_task_env_cfg.py` | 라이트박스·매트·카메라·DR | `tasks/task_env_cfg.py` |
| `tasks/omx_vials_to_rack_env_cfg.py` | Vials-To-Rack (+DR, +Eval) | `tasks/vials_to_rack_env_cfg.py` |
| `tasks/omx_cube_to_cup_env_cfg.py` | Cube-To-Cup: 큐브 하나를 컵에 넣는 단순 태스크 (+Eval) | `tasks/vials_to_rack_env_cfg.py` |
| `mdp/omx_terms.py` | 큐브가 컵 안에 들어갔는지 판정 | `mdp/terms.py` |
| `assets/usd/cup.usda` | 윗면이 열린 정사각 컵(내경 8 cm, 높이 6 cm, 정적 콜라이더) | - |
| `tasks/omx_register.py` | gym 등록 (`tasks/__init__.py` 끝의 import 한 줄로 로드됨) | `tasks/__init__.py` |
| `utils/omx_interface.py` | OMX 정규화 값 <-> sim 관절각 변환 (`OmxInterface`) | `utils/lerobot_interface.py` 의 SO101 변환 |
| `scripts/omx_lerobot_agent.py` | 리더 텔레옵 + 녹화 | `scripts/lerobot_agent.py` |
| `scripts/omx_lerobot_eval.py` | 원격 GR00T 서버로 평가 | `scripts/lerobot_eval.py` |
| `scripts/omx_reach_agent.py` | 리더 없이 평면 IK 로 바이알 파지/들어올리기 점검 (진단용, 바이알 전용) | - |
| `scripts/omx_leader_probe.py` | OMX-L 리더 읽기 전용 점검 + 그리퍼 열림/닫힘 normalized 값 측정 | - |
| `utils/omx_recorder.py` | `LeRobotRecorder` 의 서브클래스: 영상 버퍼를 GPU 가 아니라 CPU 에 쌓는다 (8 GB VRAM 에서 S 키로 멈추던 문제 수정) | `utils/lerobot_recorder.py` |

등록되는 환경: `Lerobot-Omx-Teleop-Base`, `-Task`, `-Vials-To-Rack`, `-Vials-To-Rack-DR`, `-Vials-To-Rack-Eval`, `-Vials-To-Rack-DR-Eval`, `-Cube-To-Cup`, `-Cube-To-Cup-Eval`

## omx_f_isaaclab.usda 가 하는 일
- `gripper_joint_2` 의 PhysxMimicJointAPI 제거 + `gripper_joint_1` 과 같은 DriveAPI 부여
  (Pseudo-Lab 의 teleop 스크립트가 런타임에 하던 처리를 USD 에 미리 반영. 종동 핑거는 액션 term 이 구동)
- 에셋 내장 `_dome` 라이트 비활성화
- pxr 로 합성 결과를 확인함: mimic 제거됨, joint_2 에 drive(stiffness 17.8 / damping 0.6 / maxForce 1000) 존재, dome 비활성.

## 실행
컨테이너 실행과 전체 절차는 저장소 루트 `README.md` 의 "ROBOTIS OMX-F 포팅" 절에 있다.

```bash
# 환경 등록 확인 (파이프 없이)
list_envs

# 리더 그리퍼 값 측정 (읽기 전용)
python source/sim_to_real_so101/scripts/omx_leader_probe.py --port /dev/ttyACM0 --id omx_leader_arm

# 텔레옵 + 녹화 (OMX-L 리더, USB-C). 그리퍼 값은 본인 리더에서 측정한 값으로 바꿀 것
python source/sim_to_real_so101/scripts/omx_lerobot_agent.py \
  --task Lerobot-Omx-Teleop-Cube-To-Cup --port /dev/ttyACM0 --robot_id omx_leader_arm \
  --zero_mode none --gripper_closed_norm 51.40 --gripper_open_norm 66.77 \
  --repo_id local/omx_cube_to_cup --repo_root datasets/omx_cube_to_cup \
  --task_name "Pick up the cube and place it in the cup"

# 평가 (GR00T 서버가 떠 있어야 함. 미검증)
python source/sim_to_real_so101/scripts/omx_lerobot_eval.py \
  --task Lerobot-Omx-Teleop-Cube-To-Cup-Eval --policy_host localhost --policy_port 5555 \
  --rename_map '{"ego": "front", "external_D455": "top"}' \
  --lang_instruction "Pick up the cube and place it in the cup" \
  --gripper_closed_norm 51.40 --gripper_open_norm 66.77
```

## 데이터 규약 (실물 OMX 와 이어지는 부분)
- action / observation.state 는 **LeRobot 리더 normalized 단위** 6차원 (`shoulder_pan.pos` ... `gripper.pos`).
  관절은 ±100 = ±180° (deg = n × 1.8), 그리퍼는 CLOSED_NORM → 0°, OPEN_NORM → 100°.
- 이름과 순서가 SO-101 워크숍·Pseudo-Lab OMX 데이터셋과 같아서, 상태 5 + 그리퍼 1 구성인
  워크숍의 GR00T modality 설정을 그대로 쓸 수 있을 가능성이 높다 (카메라 키만 맞추면 됨).

## 반드시 확인/조정할 것
1. **그리퍼 상수** (`OmxInterface.GRIPPER_CLOSED_NORM=50.0`, `GRIPPER_OPEN_NORM=55.3`): Pseudo-Lab 팀의 리더 개체 실측값이다.
   `omx_leader_probe.py` 로 본인 리더의 값을 재서 `--gripper_closed_norm/--gripper_open_norm` 으로 넣을 것 (이 포크 작성자의 리더는 열림 66.77 / 닫힘 51.40).
2. **리더 zero**: 리더를 sim 초기 자세에 맞춘 뒤 zero-pose homing(또는 Pseudo-Lab 의 Z 정렬 json)을 쓰면 zero≈0 이 되어
   기록된 action 이 실물 팔로워 공간과 일치한다. 녹화와 평가에서 같은 zero/그리퍼 상수를 써야 한다.
3. **로봇 방향/위치**: `assets/omx.py` 의 `ROBOT_POS`, `ROBOT_YAW_DEG`. omx_f 가 zero 자세에서 +x 로 뻗는다고 가정했다.
4. **ego 카메라 마운트**: `tasks/omx_task_env_cfg.py` 의 `EGO_CAM_POS/ROT` 는 초기값이다. 실제 OMX 카메라 위치에 맞출 것.
5. **작업 공간**: OMX 의 수평 도달 거리는 약 0.4 m. 바이알/랙 좌표(`omx_vials_to_rack_env_cfg.py`)가 닿는지 확인.
6. **게인**: `assets/omx.py` 는 USD 의 값(stiffness 17.8 등)을 그대로 쓴다. 너무 뻣뻣하거나 진동하면 actuator 의 stiffness/damping 에 값을 직접 지정.
7. **로봇 색 랜덤화 생략**: SO-101 머티리얼 경로(`Looks/material_a_3d_printed`)에 묶여 있어 OMX 에서는 뺐다.
8. 실물 팔로워/로컬 lerobot 정책(ACT 등) 경로는 연결하지 않았다. `OmxInterface.init_device` 는 follower 모드에서 로봇 객체를 만들지 않는다
   (GR00T 원격 정책 경로만 지원).

## Isaac Sim 에서 확인된 것 (Base / Task / Vials-To-Rack 환경, 8 GB VRAM 에서 구동됨)
- 팔 기하·관절 부호는 USD 에서 읽은 값과 맞다 (`omx_reach_agent.py` 의 IK/FK 오차 2~3 mm).
- 베이스 위치는 SO-101 의 x=-0.05 대신 x=0.0 (`ROBOT_POS`). 바이알까지 어깨 기준 수평 거리가 0.24~0.29 m 가 된다.
- 랙은 SO-101 의 (0.18, 0.08) 대신 (0.15, 0.12). OMX 그리퍼 손이 넓어서(좌우 약 ±5 cm) 원래 위치에서는 랙 옆면에 닿았다.
- **손목 롤**: omx_f 는 joint5=0 에서 핑거가 위아래(피치 평면)로 열린다. 바닥에 누운 바이알을 옆에서 집으려면 joint5≈90°.
- Vial_opaque.usda 의 원점은 원뿔 끝 쪽이다. 몸통 중심은 원점에서 축 방향 0.0413 m.
- 바이알이 랙에 이미 들어간 채 시작하는 reset 이 있다 (워크숍의 `rack_placement_prob=0.33`). 오류가 아니다.
- 바이알 x 는 SO-101 의 0.23 대신 0.20 (reset 시 ±0.04).
- 자동 파지(`omx_reach_agent.py`): 가까운 바이알(0.26 m)에서 1회 성공했지만 이후 3회는 실패했다. 닫을 때 그리퍼가 27° 부근에서
  멈춰 바이알을 물고 있다가 들어올릴 때 빠져나간다. 두 핑거 사이 V 자 홈이 바이알을 밀어내므로 마찰이 부족하다고 보고
  손가락/바이알 마찰을 1.5 로 올렸다 (`omx_vials_to_rack_env_cfg.py` 의 startup 이벤트). 이 효과는 아직 확인 전이다.
  팔 gain 을 4.5 배로 올려 봤지만 도움이 되지 않아 USD 값(17.8)으로 되돌렸다.

## Cube-To-Cup / 녹화에서 확인된 것
- 바이알 출력 전 단순화를 위해 만든 환경이다. 리더 텔레옵으로 큐브를 집어 컵에 넣는 것이 된다.
- 리더 연결, 관절 부호, 그리퍼 매핑은 이 환경에서 실제 리더로 확인했다 (zero 자세에서 시작).
- 원본 `LeRobotRecorder` 는 `S` 를 누른 첫 프레임에서 2 분(3600 프레임) 분량 RGB 버퍼를 GPU 에 미리 할당한다.
  640x480x3 기준 카메라당 3.3 GB 라서 카메라 2 대면 6.6 GB. Isaac Sim 이 이미 약 5 GB 를 쓰는 8 GB 노트북에서는 앱이 응답 없음이 된다.
  `utils/omx_recorder.py` 는 프레임을 CPU 에 쌓아서(GPU 사용량 증가 없음) 해결했다.
  이후 3 개 에피소드(1236 프레임)가 저장되었고 LeRobotDataset 으로 다시 읽어 영상 디코딩까지 확인했다.
- 키: `S` 시작/종료(저장), `R` 리셋(녹화 중이면 저장), `C` 취소(버림, 미검증).
- 영상은 AV1 mp4 라서 VLC 에서 열리지 않을 수 있다. 데이터는 정상이다.

## 아직 안 된 것
- Eval 환경의 성공 판정, GR00T 학습/평가, 실물 팔로워 배포.
- 컵은 정사각 임시 형상이다. 실물 컵 크기로 `assets/usd/cup.usda` 와 `omx_cube_to_cup_env_cfg.py` 의 `CUP_POS` 를 맞출 것.
- 바이알 태스크는 바이알을 출력한 뒤 실측 크기로 다시 맞춘다.

## 주의: NVIDIA-RFM-Lab 과 패키지 이름 충돌
NVIDIA-RFM-Lab 의 `Simulation_Team/sim2real/sim_to_real_so101` 도 같은 패키지 이름이다.
둘을 동시에 PYTHONPATH 에 올리지 말 것 (이 포팅은 워크숍 패키지만 사용한다).

## 검증 상태
- 실행한 것: Isaac Sim 5.1 / Isaac Lab 2.3.2 (RTX 5060 Laptop 8 GB)에서 `list_envs`, Base/Task/Vials/Cube-To-Cup 환경 구동,
  OMX-L 리더 텔레옵(Cube-To-Cup), LeRobotDataset 녹화(3 에피소드)와 재로딩.
  그 외 `OmxInterface` 변환의 왕복 수치 테스트, USD 합성 결과 확인.
- **실행하지 못한 것**: Eval 환경의 성공 종료, GR00T 학습과 원격 정책 평가, 실물 팔로워.
