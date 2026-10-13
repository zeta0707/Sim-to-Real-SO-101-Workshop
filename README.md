# Train an OMX Robot From Sim-to-Real With NVIDIA Isaac

![OMX Vial to Rack Task](images/omx_banner.png)

Welcome to this workshop on sim-to-real transfer for the OMX robot!


## ROBOTIS OMX-F 포팅 (이 포크의 추가 내용)

이 포크는 SO-101 워크숍 환경을 **ROBOTIS OMX-F 팔로워 + OMX-L 리더**로 옮긴 것이다.   
기존 SO-101 파일은 `source/sim_to_real_so101/tasks/__init__.py` 끝에 `from . import omx_register` 한 줄을 추가한 것 외에는 수정하지 않았고  
OMX 관련 파일은 모두 `omx_` 접두사로 추가했다. 파일 목록, 데이터 규약, 시뮬레이션에서 확인된 사항은 [OMX_PORT.md](OMX_PORT.md)에 있다.  
로봇 에셋(`omx_f`)은 가짜연구소(Pseudo Lab)의 [NVIDIA-RFM-Lab](https://github.com/Pseudo-Lab/NVIDIA-RFM-Lab)(MIT)에서 가져왔다.  

### 동작 확인 상태

| 항목 | 상태 |
|---|---|
| OMX-F 환경 등록·구동 (`list_envs`) | 확인 (RTX 5060 Laptop 8 GB, Isaac Sim 5.1 / Isaac Lab 2.3.2) |
| OMX-L 리더 → 시뮬레이션 텔레옵 (`Cube-To-Cup`) | 확인 (큐브를 집어 컵에 넣음) |
| LeRobotDataset 녹화 (영상 2대 + state/action) | 확인 (3개 에피소드 저장, 재로딩·디코딩 확인) |
| `-Eval` 환경의 성공 판정(`cube_in_cup`) | 미확인 |
| GR00T 학습 / `omx_lerobot_eval.py` 평가 | 미확인 |
| 실물 OMX-F 팔로워 배포 | 미구현 (follower 모드는 GR00T 원격 정책 경로만 지원) |
| `Vials-To-Rack` 환경 | 환경은 뜨지만 자동 파지(진단 스크립트)는 불안정. 바이알 3D 출력 후 재조정 예정 |

### 빠른 시작

1. 클론. 에셋(`*.usd`, `*.usda`, `*.exr` 등)이 Git LFS 로 관리되므로 먼저 `git lfs install` 을 한다.
   ```bash
   git lfs install
   git clone <이 저장소 URL> Sim-to-Real-SO-101-Workshop
   cd Sim-to-Real-SO-101-Workshop
   ```
2. 시뮬레이션/텔레옵 이미지 빌드 (저장소 루트에서):
   ```bash
   docker build -t teleop-docker -f docker/sim/Dockerfile .
   ```
   `docker/sim/entrypoint.sh` 는 이미지에 복사되므로, 이 파일을 고쳤다면 다시 빌드해야 반영된다.
3. 컨테이너 실행 (저장소 루트에서). 아래 "Starting the images" 절의 `teleop` 컨테이너 명령과 같고, 리더의 calibration 폴더(`~/.cache/huggingface/lerobot/calibration`)와 `source/`, `datasets/`, `outputs/` 마운트가 들어 있다.
   ```bash
   xhost +
   docker run --name teleop -it --privileged --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
      -e "PRIVACY_CONSENT=Y" -e DISPLAY \
      -v /dev:/dev -v /run/udev:/run/udev:ro \
      -v $HOME/.Xauthority:/root/.Xauthority \
      -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
      -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
      -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
      -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
      -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
      -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
      -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
      -v ~/docker/isaac-sim/documents:/root/Documents:rw \
      -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
      -v ./docker/env:/root/env \
      -v $(pwd)/source:/workspace/Sim-to-Real-SO-101-Workshop/source \
      -v $(pwd)/outputs:/workspace/Sim-to-Real-SO-101-Workshop/outputs \
      -v $(pwd)/datasets:/workspace/Sim-to-Real-SO-101-Workshop/datasets \
      -v $(pwd)/docker/real/scripts:/workspace/Sim-to-Real-SO-101-Workshop/docker/real/scripts \
      teleop-docker:latest
   ```
   컨테이너가 시작될 때 entrypoint 가 패키지를 `pip install -e` 하고, OMX-L 리더용 `dynamixel-sdk` 가 없으면 자동으로 설치한다(네트워크 필요).
   `--rm` 이라 컨테이너를 내리면 컨테이너 안에서 한 설치는 사라진다. 마운트한 `source/`, `datasets/`, `outputs/` 는 호스트에 남는다.
4. 컨테이너 안에서 환경이 등록됐는지 확인한다:
   ```bash
   list_envs
   ```
   `Lerobot-Omx-Teleop-*` 8 개가 보여야 한다. 표 아래에 `post_quit(): incompatible function arguments ... None` traceback 과 `Replicator ... for removal` 경고가 나오는데, `list_envs` 가 종료할 때 Isaac Sim 이 내는 메시지라서 무시해도 된다.

### OMX-L 리더 연결

- 리더는 USB 로 연결하며 보통 `/dev/ttyACM0` 이다 (`ls /dev/ttyACM*`).
- OMX-L 은 Dynamixel 공장 보정값을 쓴다. LeRobot 은 `~/.cache/huggingface/lerobot/calibration/teleoperators/omx_leader/<id>.json` 을 쓴다. 이미 보정 파일이 있다면 먼저 폴더를 백업해 두는 것을 권장한다.
- **그리퍼 열림/닫힘 값은 리더 개체마다 다르다.** 먼저 측정한다:
  ```bash
  python source/sim_to_real_so101/scripts/omx_leader_probe.py --port /dev/ttyACM0 --id omx_leader_arm
  ```
  화면 안내에 따라 먼저 트리거를 **놓은 상태(열림)**로, 다음에 **끝까지 쥔 상태(닫힘)**로 각각 몇 초 유지하면 `--gripper_open_norm`, `--gripper_closed_norm` 값이 출력된다. 이 스크립트는 읽기만 하며 보정 파일은 건드리지 않는다. 아래 예의 `66.77` / `51.40` 은 이 포크 작성자의 리더에서 측정한 값이므로 **본인 값으로 바꿔야 한다.** 녹화와 평가에서 같은 값을 써야 한다.
- 실행 전에 리더를 시뮬레이션의 초기 자세(모든 관절 zero)에 맞춰 둔다. `--zero_mode none` 은 리더의 값 0 을 시뮬레이션 관절 0 으로 그대로 대응시킨다. 실행 직후 시뮬레이션 팔이 갑자기 튀지 않는지 확인한다.

### 텔레옵 + 녹화

```bash
python source/sim_to_real_so101/scripts/omx_lerobot_agent.py \
  --task Lerobot-Omx-Teleop-Cube-To-Cup \
  --port /dev/ttyACM0 --robot_id omx_leader_arm \
  --zero_mode none \
  --gripper_closed_norm 51.40 --gripper_open_norm 66.77 \
  --repo_id local/omx_cube_to_cup --repo_root datasets/omx_cube_to_cup \
  --task_name "Pick up the cube and place it in the cup"
```

- `--repo_id --repo_root --task_name` 을 모두 주면 녹화 모드가 되고, 하나라도 빼면 녹화 없이 텔레옵만 한다.
- 키: `S` 녹화 시작/종료(종료하면 에피소드가 **저장**된다), `R` 월드 리셋(녹화 중이었다면 **저장하고** 멈춘다), `C` 녹화 중인 에피소드를 **취소**(저장하지 않고 버림). 실패한 시도는 `C` 로 버린다(`C` 는 아직 실제로 눌러 확인하지 않았다).
- 저장은 별도 스레드에서 처리되며 `Episode N saved.` 가 나오면 끝난 것이다.
- `--repo_root` 폴더가 이미 있으면 기존 데이터셋을 열어 에피소드를 이어서 추가한다. 폴더만 있고 내용이 불완전하면 `folder already exists` 에러가 나므로 지우고 다시 시작한다.
- 녹화는 영상(ego, external_D455) 프레임을 CPU 메모리에 쌓는다(`utils/omx_recorder.py`). 1분 에피소드면 대략 3 GB 의 RAM 을 쓰므로 에피소드는 길지 않게 한다.

저장된 데이터셋 확인:

```bash
python - <<'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/omx_cube_to_cup", root="datasets/omx_cube_to_cup")
print("episodes:", ds.meta.total_episodes, "frames:", ds.meta.total_frames, "fps:", ds.fps)
print("features:", list(ds.features.keys()))
f = ds[0]
print("action", f["action"], "\nstate ", f["observation.state"])
print({k: tuple(v.shape) for k, v in f.items() if k.startswith("observation.images")})
EOF
```

영상은 AV1(mp4)이다. VLC 등 일부 플레이어는 열지 못할 수 있으며, `ffplay` 나 `mpv` 로는 열린다.

### OMX 환경(Task)

| 환경 | 설명 |
|---|---|
| `Lerobot-Omx-Teleop-Base` | 디버그: 로봇만 (카메라 없음) |
| `Lerobot-Omx-Teleop-Task` | 디버그: 라이트박스, 매트, 카메라 |
| `Lerobot-Omx-Teleop-Cube-To-Cup` | **큐브 하나를 집어 컵에 넣기.** 바이알 없이 시작하는 단순 태스크. 큐브 위치는 reset 마다 ±3 cm 랜덤 |
| `Lerobot-Omx-Teleop-Cube-To-Cup-Eval` | 위와 같고 성공(`cube_in_cup`) 시 종료, 제한 시간 20 초 |
| `Lerobot-Omx-Teleop-Vials-To-Rack` | 바이알을 집어 랙에 넣기 (원본 워크숍 태스크의 OMX 판) |
| `Lerobot-Omx-Teleop-Vials-To-Rack-DR` | 위 + 도메인 랜덤화 |
| `Lerobot-Omx-Teleop-Vials-To-Rack-Eval` / `-DR-Eval` | 평가용 |

Cube-To-Cup 의 장면 구성:
- 큐브: 한 변 3 cm, 질량 30 g, 마찰 1.5. 초기 위치 (0.22, -0.04) m 근처.
- 컵: 위가 열린 정사각 컵(내경 8 cm, 높이 6 cm, 벽 6 mm), (0.17, 0.10) m. 실물 컵 크기는 아직 반영하지 않았다.
- 성공 판정: 큐브 중심이 컵 중심 기준 xy ±3.5 cm 이고 높이가 0.04~0.075 m 일 때.
- 카메라: ego(손목)와 external_D455(탑뷰), 각 640x480.

### 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: dynamixel_sdk` | entrypoint 가 설치하지 못한 경우(네트워크). `python -m pip install --no-deps "dynamixel-sdk>=3.7.31,<3.9.0"` |
| `S` 를 누르면 Isaac Sim 이 "응답 없음" | 원본 `LeRobotRecorder` 가 2분 분량 영상 버퍼를 GPU 에 미리 할당(카메라 2대 6.6 GB)해서 8 GB VRAM 에서 멈추던 문제. `utils/omx_recorder.py`(CPU 버퍼)로 해결 |
| Ctrl-C 로 안 꺼짐 | 다른 터미널에서 `docker kill teleop` |
| `Failed to initialize dataset. folder already exists` | `datasets/...` 폴더가 불완전하게 남은 것. 비어 있으면 지우고 다시 실행 |
| 두 `sim_to_real_so101` 패키지 충돌 | NVIDIA-RFM-Lab 의 같은 이름 패키지를 PYTHONPATH 에 올리지 않는다 |

# Fork한 원본 README
# Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac

![SO-101 Vial to Rack Task](images/so101_banner.png)

Welcome to this workshop on sim-to-real transfer for the SO-101 robot!

This repository contains the assets and code to accompany this [learning content](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html).

The rest of this README will help you setup the environment and ensure everything is installed correctly.

You can also use this repo as a basis for trying out your own tasks.

## Requirements

This content was tested on the following GPUs:

- NVIDIA RTX 6000 Pro (Blackwell)
- NVIDIA RTX 5090 (Blackwell)
- NVIDIA RTX 6000 (Ada)

OS and Software tested:
- Ubuntu Linux >22.04
- Docker
- CUDA Toolkit
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


## Installation

1. Create directory and clone this repo
```bash
mkdir ~/sim2real
cd ~/sim2real
git clone https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop.git
```

### Building the Docker images

2. Navigate to the repo
```bash
cd ~/sim2real/Sim-to-Real-SO-101-Workshop
```

#### Teleop & Simulation container

3. From the repo root directory, run:
```bash
docker build -t teleop-docker -f docker/sim/Dockerfile .
```

#### Real Robot & Inference Server - this may take a while to build


For **Blackwell** architecture GPUs:

4. From the repo root directory, run:
```bash
./docker/real/build.sh blackwell
```
For **Ada** architecture GPUs:

4. From the repo root directory:
```bash
./docker/real/build.sh ada
```

5. Continue with the course instructions [here](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/datasets-and-models.html).

### Starting the images

To start the Teleop & Simulation container:

```bash
xhost + 
docker run --name teleop -it --privileged --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY \
   -v /dev:/dev \
   -v /run/udev:/run/udev:ro \
   -v $HOME/.Xauthority:/root/.Xauthority \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
   -v ./docker/env:/root/env \
   -v $(pwd)/source:/workspace/Sim-to-Real-SO-101-Workshop/source \
   -v $(pwd)/outputs:/workspace/Sim-to-Real-SO-101-Workshop/outputs \
   -v $(pwd)/datasets:/workspace/Sim-to-Real-SO-101-Workshop/datasets \
   -v $(pwd)/docker/real/scripts:/workspace/Sim-to-Real-SO-101-Workshop/docker/real/scripts \
   teleop-docker:latest
```

To start the Real Robot & Inference Server:

```bash
docker run -it --rm --name real-robot --network host --privileged --gpus all \
    -e DISPLAY \
    -v /dev:/dev \
    -v /run/udev:/run/udev:ro \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
    -v ./docker/env:/root/env \
    -v ~/sim2real/models:/workspace/models \
    -v $(pwd)/docker/real/scripts:/Isaac-GR00T/gr00t/eval/real_robot/SO100 \
    real-robot \
    /bin/bash
```

## Models and Datasets

### Downloading model weights

First, [install the HuggingFace command-line-interface (CLI)](https://huggingface.co/docs/huggingface_hub/en/guides/cli#command-line-interface-cli)

The models used in the course are listed in the course instructions [here](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/datasets-and-models.html).


You can either download them ahead of time, or as you get to them in the course.

## Tasks

### Tasks

#### Debug envs
- `Lerobot-So101-Teleop-Base` : Teleop debug
- `Lerobot-So101-Teleop-Task` : Lightbox, cameras, non-task related debug

#### Tasks
- `Lerobot-So101-Teleop-Vials-To-Rack` - Main task for the workshop - pick up the vial and place it in the yellow rack
- `Lerobot-So101-Teleop-Vials-To-Rack-DR` - Same as above but with domain randomization

#### Eval
- `Lerobot-So101-Teleop-Vials-To-Rack-Eval` - Evaluation without domain randomization (fixed orange robot, no lighting/mat DR)
- `Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval` - Evaluation with full domain randomization


## Commands

- `list_envs` - List environments in this repo
- `zero_agent` - Debug script with zero actions
- `random_agent` - Debug script with random actions
- `lerobot_agent` - LeRobot SO101 teleop script
- `lerobot_eval` - Model evaluation script
- `lerobot_push_dataset` - LeRobot Dataset push to hub script

## Contributions
We are not currently accepting contributions for this project.
