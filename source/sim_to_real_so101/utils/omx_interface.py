# SPDX-License-Identifier: Apache-2.0
#
# OMX 용 LeRobot 인터페이스 (워크숍 utils/lerobot_interface.py 의 OMX 대응 신규 파일).
#
# 워크숍의 LeRobotSO101Interface 를 상속해서 "정규화 값 <-> sim 관절각(rad)" 변환 두 개만 OMX 규약으로 덮어쓴다.
# 나머지(레코더/GR00T 원격 정책/평가 루프가 쓰는 메서드)는 그대로 재사용한다.
#
# 변환 규약 (Pseudo-Lab/NVIDIA-RFM-Lab 의 lerobot_omx_interface.py 에서 검증된 상수를 그대로 가져옴)
#   관절 5축 : 정규화 ±100 = ±180°  ->  deg = (n - zero) * 1.8   (sim 관절과 각도 1:1)
#   그리퍼   : n = CLOSED_NORM(꽉 닫힘) -> 0°, OPEN_NORM(활짝) -> 100°  (선형)
#   action / observation.state 모두 이 "정규화 값" 단위로 기록된다 (LeRobot 리더-팔로워 텔레옵과 같은 공간).
#
# !! 그리퍼 상수(CLOSED/OPEN)는 리더 개체마다 다르다. 아래 기본값은 Pseudo-Lab 팀이 자기 OMX-L 로 실측한 값이다.
#    본인 리더에서 트리거를 꽉 닫았을 때/끝까지 열었을 때의 normalized 값을 읽어서 --gripper_closed_norm,
#    --gripper_open_norm 으로 덮어쓰거나 아래 클래스 상수를 고칠 것.
import json
import os

import torch

from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface


class OmxInterface(LeRobotSO101Interface):
    # 이름은 SO-101 과 동일하게 둔다: Pseudo-Lab 의 OMX 데이터셋/변환 스크립트(JOINT_FEATURE_NAMES)와 호환되고,
    # 워크숍의 레코더/GR00T 클라이언트가 이 이름 순서(SO101_JOINT_ORDER)를 그대로 쓰기 때문.
    # 순서 = [joint1, joint2, joint3, joint4, joint5, gripper_joint_1]

    JOINT_DEG_PER_UNIT = 1.8
    GRIPPER_CLOSED_NORM = 50.0
    GRIPPER_OPEN_NORM = 55.3
    GRIPPER_OPEN_DEG = 100.0
    # 관절별 고정 오프셋(deg). j5 의 -90° 손목 보정은 omx_f.usd 쪽(joint5 localRot0=Rx(-90))에 구워져 있어 0.
    JOINT_OFFSET_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]
    # 방향 부호. 실물 OMX-L 은 sim omx_f 와 같은 규약이라 전부 +1 (Pseudo-Lab 육안 검증).
    SIGN = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def __init__(
        self,
        *args,
        leader_type: str = "omx",
        zero_json: str | None = None,
        gripper_closed_norm: float | None = None,
        gripper_open_norm: float | None = None,
        **kwargs,
    ):
        if leader_type not in ("omx", "so101"):
            raise ValueError(f"leader_type 은 'omx' 또는 'so101' 이어야 한다: {leader_type}")
        super().__init__(*args, **kwargs)
        self.leader_type = leader_type
        if gripper_closed_norm is not None:
            self.GRIPPER_CLOSED_NORM = float(gripper_closed_norm)
        if gripper_open_norm is not None:
            self.GRIPPER_OPEN_NORM = float(gripper_open_norm)

        self._sign = torch.tensor(self.SIGN, dtype=torch.float32, device=self.device)
        self._offset = torch.tensor(self.JOINT_OFFSET_DEG, dtype=torch.float32, device=self.device)

        # zero 정렬: "sim 초기 자세에 리더를 맞추고 캡처한 normalized 값(6)". 없으면 None (= 0 기준).
        self._zero_norm: list[float] | None = None
        self.load_zero_norm(zero_json)

    # ------------------------------------------------------------------ zero 정렬
    def default_zero_json_path(self) -> str:
        base = os.environ.get("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
        return os.path.join(base, "calibration", "teleoperators", "omx_leader", f"{self.id}_zero.json")

    def load_zero_norm(self, path: str | None = None) -> bool:
        path = path or self.default_zero_json_path()
        if not os.path.exists(path):
            print(f"[OMX] zero json 없음 -> zero=0 기준으로 동작: {path}", flush=True)
            self._zero_norm = None
            return False
        with open(path) as f:
            self._zero_norm = [float(v) for v in json.load(f)["zero_norm"]]
        print(f"[OMX] zero 로드: {path} -> {['%.2f' % v for v in self._zero_norm]}", flush=True)
        self._warn_if_zero_far()
        return True

    def set_zero_norm(self, values) -> None:
        """리더의 현재 normalized 값(6)을 zero 기준으로 설정 (예: 시작 시 캡처)."""
        self._zero_norm = [float(v) for v in values]
        print(f"[OMX] zero 캡처: {['%.2f' % v for v in self._zero_norm]}", flush=True)
        self._warn_if_zero_far()

    def _warn_if_zero_far(self):
        # action 은 "리더 raw normalized" 로 기록된다. zero 가 0 에서 멀면, 기록된 action 이 실물 팔로워가 받는
        # 값 공간과 어긋날 수 있다(리더를 zero-pose 로 homing 해서 zero≈0 이 되게 하는 것을 권장).
        if self._zero_norm is not None and max(abs(v) for v in self._zero_norm[:5]) > 3.0:
            print(
                "[OMX][WARN] zero 가 0 에서 3 units(≈5.4°) 이상 떨어져 있다. 기록된 action 이 실물 팔로워 공간과 "
                "어긋날 수 있으니 리더를 zero-pose 로 homing 했는지 확인할 것.",
                flush=True,
            )

    # ------------------------------------------------------------------ LeRobot 장치
    def make_cfg(self):
        if self.kind == "leader" and self.leader_type == "omx":
            # lerobot 에 omx_leader 가 포함된 버전 필요 (Pseudo-Lab 이 쓰는 것과 동일한 import 경로)
            from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig

            return OmxLeaderConfig(port=self.port, id=self.id)
        return super().make_cfg()

    def init_device(self, visualize: bool = False):
        if self.kind == "follower":
            # 시뮬레이션 평가는 실물 팔로워를 만들지 않는다(GR00T 원격 정책 경로는 robot 객체가 필요 없음).
            # lerobot 로컬 정책(ACT 등) 경로를 쓰려면 이 분기를 지우고 OMX 팔로워 config 를 연결해야 한다.
            self.robot = None
            print("[OMX] follower 모드: 실물 로봇 연결 없이 sim 변환만 사용", flush=True)
            return
        super().init_device(visualize=visualize)

    # ------------------------------------------------------------------ 정규화 <-> rad
    def get_mapped_actions_vectorized(self, raw_values: torch.Tensor) -> torch.Tensor:
        """정규화 값(6) -> sim 목표각(rad, 6)."""
        raw = raw_values.to(self._sign.dtype)
        deg = torch.empty_like(raw)
        span = self.GRIPPER_OPEN_NORM - self.GRIPPER_CLOSED_NORM
        if self._zero_norm is not None:
            z = torch.tensor(self._zero_norm, dtype=raw.dtype, device=raw.device)
            deg[:5] = (raw[:5] - z[:5]) * self.JOINT_DEG_PER_UNIT + self._offset
            deg[5] = (raw[5] - z[5]) / span * self.GRIPPER_OPEN_DEG
        else:
            deg[:5] = raw[:5] * self.JOINT_DEG_PER_UNIT + self._offset
            deg[5] = (raw[5] - self.GRIPPER_CLOSED_NORM) / span * self.GRIPPER_OPEN_DEG
        return deg * torch.pi / 180.0 * self._sign

    def get_raw_actions_from_radians(self, rad_values: torch.Tensor) -> torch.Tensor:
        """sim 관절각(rad, 6) -> 정규화 값(6). get_mapped_actions_vectorized 의 정확한 역변환."""
        rad = rad_values.to(self._sign.dtype)
        deg = rad * 180.0 / torch.pi * self._sign
        raw = torch.empty_like(deg)
        span = self.GRIPPER_OPEN_NORM - self.GRIPPER_CLOSED_NORM
        if self._zero_norm is not None:
            z = torch.tensor(self._zero_norm, dtype=deg.dtype, device=deg.device)
            raw[:5] = (deg[:5] - self._offset) / self.JOINT_DEG_PER_UNIT + z[:5]
            raw[5] = deg[5] / self.GRIPPER_OPEN_DEG * span + z[5]
        else:
            raw[:5] = (deg[:5] - self._offset) / self.JOINT_DEG_PER_UNIT
            raw[5] = deg[5] / self.GRIPPER_OPEN_DEG * span + self.GRIPPER_CLOSED_NORM
        return raw
