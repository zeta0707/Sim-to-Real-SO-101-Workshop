# SPDX-License-Identifier: Apache-2.0
#
# 로컬 LeRobot 정책(ACT 등)을 시뮬레이션 평가에 연결하는 클래스.
# GR00TRemotePolicy 와 같은 인터페이스(connect / reset / get_action)를 제공해서 omx_lerobot_eval.py 가 그대로 쓴다.
#
# 워크숍의 LeRobotSO101Interface.make_policy 는 실물 로봇 객체(robot.observation_features)가 필요하지만
# OMX 평가는 로봇 객체를 만들지 않으므로(OmxInterface.init_device 의 follower 분기), 필요한 부분만 직접 구현했다.
# 입력 키 이름은 녹화 데이터셋과 같다: observation.state, observation.images.<카메라 이름>.
import os

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


class OmxLocalPolicy:
    def __init__(
        self,
        robot_iface,
        checkpoint: str,
        task: str,
        device: str = "cuda",
        n_action_steps: int | None = None,
    ):
        self._iface = robot_iface
        self._checkpoint = checkpoint
        self._task = task
        self._device = device
        self._n_action_steps = n_action_steps
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None

    def connect(self):
        """체크포인트(pretrained_model 폴더)에서 정책과 전/후처리 파이프라인을 읽는다."""
        ckpt = self._checkpoint
        if not os.path.isdir(ckpt):
            raise FileNotFoundError(f"체크포인트 폴더가 없다: {ckpt}")
        print(f"[INFO]: 체크포인트 파일: {sorted(os.listdir(ckpt))}", flush=True)

        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.pretrained_path = ckpt
        cfg.device = self._device
        if self._n_action_steps is not None:
            # ACT: 정책이 한 번 추론해 낸 chunk(chunk_size) 중 몇 스텝을 실행하고 다시 추론할지.
            if self._n_action_steps > cfg.chunk_size:
                raise ValueError(f"n_action_steps({self._n_action_steps}) 는 chunk_size({cfg.chunk_size}) 이하여야 한다")
            cfg.n_action_steps = self._n_action_steps
        print(f"[INFO]: policy type={cfg.type}, chunk_size={getattr(cfg, 'chunk_size', None)}, "
              f"n_action_steps={getattr(cfg, 'n_action_steps', None)}", flush=True)

        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(ckpt, config=cfg)
        self.policy.to(self._device)
        self.policy.eval()

        # 정규화 통계는 체크포인트의 전/후처리 파일에 들어 있다. 입력 장치만 덮어쓴다.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": self._device}},
        )
        print("[INFO]: 정책/전처리/후처리 로드 완료", flush=True)

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _build_observation(self, joint_positions: torch.Tensor, visual_obs: dict) -> dict:
        # state: 녹화 때와 같은 변환(sim rad -> 정규화 값 6)
        state = self._iface.get_raw_actions_from_radians(joint_positions).detach().cpu().numpy().astype(np.float32)
        obs = {"observation.state": state}
        for camera in self._iface.cameras.keys():
            img = visual_obs[f"rgb_{camera}"][0].detach()
            # 녹화 때(OmxLeRobotRecorder)와 똑같이 uint8 HWC 로 변환한다.
            obs[f"observation.images.{camera}"] = np.ascontiguousarray(img.to(torch.uint8).cpu().numpy())
        return obs

    def get_action(self, joint_positions: torch.Tensor, visual_obs: dict, log: bool = False) -> torch.Tensor:
        """다음 sim 액션(rad, 6)을 반환한다. chunk 관리는 정책(select_action)이 한다."""
        observation = self._build_observation(joint_positions, visual_obs)
        action = predict_action(
            observation=observation,
            policy=self.policy,
            device=get_safe_torch_device(self._device),
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=self.policy.config.use_amp,
            task=self._task,
            robot_type="omx_f",
        )
        # action: 정규화 값 6 (녹화된 action 과 같은 공간)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu()
        raw = torch.as_tensor(np.asarray(action), dtype=torch.float32).reshape(-1)[:6].to(self._iface.device)
        return self._iface.get_mapped_actions_vectorized(raw)
