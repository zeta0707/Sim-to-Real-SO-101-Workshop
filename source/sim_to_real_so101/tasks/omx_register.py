# SPDX-License-Identifier: Apache-2.0
#
# OMX 환경의 gym 등록.
# import_packages() 는 하위 "패키지"만 import 하고 일반 모듈은 import 하지 않으므로,
# tasks/__init__.py 끝에 `from . import omx_register` 한 줄을 추가해 이 모듈을 불러온다.
# (`import sim_to_real_so101.tasks` 만으로 등록됨)
import gymnasium as gym

_OMX_ENVS = {
    "Lerobot-Omx-Teleop-Base": "omx_env_cfg:OmxTeleopEnvCfg",
    "Lerobot-Omx-Teleop-Task": "omx_task_env_cfg:OmxTaskEnvCfg",
    "Lerobot-Omx-Teleop-Vials-To-Rack": "omx_vials_to_rack_env_cfg:OmxVialsToRackEnvCfg",
    "Lerobot-Omx-Teleop-Vials-To-Rack-DR": "omx_vials_to_rack_env_cfg:OmxVialsToRackDREnvCfg",
    "Lerobot-Omx-Teleop-Vials-To-Rack-Eval": "omx_vials_to_rack_env_cfg:OmxVialsToRackEvalEnvCfg",
    "Lerobot-Omx-Teleop-Vials-To-Rack-DR-Eval": "omx_vials_to_rack_env_cfg:OmxVialsToRackEvalDREnvCfg",
    "Lerobot-Omx-Teleop-Cube-To-Cup": "omx_cube_to_cup_env_cfg:OmxCubeToCupEnvCfg",
    "Lerobot-Omx-Teleop-Cube-To-Cup-Eval": "omx_cube_to_cup_env_cfg:OmxCubeToCupEvalEnvCfg",
    # 강화학습용 (상태 기반 관찰, 카메라 없음). scripts/omx_rl_train.py 로 학습한다.
    "Lerobot-Omx-Cube-To-Cup-RL": "omx_cube_to_cup_rl_env_cfg:OmxCubeToCupRLEnvCfg",
}

_PKG = __name__.rsplit(".", 1)[0]  # sim_to_real_so101.tasks

for _id, _entry in _OMX_ENVS.items():
    if _id in gym.registry:
        continue
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{_PKG}.{_entry}"},
    )
