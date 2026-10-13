# SPDX-License-Identifier: Apache-2.0
#
# OMX Cube-To-Cup 태스크의 성공 판정.
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def cube_in_cup(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    cup_xy: tuple[float, float] = (0.17, 0.10),
    inner_half: float = 0.035,
    z_min: float = 0.04,
    z_max: float = 0.075,
) -> torch.Tensor:
    """큐브 중심이 컵 안쪽 범위 안에 들어가 있으면 True. 좌표는 env 원점 기준(월드 - env_origins).

    컵 안쪽 반폭은 0.040 m, 컵 바닥 위 큐브 중심 높이는 약 0.056 m, 컵 림 높이는 0.095 m 이다.
    inner_half(기본 0.035)는 벽에 걸쳐 있는 경우를 빼기 위해 안쪽 반폭보다 조금 작게 잡았고,
    z_max(0.075)는 림 위에 걸쳐 있는 경우를 빼기 위한 값이다.
    """
    cube = env.scene[cube_cfg.name]
    pos = cube.data.root_pos_w - env.scene.env_origins
    dx = (pos[:, 0] - cup_xy[0]).abs()
    dy = (pos[:, 1] - cup_xy[1]).abs()
    return (dx < inner_half) & (dy < inner_half) & (pos[:, 2] > z_min) & (pos[:, 2] < z_max)
