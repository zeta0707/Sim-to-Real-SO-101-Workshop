# SPDX-License-Identifier: Apache-2.0
#
# OMX Cube-To-Cup 강화학습용 MDP 요소: 델타 관절 액션, 상태 관찰, 보상, 종료 조건.
# 카메라 없이 상태(관절, 큐브, 컵 위치)만 쓴다. 성공 판정은 omx_terms.cube_in_cup 과 같다.
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from sim_to_real_so101.mdp.omx_actions import OmxJointPositionAction, OmxJointPositionActionCfg


# ---------------------------------------------------------------------------
# 액션: 델타 관절 목표
#   target_t = clip(target_{t-1} + delta_scale * clip(a, -1, 1), 관절 한계)
# 정책 출력 a 는 [-1, 1] 로 보고, 한 제어 스텝에 움직일 수 있는 최대 각도가 delta_scale 이다.
# 현재 관절각이 아니라 이전 "목표"에 더하므로 중력으로 처진 만큼 누적 오차가 생기지 않는다.
# 리셋 시 목표는 리셋된 현재 관절각으로 되돌린다. gripper_joint_2 미러는 부모 클래스가 처리한다.
# ---------------------------------------------------------------------------
class OmxDeltaJointPositionAction(OmxJointPositionAction):
    cfg: "OmxDeltaJointPositionActionCfg"

    def __init__(self, cfg: "OmxDeltaJointPositionActionCfg", env):
        super().__init__(cfg, env)
        if len(cfg.delta_scale) != self.action_dim:
            raise ValueError(f"delta_scale 길이 {len(cfg.delta_scale)} != 액션 차원 {self.action_dim}")
        self._delta = torch.tensor(cfg.delta_scale, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._target = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        a = torch.clamp(actions, -1.0, 1.0)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]  # (N, J, 2)
        self._target = torch.clamp(self._target + a * self._delta, limits[..., 0], limits[..., 1])
        self._processed_actions = self._target.clone()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        # 이벤트(reset_joints)가 액션 리셋보다 먼저 실행되므로 여기서 읽는 값이 리셋된 관절각이다.
        self._target[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]


@configclass
class OmxDeltaJointPositionActionCfg(OmxJointPositionActionCfg):
    class_type: type = OmxDeltaJointPositionAction

    # joint1~5 는 0.05 rad/step, gripper 는 0.08 rad/step (30 Hz 제어에서 각각 약 1.5, 2.4 rad/s)
    delta_scale: tuple = (0.05, 0.05, 0.05, 0.05, 0.05, 0.08)


# ---------------------------------------------------------------------------
# 관찰 (모두 env 원점 기준 좌표. 로봇 베이스가 env 원점에 있고 yaw 0 이라 로봇 좌표와 같다)
# ---------------------------------------------------------------------------
def ee_pos_env(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee = env.scene[ee_frame_cfg.name]
    return ee.data.target_pos_w[:, 0, :] - env.scene.env_origins


def cube_pos_env(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    return env.scene[cube_cfg.name].data.root_pos_w - env.scene.env_origins


def cube_quat_env(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    return env.scene[cube_cfg.name].data.root_quat_w


def cube_minus_ee(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    return cube_pos_env(env, cube_cfg) - ee_pos_env(env, ee_frame_cfg)


def cube_minus_cup_xy(
    env: ManagerBasedRLEnv,
    cup_xy: tuple[float, float],
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    pos = cube_pos_env(env, cube_cfg)
    cup = torch.tensor(cup_xy, dtype=pos.dtype, device=pos.device)
    return pos[:, :2] - cup


# ---------------------------------------------------------------------------
# 보상
# ---------------------------------------------------------------------------
def ee_pointing_down(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """그리퍼가 가리키는 방향(ee 프레임 +x, link5 에서 핑거 쪽)이 아래를 향할수록 1 에 가까운 값.
    수평이면 0, 정확히 아래면 1, 위를 향하면 0."""
    ee = env.scene[ee_frame_cfg.name]
    q = ee.data.target_quat_w[:, 0, :]
    x_axis = torch.zeros_like(ee.data.target_pos_w[:, 0, :])
    x_axis[:, 0] = 1.0
    az = quat_apply(q, x_axis)[:, 2]
    return torch.clamp(-az, 0.0, 1.0)


def reach_cube(
    env: ManagerBasedRLEnv,
    std: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """ee 가 큐브에 가까울수록 1 에 가까운 값."""
    d = torch.norm(cube_minus_ee(env, cube_cfg, ee_frame_cfg), dim=1)
    return 1.0 - torch.tanh(d / std)


def cube_lifted(
    env: ManagerBasedRLEnv, z_min: float, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube")
) -> torch.Tensor:
    """큐브 중심이 z_min 보다 높으면 1. 매트 위 정지 높이는 약 0.052 m, 컵 림 높이는 0.095 m 이다."""
    return (cube_pos_env(env, cube_cfg)[:, 2] > z_min).float()


def carry_to_cup(
    env: ManagerBasedRLEnv,
    std: float,
    z_min: float,
    cup_xy: tuple[float, float],
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """들린 상태에서만, 큐브가 컵 xy 에 가까울수록 보상."""
    d = torch.norm(cube_minus_cup_xy(env, cup_xy, cube_cfg), dim=1)
    return cube_lifted(env, z_min, cube_cfg) * (1.0 - torch.tanh(d / std))


def release_over_cup(
    env: ManagerBasedRLEnv,
    z_min: float,
    cup_xy: tuple[float, float],
    xy_tol: float,
    open_min: float,
    open_max: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper_joint_1"]),
) -> torch.Tensor:
    """큐브를 든 채 컵 위(xy 오차 < xy_tol)에 있을 때, 그리퍼가 open_min 을 넘어 열린 정도(0~1)를 보상한다.

    open_min 은 큐브를 쥐고 있을 때의 손가락 벌어짐(큐브 폭 3 cm 에 해당, 약 0.7 rad)보다 커야 한다.
    그렇지 않으면 쥐고만 있어도 보상이 나와서 놓을 이유가 없어진다.

    컵 림(0.095 m)이 높아서 큐브를 쥔 채로는 컵 안으로 내릴 수 없다. 컵 위에서 놓아야 성공하므로
    '들고만 있기'가 보상을 독점하지 않도록 놓는 동작에 직접 신호를 준다.
    """
    d = torch.norm(cube_minus_cup_xy(env, cup_xy, cube_cfg), dim=1)
    gate = cube_lifted(env, z_min, cube_cfg) * (d < xy_tol).float()
    g = env.scene[gripper_cfg.name].data.joint_pos[:, gripper_cfg.joint_ids].reshape(env.num_envs, -1)[:, 0]
    return gate * torch.clamp((g - open_min) / (open_max - open_min), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 종료: 큐브가 매트에서 떨어졌거나 작업 영역 밖으로 밀려났다
# ---------------------------------------------------------------------------
def cube_out_of_bounds(
    env: ManagerBasedRLEnv,
    z_min: float = 0.005,
    max_radius: float = 0.6,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    pos = cube_pos_env(env, cube_cfg)
    return (pos[:, 2] < z_min) | (torch.norm(pos[:, :2], dim=1) > max_radius)
