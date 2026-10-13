# SPDX-License-Identifier: Apache-2.0
#
# OMX 전용 액션 term.
#
# OMX 그리퍼는 gripper_joint_1 / gripper_joint_2 두 핑거가 서로 반대로 움직인다. 정책/리더 쪽은 그리퍼를
# 1축(gripper)으로만 다루므로, 액션은 6차원(joint1~5 + gripper_joint_1)으로 유지하고
# gripper_joint_2 의 목표각은 -1 * gripper_joint_1 로 이 term 이 대신 써 준다.
# (Pseudo-Lab teleop 스크립트의 "gripper_joint_2 는 -1 미러" 처리와 동일하며, USD 쪽 mimic 은
#  omx_f_isaaclab.usda 에서 제거되어 있다.)
from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class OmxJointPositionAction(JointPositionAction):
    """JointPositionAction + 종동 핑거 미러."""

    cfg: "OmxJointPositionActionCfg"

    def __init__(self, cfg: "OmxJointPositionActionCfg", env):
        super().__init__(cfg, env)
        mirror_ids, _ = self._asset.find_joints(cfg.mirror_joint_name)
        if len(mirror_ids) != 1:
            raise ValueError(f"mirror joint '{cfg.mirror_joint_name}' 을(를) 찾지 못했다: {mirror_ids}")
        self._mirror_ids = mirror_ids
        if cfg.source_joint_name not in self._joint_names:
            raise ValueError(
                f"source joint '{cfg.source_joint_name}' 가 액션 joint_names {self._joint_names} 에 없다."
            )
        self._source_idx = self._joint_names.index(cfg.source_joint_name)

    def apply_actions(self):
        super().apply_actions()
        src = self.processed_actions[:, self._source_idx : self._source_idx + 1]
        self._asset.set_joint_position_target(src * self.cfg.mirror_gain, joint_ids=self._mirror_ids)


@configclass
class OmxJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = OmxJointPositionAction

    mirror_joint_name: str = "gripper_joint_2"
    source_joint_name: str = "gripper_joint_1"
    mirror_gain: float = -1.0
