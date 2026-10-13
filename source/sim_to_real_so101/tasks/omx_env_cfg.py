# SPDX-License-Identifier: Apache-2.0
#
# OMX-F teleop 베이스 환경. tasks/so101_env_cfg.py 의 OMX 대응 신규 파일.
#
# SO-101 과 달라진 점
#   - 로봇: OMX_F_CFG (joint1~5 + gripper_joint_1, 종동 gripper_joint_2 는 액션 term 이 -1 미러)
#   - 액션: OmxJointPositionActionCfg (6차원, 단위 rad)
#   - 관측 joint_pos 는 제어 대상 6축만 반환한다. omx_f 는 관절이 7개(gripper_joint_2 포함)라서
#     그대로 두면 평가/기록 스크립트가 기대하는 6차원과 어긋난다.
#   - ee_frame: link0 기준 link5 + end_effector_link 오프셋
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass

from sim_to_real_so101.assets.omx import OMX_ARM_JOINTS, OMX_F_CFG
from sim_to_real_so101.mdp import (
    ee_frame_state,
    joint_pos,
    joint_pos_rel,
    reset_joints_by_offset,
)
from sim_to_real_so101.mdp.omx_actions import OmxJointPositionActionCfg

# 제어 대상 6축만 순서 고정으로 선택 (preserve_order=True: OMX_ARM_JOINTS 순서 그대로)
ARM6 = SceneEntityCfg("robot", joint_names=OMX_ARM_JOINTS, preserve_order=True)


@configclass
class LerobotOmxBaseSceneCfg(InteractiveSceneCfg):
    env_spacing = 4.0
    num_envs = 1

    robot: ArticulationCfg = OMX_F_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                # FrameTransformer 는 rigid body 만 지원한다. end_effector_link 는 Xform 이라서
                # link5(rigid body) 를 대상으로 하고 USD 상의 오프셋(x=0.09193, y=-0.0016)을 적용한다.
                prim_path="{ENV_REGEX_NS}/Robot/link5",
                name="gripper",
                offset=OffsetCfg(pos=(0.09193, -0.0016, 0.0)),
            ),
        ],
    )


@configclass
class OmxBaseLitSceneCfg(LerobotOmxBaseSceneCfg):
    """Base 환경 전용: 조명이 없으면 어두운 로봇이 거의 안 보여서 돔 라이트를 추가한다.
    (Task/Vials 환경은 라이트박스 조명을 쓰므로 이 씬을 쓰지 않는다.)"""

    sky_light = AssetBaseCfg(
        prim_path="/World/OmxBaseLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)),
    )


@configclass
class OmxActionsCfg:
    joint_positions = OmxJointPositionActionCfg(
        asset_name="robot",
        joint_names=OMX_ARM_JOINTS,
        scale=1,
        use_default_offset=False,
    )


@configclass
class OmxObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 이름(joint_pos_obs)은 워크숍 스크립트(lerobot_eval/agent)가 쓰는 키와 같아야 한다.
        joint_pos_obs = ObsTerm(func=joint_pos, params={"asset_cfg": ARM6})
        joint_pos_rel = ObsTerm(func=joint_pos_rel, params={"asset_cfg": ARM6})
        ee_frame_state = ObsTerm(
            func=ee_frame_state,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class OmxEventCfg:
    # 리셋: 기본 자세(zero)로 복귀. gripper_joint_2 포함 전 관절.
    reset_robot_position = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )
    # 로봇 색 랜덤화는 SO-101 전용 머티리얼 경로(Looks/material_a_3d_printed)에 묶여 있어 OMX 에서는 생략.


@configclass
class OmxTeleopEnvCfg(ManagerBasedRLEnvCfg):
    scene: OmxBaseLitSceneCfg = OmxBaseLitSceneCfg()

    observations: OmxObservationsCfg = OmxObservationsCfg()
    actions: OmxActionsCfg = OmxActionsCfg()
    events: OmxEventCfg = OmxEventCfg()

    rewards = None  # teleop 용: 보상 없음
    terminations = None

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5

        self.scene.num_envs = 1
        self.viewer.eye = (-0.25, -0.4, 0.22)
        self.viewer.lookat = (0.15, 0.0, 0.12)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        self.sim.render.rendering_mode = "quality"
        self.sim.render.enable_translucency = False
