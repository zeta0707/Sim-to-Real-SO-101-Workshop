# SPDX-License-Identifier: Apache-2.0
#
# OMX-F 용 Vials-To-Rack 태스크 (+DR, +Eval). tasks/vials_to_rack_env_cfg.py 의 OMX 대응 신규 파일.
# 바이알/랙 에셋, 리셋/성공 판정(mdp), 관측/종료 조건은 워크숍 것을 그대로 재사용하고
# 로봇·접촉 센서·이벤트만 OMX 로 바꿨다.
#
# 접촉 센서: SO-101 은 jaw 한 개였지만 OMX 는 두 핑거(link6, link7)다.
#   워크숍의 파지 판정 term 은 force_matrix_w 를 body 축으로 합산하므로(dim=1 sum) 핑거 2개여도 그대로 동작한다.
import numpy as np

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat

from sim_to_real_so101.assets.omx import OMX_F_CONTACT_GRASP_CFG
from sim_to_real_so101.mdp import reset_vials_rack, randomize_sky_light, randomize_mat_rotation

from .omx_task_env_cfg import (
    OmxTaskEnvCfg,
    OmxTaskEventCfg,
    OmxTaskObservationsCfg,
    OmxTaskSceneCfg,
)
from .vials_to_rack_env_cfg import (
    VIAL_SPAWN_Z,
    VialsToRackObservationsCfg,
    VialsToRackTerminationsCfg,
    assets_path,
    rack,
    vial,
)

# OMX 의 수평 도달 거리는 약 0.4 m 이다. 바이알(x=0.23, 베이스 x=-0.05 기준 약 0.28 m)은 닿지만
# 여유가 크지 않으니 실제 배치에 맞춰 아래 좌표를 조정할 것.
FRICTION = 1.5  # 손가락/바이알 정/동 마찰계수

# 바이알 x 는 SO-101 의 0.23 에서 0.20 으로 당겼다 (팔을 덜 뻗고 잡도록). 거리가 파지 성공을 좌우한다는 확증은 없다:
# 당긴 뒤에도 수평 0.22~0.29 m 에서 3번 실패했다. reset 시 x 는 ±0.04 흔들린다.
@configclass
class OmxVialsToRackSceneCfg(OmxTaskSceneCfg):
    robot: ArticulationCfg = OMX_F_CONTACT_GRASP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    vial_1 = vial.replace()
    vial_1.prim_path = "{ENV_REGEX_NS}/Vial_1"
    vial_1.init_state.pos = (0.20, -0.08, VIAL_SPAWN_Z)
    vial_1.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    vial_2 = vial.replace()
    vial_2.prim_path = "{ENV_REGEX_NS}/Vial_2"
    vial_2.init_state.pos = (0.20, 0, VIAL_SPAWN_Z)
    vial_2.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    vial_3 = vial.replace()
    vial_3.prim_path = "{ENV_REGEX_NS}/Vial_3"
    vial_3.init_state.pos = (0.20, -0.16, VIAL_SPAWN_Z)
    vial_3.init_state.rot = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)

    rack_left = rack.replace()
    rack_left.prim_path = "{ENV_REGEX_NS}/Rack_Left"
    # SO-101 은 (0.18, 0.08). OMX 그리퍼는 손이 넓어서(좌우 약 ±5 cm) 바이알(y≈0)을 집을 때 랙 옆면에 닿았다.
    # 랙은 원점에서 +x/+y 로 0.12 m 뻗으므로 y 를 0.04 m, x 를 0.03 m 옮겨 손 폭만큼 간격을 확보한다.
    rack_left.init_state.pos = (0.15, 0.12, 0.06)

    # 두 핑거(link6, link7)에 접촉 센서. 파지 판정 term 이 sensor 이름 "contact_grasp" 을 참조한다.
    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link[67]",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Vial_1",
            "{ENV_REGEX_NS}/Vial_2",
            "{ENV_REGEX_NS}/Vial_3",
        ],
    )


@configclass
class OmxVialsToRackDRSceneCfg(OmxVialsToRackSceneCfg):
    sky_light = AssetBaseCfg(
        prim_path="/World/sky_light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{assets_path}/hdri/moon_lab_1k.exr",
            visible_in_primary_ray=False,
            enable_color_temperature=True,
            color_temperature=6500.0,
        ),
    )


@configclass
class OmxVialsToRackEventCfg(OmxTaskEventCfg):
    # 손가락 판 사이 V 자 홈에 바이알이 끼면 닫는 힘이 바이알을 손가락 끝 쪽으로 밀어낸다(접촉각 약 27 도, tan=0.5).
    # PhysX 기본 마찰(약 0.5, 두 물체 평균)로는 이를 못 버텨 들어올릴 때 빠져나간다. 손가락과 바이알 마찰을 높인다.
    # 서로 평균되므로 양쪽 다 올려야 한다. 실물 OMX 의 고무 패드 수준을 가정한 값이다.
    finger_friction = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link6", "link7"]),
            "static_friction_range": (FRICTION, FRICTION),
            "dynamic_friction_range": (FRICTION, FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    vial_1_friction = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("vial_1"),
            "static_friction_range": (FRICTION, FRICTION),
            "dynamic_friction_range": (FRICTION, FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    vial_2_friction = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("vial_2"),
            "static_friction_range": (FRICTION, FRICTION),
            "dynamic_friction_range": (FRICTION, FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    vial_3_friction = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("vial_3"),
            "static_friction_range": (FRICTION, FRICTION),
            "dynamic_friction_range": (FRICTION, FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    reset_vials_setup = EventTerm(
        func=reset_vials_rack,
        mode="reset",
        params={
            "vials": ["vial_1", "vial_2", "vial_3"],
            "rack": "rack_left",
            "rack_pose_range": {"x": (-0.04, 0.04), "y": (-0.01, 0.01), "yaw": (-0.5, 0.5)},
            "pose_range": {
                "x": (-0.04, 0.04),
                "y": (-0.01, 0.01),
                "roll": (-0.3, 0.3),
                "yaw": (0.0, 0.0),
            },
            "fixed_vial_z": 0.05,
        },
    )


@configclass
class OmxVialsToRackEventDRCfg(OmxVialsToRackEventCfg):
    reset_sky_light = EventTerm(
        func=randomize_sky_light,
        mode="reset",
        params={
            "exposure_range": (-4.0, 3.0),
            "temperature_range": (2500.0, 9500.0),
            "textures_root": f"{assets_path}/hdri",
            "asset_cfg": SceneEntityCfg("sky_light"),
        },
    )
    # OMX 에서는 DR 시 매트 회전 범위만 키운다 (로봇 색 랜덤화는 SO-101 머티리얼 경로 전용이라 제외)
    reset_mat_rotation = EventTerm(
        func=randomize_mat_rotation,
        mode="reset",
        params={"yaw_range": (-0.3, 0.3), "asset_cfg": SceneEntityCfg("mat")},
    )


@configclass
class OmxVialsToRackObservationsCfg(OmxTaskObservationsCfg):
    subtask_terms: VialsToRackObservationsCfg.SubtaskCfg = VialsToRackObservationsCfg.SubtaskCfg()


@configclass
class OmxVialsToRackEnvCfg(OmxTaskEnvCfg):
    scene: OmxVialsToRackSceneCfg = OmxVialsToRackSceneCfg()
    events: OmxVialsToRackEventCfg = OmxVialsToRackEventCfg()
    observations: OmxVialsToRackObservationsCfg = OmxVialsToRackObservationsCfg()


@configclass
class OmxVialsToRackDREnvCfg(OmxVialsToRackEnvCfg):
    scene: OmxVialsToRackDRSceneCfg = OmxVialsToRackDRSceneCfg()
    events: OmxVialsToRackEventDRCfg = OmxVialsToRackEventDRCfg()


@configclass
class OmxVialsToRackEvalEnvCfg(OmxVialsToRackEnvCfg):
    terminations: VialsToRackTerminationsCfg = VialsToRackTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 450 / 60.0


@configclass
class OmxVialsToRackEvalDREnvCfg(OmxVialsToRackDREnvCfg):
    terminations: VialsToRackTerminationsCfg = VialsToRackTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 450 / 60.0
