# SPDX-License-Identifier: Apache-2.0
#
# OMX-F 용 Cube-To-Cup 태스크: 매트 위의 큐브 하나를 집어 컵에 넣는다.
# 바이알/랙 대신 큐브(절차 생성)와 컵(assets/usd/cup.usda, 윗면이 열린 정사각 용기)을 쓰는 단순한 태스크다.
# 라이트박스/카메라/렌더 설정은 omx_task_env_cfg 의 것을 그대로 쓴다.
import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from sim_to_real_so101.mdp.omx_terms import cube_in_cup

from .omx_task_env_cfg import (
    OmxTaskEnvCfg,
    OmxTaskEventCfg,
    OmxTaskObservationsCfg,
    OmxTaskSceneCfg,
    assets_path,
)

MAT_TOP_Z = 0.035  # 매트 윗면 (mat.usda 가 z=0.032 에 있고 두께 ±0.003)
CUBE_SIZE = 0.03
CUBE_START = (0.22, -0.04)  # (x, y), 어깨 기준 수평 거리 약 0.23 m
CUP_POS = (0.17, 0.10)  # 컵 중앙 (x, y). 컵은 y=0.054~0.146 이라 큐브(y≈-0.04)를 집는 손(좌우 ±5 cm)과 겹치지 않는다.
FRICTION = 1.5  # 손가락/큐브 정/동 마찰계수


@configclass
class OmxCubeToCupSceneCfg(OmxTaskSceneCfg):
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.10)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=FRICTION, dynamic_friction=FRICTION),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(CUBE_START[0], CUBE_START[1], MAT_TOP_Z + CUBE_SIZE / 2 + 0.002)),
    )

    # 정적 콜라이더만 있는 USD (강체 아님). 원점 = 컵 바닥 아래면 중앙.
    cup = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{assets_path}/usd/cup.usda"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(CUP_POS[0], CUP_POS[1], MAT_TOP_Z)),
    )


@configclass
class OmxCubeToCupEventCfg(OmxTaskEventCfg):
    # 손가락 마찰을 올린다 (큐브 쪽은 스폰 설정에 이미 들어 있다). 마찰은 두 물체 평균으로 적용된다.
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

    reset_cube = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.5, 0.5)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class OmxCubeToCupTerminationsCfg:
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)

    success = DoneTerm(
        func=cube_in_cup,
        time_out=False,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "cup_xy": CUP_POS,
            "inner_half": 0.035,
            "z_min": 0.04,
            "z_max": 0.075,
        },
    )


@configclass
class OmxCubeToCupEnvCfg(OmxTaskEnvCfg):
    scene: OmxCubeToCupSceneCfg = OmxCubeToCupSceneCfg()
    events: OmxCubeToCupEventCfg = OmxCubeToCupEventCfg()
    observations: OmxTaskObservationsCfg = OmxTaskObservationsCfg()


@configclass
class OmxCubeToCupEvalEnvCfg(OmxCubeToCupEnvCfg):
    terminations: OmxCubeToCupTerminationsCfg = OmxCubeToCupTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # 시연 20개의 평균 길이가 약 17.4 s(최대 28 s)라서 20 s 는 정책에게 빠듯하다.
        self.episode_length_s = 30.0
