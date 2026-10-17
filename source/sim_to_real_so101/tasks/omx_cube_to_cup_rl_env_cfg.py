# SPDX-License-Identifier: Apache-2.0
#
# OMX-F Cube-To-Cup 강화학습 환경. 모방학습용 Cube-To-Cup 과 같은 물리/배치(큐브, 컵, 매트, 마찰)를 쓰되
#   - 카메라/라이트박스/렌더 설정을 빼고 (상태 기반 관찰, headless 병렬 학습용)
#   - 액션을 델타 관절 목표로 바꾸고
#   - 보상과 종료(성공/낙하/시간 초과)를 붙였다.
import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from sim_to_real_so101.mdp.omx_rl import (
    OmxDeltaJointPositionActionCfg,
    carry_to_cup,
    cube_lifted,
    cube_minus_cup_xy,
    cube_minus_ee,
    cube_out_of_bounds,
    cube_pos_env,
    cube_quat_env,
    ee_pointing_down,
    ee_pos_env,
    reach_cube,
    release_over_cup,
)
from sim_to_real_so101.mdp.omx_terms import cube_in_cup
from sim_to_real_so101.assets.omx import OMX_ARM_JOINTS

from .omx_cube_to_cup_env_cfg import CUBE_SIZE, CUBE_START, CUP_POS, FRICTION, MAT_TOP_Z
from .omx_env_cfg import ARM6, LerobotOmxBaseSceneCfg, OmxEventCfg
from .omx_task_env_cfg import assets_path

CUBE_REST_Z = MAT_TOP_Z + CUBE_SIZE / 2 + 0.002  # 매트 위에 놓인 큐브 중심 높이 (약 0.052)
FLOOR_COLOR = (0.82, 0.93, 0.82)  # 연한 초록. 흰색은 (0.95, 0.95, 0.95)
LIFT_Z = 0.075  # 이 높이 위로 올라오면 "들었다"고 본다 (success 의 z_max 와 같은 값)


@configclass
class OmxCubeToCupRLSceneCfg(LerobotOmxBaseSceneCfg):
    # 재생(GUI)용 시각 요소. 충돌/물리가 없어서 학습에는 영향이 없다.
    sky_light = AssetBaseCfg(
        prim_path="/World/OmxRLLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)),
    )
    # env 하나당 1.5 x 1.5 m 짜리 얇은 바닥판 (색은 FLOOR_COLOR 로 바꾼다)
    floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.5, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=FLOOR_COLOR),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.0, -0.002)),
    )

    # 매트는 큐브가 놓이는 바닥이다 (조명/카메라 없음).
    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{assets_path}/usd/mat.usda"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.22, 0, 0.032), rot=(0.7071068, 0.0, 0.0, 0.7071068)),
    )

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(CUBE_START[0], CUBE_START[1], CUBE_REST_Z)),
    )

    cup = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{assets_path}/usd/cup.usda"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(CUP_POS[0], CUP_POS[1], MAT_TOP_Z)),
    )


@configclass
class OmxRLActionsCfg:
    joint_positions = OmxDeltaJointPositionActionCfg(
        asset_name="robot",
        joint_names=OMX_ARM_JOINTS,
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class OmxRLObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, params={"asset_cfg": ARM6})  # 6
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, params={"asset_cfg": ARM6})  # 6
        ee_pos = ObsTerm(func=ee_pos_env)  # 3
        cube_pos = ObsTerm(func=cube_pos_env)  # 3
        cube_quat = ObsTerm(func=cube_quat_env)  # 4
        cube_minus_ee = ObsTerm(func=cube_minus_ee)  # 3
        cube_minus_cup = ObsTerm(func=cube_minus_cup_xy, params={"cup_xy": CUP_POS})  # 2
        last_action = ObsTerm(func=isaac_mdp.last_action)  # 6

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class OmxRLEventCfg(OmxEventCfg):
    # 모방학습 환경과 같은 손가락 마찰, 같은 큐브 초기 위치 분포
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
class OmxRLRewardsCfg:
    # 단계: 큐브에 다가가기 -> 들기 -> 컵 위로 옮기기 -> 컵에 넣기(성공)
    # 시작 자세에서 ee 와 큐브는 약 0.3 m 떨어져 있어서, 먼 거리용(0.2)과 가까운 거리용(0.05) 두 항으로 나눴다.
    reach = RewTerm(func=reach_cube, weight=1.0, params={"std": 0.2})
    reach_fine = RewTerm(func=reach_cube, weight=2.0, params={"std": 0.05})
    # 성공하면 에피소드가 끝나므로, 성공 보상이 "계속 들고 있을 때 앞으로 받을 형성 보상 합"보다 훨씬 커야 한다.
    # (gamma 0.98 -> 유효 지평 약 50 스텝. 형성 보상 합 13/스텝 x 50 = 650 < 성공 1000)
    # 그리퍼가 아래를 향하도록 (수평 0 ~ 수직 아래 1). 성공 보상(1000) 에 비해 아주 작게 둔다.
    wrist_down = RewTerm(func=ee_pointing_down, weight=8.0)
    lift = RewTerm(func=cube_lifted, weight=4.0, params={"z_min": LIFT_Z})
    carry = RewTerm(func=carry_to_cup, weight=6.0, params={"std": 0.1, "z_min": LIFT_Z, "cup_xy": CUP_POS})
    carry_fine = RewTerm(func=carry_to_cup, weight=3.0, params={"std": 0.03, "z_min": LIFT_Z, "cup_xy": CUP_POS})
    # 컵 위에서 그리퍼를 쥔 폭 이상으로 열면 보상 (gripper_joint_1 범위 0~1.745 rad). 들고만 있는 정책이 놓기를 찾도록 돕는다.
    release = RewTerm(
        func=release_over_cup,
        weight=8.0,
        params={
            "z_min": LIFT_Z,
            "cup_xy": CUP_POS,
            "xy_tol": 0.03,
            # 실측(omx_rl_play.py --probe): 큐브를 쥐고 있을 때 gripper_joint_1 = 0.001~0.002 rad (관절 한계 근처).
            # 그래서 조금만 열려도 보상이 나오게 0.1 rad 부터, 0.7 rad 에서 포화시킨다.
            "open_min": 0.1,
            "open_max": 0.7,
            # 관절 id 는 params 에 들어 있는 SceneEntityCfg 만 매니저가 풀어 준다 (기본 인자는 안 풀림)
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint_1"]),
        },
    )
    success = RewTerm(func=isaac_mdp.is_terminated_term, weight=1000.0, params={"term_keys": "success"})
    cube_lost = RewTerm(func=isaac_mdp.is_terminated_term, weight=-10.0, params={"term_keys": "cube_out"})
    action_rate = RewTerm(func=isaac_mdp.action_rate_l2, weight=-1.0e-3)
    joint_vel = RewTerm(func=isaac_mdp.joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": ARM6})


@configclass
class OmxRLTerminationsCfg:
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    # 성공 판정은 모방학습 평가(Cube-To-Cup-Eval)와 같은 함수/파라미터다.
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
    cube_out = DoneTerm(func=cube_out_of_bounds, time_out=False, params={"z_min": 0.005, "max_radius": 0.6})


@configclass
class OmxCubeToCupRLEnvCfg(ManagerBasedRLEnvCfg):
    scene: OmxCubeToCupRLSceneCfg = OmxCubeToCupRLSceneCfg()
    observations: OmxRLObservationsCfg = OmxRLObservationsCfg()
    actions: OmxRLActionsCfg = OmxRLActionsCfg()
    events: OmxRLEventCfg = OmxRLEventCfg()
    rewards: OmxRLRewardsCfg = OmxRLRewardsCfg()
    terminations: OmxRLTerminationsCfg = OmxRLTerminationsCfg()

    def __post_init__(self) -> None:
        # 물리 120 Hz, 정책 30 Hz (모방학습 데이터의 30 fps 와 같다)
        self.sim.dt = 1 / 120
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0

        self.scene.num_envs = 512
        self.scene.env_spacing = 1.5

        self.viewer.eye = (-0.25, -0.4, 0.22)
        self.viewer.lookat = (0.15, 0.0, 0.12)
