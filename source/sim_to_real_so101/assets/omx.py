# SPDX-License-Identifier: Apache-2.0
#
# ROBOTIS OMX-F (OpenMANIPULATOR-X follower) Isaac Lab articulation 설정.
# Sim-to-Real-SO-101-Workshop 의 assets/so101.py 에 대응하는 신규 파일 (기존 파일은 수정하지 않음).
#
# 에셋: usd/omx_f/omx_f_isaaclab.usda  ->  원본 omx_f.usd(Pseudo-Lab/NVIDIA-RFM-Lab 의 Assets/omx_f) 위에
#       오버라이드 레이어만 얹은 파일이다. 오버라이드 내용:
#         - gripper_joint_2 의 PhysxMimicJointAPI 제거 + joint_1 과 같은 DriveAPI 부여
#           (Pseudo-Lab teleop 스크립트가 런타임에 하던 처리를 USD 에 미리 반영. 종동 핑거는 액션 term 이 -1 로 구동)
#         - 에셋 내장 _dome 라이트 비활성화
#
# 게인: stiffness/damping/effort 를 None 으로 두면 USD 의 값(stiffness 17.8, damping 0.6, maxForce 1000)을
#       그대로 쓴다. Pseudo-Lab 팀이 teleop 에서 검증한 값이다. 학습/배포에서 움직임이 너무 "딱딱"하면
#       아래 actuators 의 stiffness/damping 에 값을 직접 넣어 낮추면 된다.
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

here = os.path.dirname(os.path.abspath(__file__))

# 제어 대상 6축 (leader 의 6축과 1:1) + 종동 핑거 1축
OMX_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper_joint_1"]
OMX_MIRROR_JOINT = "gripper_joint_2"

# 베이스 배치. SO-101 워크숍은 (-0.05, 0, 0) + yaw 90° 였다. OMX 는 zero 자세에서 +x 쪽으로 뻗는다고
# 가정해 yaw 0 으로 둔다. 방향이 틀리면 ROBOT_YAW_DEG 만 바꾸면 된다.
# x 를 SO-101 의 -0.05 에서 0.0 으로 옮겼다: OMX 어깨 축은 베이스 원점에서 x=-0.011, 높이 0.0975 m 이고
# 바이알(x=0.23, y=-0.16~0)까지 어깨 기준 수평 거리가 0.29~0.33 m 라서, 위에서 집는 자세(약 0.30 m 한계)에는
# 빠듯했다. 0.0 으로 옮기면 0.24~0.29 m 가 되어 여유가 생긴다.
ROBOT_POS = (0.0, 0.0, 0.0)
ROBOT_YAW_DEG = 0.0


def _yaw_quat(yaw_deg: float):
    import math

    h = math.radians(yaw_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))  # (w, x, y, z)


OMX_F_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{here}/usd/omx_f/omx_f_isaaclab.usda",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        # fix_root_link 는 지정하지 않는다: 에셋의 root_joint(FixedJoint, world 고정)가 이미 베이스를 고정한다.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # zero 자세 = 리더 암 zero-pose homing 기준 자세 (Pseudo-Lab 문서의 "sim 초기 자세")
        joint_pos={".*": 0.0},
        pos=ROBOT_POS,
        rot=_yaw_quat(ROBOT_YAW_DEG),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-5]"],
            stiffness=None,  # None = USD 값 사용 (17.8). 80 으로 높여 봤지만 파지 시 손이 더 밀려 올라가서 되돌렸다.
            damping=None,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint_[12]"],
            stiffness=None,
            damping=None,
        ),
    },
)

# 접촉 센서(파지 판정)용: 핑거 link6 / link7 에 contact report 활성화
OMX_F_CONTACT_GRASP_CFG = OMX_F_CFG.copy()
OMX_F_CONTACT_GRASP_CFG.spawn.activate_contact_sensors = True
