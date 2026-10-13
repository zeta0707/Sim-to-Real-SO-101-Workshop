# SPDX-License-Identifier: Apache-2.0
#
# OMX-F 태스크 베이스 환경 (조명 박스 + 매트 + 카메라 + 도메인 랜덤화).
# tasks/task_env_cfg.py 의 OMX 대응 신규 파일. 라이트박스/매트/외부 카메라(D455)/렌더 설정은
# 워크숍 에셋을 그대로 재사용하고, 로봇 의존 부분(ego 카메라 마운트, 이벤트)만 OMX 로 바꿨다.
import os

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from sim_to_real_so101 import assets
from sim_to_real_so101.mdp import (
    randomize_camera_focal_length,
    randomize_camera_pose,
    randomize_light_exposure,
    randomize_mat_rotation,
)

from .omx_env_cfg import (
    LerobotOmxBaseSceneCfg,
    OmxEventCfg,
    OmxObservationsCfg,
    OmxTeleopEnvCfg,
)
from .task_env_cfg import TaskObservationsCfg, camera_object

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

# ---------------------------------------------------------------------------
# ego(손목) 카메라 마운트 -- 초기값이다. 실제 OMX 에 단 카메라 위치에 맞춰 조정할 것.
#   부모: link5 (손목 롤 링크, +x = 그리퍼가 향하는 방향, end_effector_link 는 x=0.092 m)
#   pos : link5 좌표계 기준 (x 앞, y 좌, z 위)
#   rot : opengl 카메라 규약(-Z 가 시선, +Y 가 위). 시선 = link5 +x, 이미지 위쪽 = link5 +z
#         -> 쿼터니언 (w, x, y, z) = (0.5, 0.5, -0.5, -0.5)
# ---------------------------------------------------------------------------
EGO_CAM_POS = (0.04, 0.0, 0.035)
EGO_CAM_ROT_WXYZ = (0.5, 0.5, -0.5, -0.5)


@configclass
class OmxTaskSceneCfg(LerobotOmxBaseSceneCfg):
    lightstudio = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LightStudio",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/lightbox-simple.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.1, 0, 0.0257)),
    )
    lightbox_light = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/LightStudio/LightBox/RectLight")

    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{assets_path}/usd/mat.usda"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.22, 0, 0.032),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),  # yaw 90°
        ),
    )

    # Cameras (이름이 camera_ 로 시작해야 agent/eval 스크립트가 자동으로 찾는다)
    camera_ego = camera_object.replace()
    camera_ego.prim_path = "{ENV_REGEX_NS}/Robot/link5/gripper_cam"
    # offset 은 새 객체로 교체한다(공유된 camera_object.offset 을 건드리지 않기 위함)
    camera_ego.offset = TiledCameraCfg.OffsetCfg(pos=EGO_CAM_POS, rot=EGO_CAM_ROT_WXYZ, convention="opengl")

    camera_external_D455 = camera_object.replace()
    camera_external_D455.prim_path = (
        "{ENV_REGEX_NS}/LightStudio/LightBox/camera_mount/rsd455/RSD455/Camera_OmniVision_OV9782_Right"
    )
    camera_external_D455.spawn = None


@configclass
class OmxTaskEventCfg(OmxEventCfg):
    reset_lightbox_light_exposure = EventTerm(
        func=randomize_light_exposure,
        mode="reset",
        params={"exposure_range": (-3.0, 1.0), "asset_cfg": SceneEntityCfg("lightbox_light")},
    )
    reset_mat_rotation = EventTerm(
        func=randomize_mat_rotation,
        mode="reset",
        params={"yaw_range": (-0.1, 0.1), "asset_cfg": SceneEntityCfg("mat")},
    )
    reset_camera_ego_fov = EventTerm(
        func=randomize_camera_focal_length,
        mode="reset",
        params={"focal_length_range": (12.0, 15.0), "asset_cfg": SceneEntityCfg("camera_ego")},
    )
    reset_camera_external_pose = EventTerm(
        func=randomize_camera_pose,
        mode="reset",
        params={
            "prim_path_pattern": "{ENV_REGEX_NS}/LightStudio/LightBox/camera_mount",
            "pos_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)},
            "rot_range": {"roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05)},
        },
    )


@configclass
class OmxTaskObservationsCfg(OmxObservationsCfg):
    # 워크숍과 동일한 visual 그룹(rgb/depth/seg x ego/external)을 그대로 재사용
    visual: TaskObservationsCfg.VisualCfg = TaskObservationsCfg.VisualCfg()


@configclass
class OmxTaskEnvCfg(OmxTeleopEnvCfg):
    scene: OmxTaskSceneCfg = OmxTaskSceneCfg()
    events: OmxTaskEventCfg = OmxTaskEventCfg()
    observations: OmxTaskObservationsCfg = OmxTaskObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.sim.render.enable_translucency = True
        self.sim.render.carb_settings = {
            "rtx.reflections.enabled": True,
            "rtx.translucency.reflectAtAllBounce": True,
            "rtx.translucency.sampleRoughness": True,
            "rtx.translucency.reflectionThroughputThreshold": 0.05,
            "rtx.translucency.maxRefractionBounces": 5,
            "rtx.raytracing.fractionalCutoutOpacity": True,
        }
