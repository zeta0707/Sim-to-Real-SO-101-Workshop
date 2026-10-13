# OMX 용 LeRobotRecorder. 워크샵 원본(utils/lerobot_recorder.py)은 건드리지 않고 버퍼 처리만 바꾼다.
#
# 원본 문제: 첫 프레임에서 2분(3600프레임) 분량 RGB 버퍼를 GPU 에 미리 할당한다.
#   640x480x3 x 3600 = 카메라당 3.3GB -> 카메라 2대 6.6GB. 8GB VRAM 노트북에서는 S 를 누르는 순간 멈춘다.
# 여기서는 프레임을 CPU(numpy)에 들어오는 만큼만 쌓는다(미리 할당 없음, 길이 제한 없음).
import numpy as np
import torch

from .lerobot_recorder import LeRobotRecorder


class OmxLeRobotRecorder(LeRobotRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.save_mp4 or self.depth or self.instance_id_seg:
            print("[OMX][WARN] OmxLeRobotRecorder 는 RGB(LeRobotDataset)만 저장한다. --save_mp4/--depth/--instance_id_seg 는 무시된다.")
        self.save_mp4 = False
        self.depth = False
        self.instance_id_seg = False
        self._reset_buffers()

    def _reset_buffers(self):
        self._actions = []
        self._obs = []
        self._rgb = {name: [] for name in self.cameras.keys()}
        self.current_frame = 0

    def push_frame_to_buffer(self, action, observation, visual_buffers, depth_buffers, instance_id_seg_buffers):
        self._actions.append(action.detach().cpu().numpy().astype(np.float32).copy())
        self._obs.append(observation.detach().cpu().numpy().astype(np.float32).copy())
        for name in self.cameras.keys():
            frame = visual_buffers[name]
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().to(torch.uint8).cpu().numpy()
            self._rgb[name].append(np.ascontiguousarray(frame))
        self.current_frame += 1

    def save_episode(self, event):
        if event.event_name != self.STOP_RECORDING_EVENT:
            return
        if self.current_frame == 0:
            print("[INFO]: 녹화된 프레임이 없어 에피소드를 저장하지 않는다.")
            return
        print(f"[INFO]: Episode {self.current_frame} frames -> queue")
        self.episode_queue.put(
            {
                "action_buffers": self._actions,
                "observation_buffer_tensor": self._obs,
                "rgb_buffer_tensors": self._rgb,
                "total_frames": self.current_frame,
            }
        )
        self._reset_buffers()
        print("[INFO]: Cleared buffers")

    def cancel_recording(self, event):
        if event.event_name == self.CANCEL_RECORDING_EVENT:
            self._reset_buffers()
            print("[INFO]: Episode cancelled. Cleared buffers")
