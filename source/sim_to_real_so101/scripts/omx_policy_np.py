# SPDX-License-Identifier: Apache-2.0
#
# 학습된 RSL-RL actor 를 numpy 만으로 실행한다 (Isaac Sim/torch 없이). 실제 로봇 제어 루프에서 그대로 import 해서 쓴다.
#
#   from omx_policy_np import OmxPolicy
#   policy = OmxPolicy("outputs/rl/omx_policy_r10.npz")
#   action = policy(obs33)          # shape (6,), 이미 [-1, 1] 로 clip 됨
#
# 관측 33차원의 순서 (mdp/omx_rl.py, omx_cube_to_cup_rl_env_cfg.py 와 같다):
#   0:6   joint_pos       arm 5 + gripper 1 (rad)
#   6:12  joint_vel       (rad/s)
#   12:15 ee_pos          로봇 베이스 기준 (m), link5 에서 (0.09193, -0.0016, 0) 오프셋
#   15:18 cube_pos        로봇 베이스 기준 (m)
#   18:22 cube_quat       (w, x, y, z)
#   22:25 cube_minus_ee   cube_pos - ee_pos
#   25:27 cube_minus_cup  cube xy - cup xy
#   27:33 last_action     직전에 낸 행동 6개 ([-1,1], clip 후)
import numpy as np

OBS_DIM = 33
ACT_DIM = 6


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


class OmxPolicy:
    def __init__(self, path):
        d = np.load(path)
        self.n_layers = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float64) for i in range(self.n_layers)]
        self.b = [d[f"b{i}"].astype(np.float64) for i in range(self.n_layers)]
        self.mean = d["obs_mean"].astype(np.float64)
        self.std = d["obs_std"].astype(np.float64)
        self.eps = float(d["obs_eps"])
        self.delta_scale = d["delta_scale"].astype(np.float64)

    def raw(self, obs):
        """clip 하기 전의 actor 출력 (검증용)."""
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / (self.std + self.eps)
        for i in range(self.n_layers):
            x = x @ self.W[i].T + self.b[i]
            if i < self.n_layers - 1:
                x = _elu(x)
        return x

    def __call__(self, obs):
        return np.clip(self.raw(obs), -1.0, 1.0)
