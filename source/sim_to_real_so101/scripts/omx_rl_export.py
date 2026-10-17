# SPDX-License-Identifier: Apache-2.0
#
# RSL-RL 체크포인트(.pt) 에서 actor 가중치와 관측 정규화 통계를 꺼내 .npz 로 저장하고,
# 시뮬레이터에서 기록한 (관측, 행동) 기준값과 비교해 같은 출력을 내는지 확인한다.
# Isaac Sim 없이 일반 python 으로 돌아간다 (torch, numpy 만 필요).
#
#   1) 기준값 기록 (Isaac 컨테이너):
#      python source/sim_to_real_so101/scripts/omx_rl_play.py --headless --num_envs 16 --num_episodes 20 \
#        --checkpoint <model.pt> --dump_ref outputs/rl/ref_r10.npz
#   2) 내보내기 + 검증:
#      python source/sim_to_real_so101/scripts/omx_rl_export.py --checkpoint <model.pt> \
#        --out outputs/rl/omx_policy_r10.npz --ref outputs/rl/ref_r10.npz
import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omx_policy_np import OBS_DIM, OmxPolicy  # noqa: E402

DELTA_SCALE = (0.05, 0.05, 0.05, 0.05, 0.05, 0.08)  # mdp/omx_rl.py: OmxDeltaJointPositionActionCfg.delta_scale

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--ref", default=None, help="omx_rl_play.py --dump_ref 로 만든 기준값")
args = parser.parse_args()

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
print("[export] 체크포인트 키:", list(ck.keys()))
sd = ck["model_state_dict"]

# --- actor 선형층 찾기: 'actor.0.weight', 'actor.2.weight', ... (또는 'actor.mlp.0.weight')
layers = {}
for k, v in sd.items():
    m = re.match(r"^(actor(?:\.mlp)?)\.(\d+)\.(weight|bias)$", k)
    if m and "critic" not in k:
        layers.setdefault(int(m.group(2)), {})[m.group(3)] = v.double().numpy()
if not layers:
    print("[export][오류] actor 층을 찾지 못했다. model_state_dict 키:")
    for k, v in sd.items():
        print("   ", k, tuple(v.shape))
    sys.exit(1)
idx = sorted(layers)
W = [layers[i]["weight"] for i in idx]
b = [layers[i]["bias"] for i in idx]
print("[export] actor 층 크기:", [w.shape for w in W])
assert W[0].shape[1] == OBS_DIM, f"입력 차원 {W[0].shape[1]} != {OBS_DIM}"
assert W[-1].shape[0] == 6, f"출력 차원 {W[-1].shape[0]} != 6"

# --- 관측 정규화 통계
norm = None
for key in ("obs_norm_state_dict", "actor_obs_norm_state_dict"):
    if key in ck and ck[key]:
        norm = ck[key]
        break
if norm is None:
    cand = {k: v for k, v in sd.items() if "normalizer" in k or "obs_norm" in k}
    norm = {k.split(".")[-1]: v for k, v in cand.items()} if cand else None
if norm is None:
    print("[export][경고] 정규화 통계를 찾지 못했다 -> 항등 변환(mean 0, std 1)으로 저장. 학습 때 정규화를 켰다면 결과가 틀린다.")
    mean, std = np.zeros(OBS_DIM), np.ones(OBS_DIM)
else:
    print("[export] 정규화 키:", {k: tuple(v.shape) for k, v in norm.items() if hasattr(v, "shape")})
    g = lambda *names: next((norm[n] for n in names if n in norm), None)
    mean = g("_mean", "running_mean", "mean")
    std = g("_std", "running_std", "std")
    if std is None:
        var = g("_var", "running_var", "var")
        std = torch.sqrt(var) if var is not None else None
    if mean is None or std is None:
        print("[export][오류] 정규화 mean/std 키를 몰라서 중단. 위 키 목록을 알려 달라.")
        sys.exit(1)
    mean = mean.double().numpy().reshape(-1)
    std = std.double().numpy().reshape(-1)
EPS = 1e-2  # rsl_rl EmpiricalNormalization 기본값: (x - mean) / (std + eps)

save = {f"W{i}": W[i] for i in range(len(W))}
save.update({f"b{i}": b[i] for i in range(len(b))})
save.update(
    n_layers=len(W), obs_mean=mean, obs_std=std, obs_eps=EPS, delta_scale=np.array(DELTA_SCALE),
    source=os.path.abspath(args.checkpoint),
)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, **save)
print(f"[export] 저장: {args.out}")

# --- 검증
if args.ref:
    ref = np.load(args.ref)
    obs, act = ref["obs"], ref["act"]
    pol = OmxPolicy(args.out)
    out = np.stack([pol.raw(o) for o in obs])
    err = np.abs(out - act)
    print(f"[export] 기준 샘플 {len(obs)} 개: 최대 오차 {err.max():.2e}, 평균 오차 {err.mean():.2e}")
    for j, name in enumerate(["joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]):
        print(f"           {name:8s} 최대 오차 {err[:, j].max():.2e}")
    ok = err.max() < 1e-3
    print("[export] 검증", "통과" if ok else "실패 (EPS 값/정규화 키를 확인)")
    sys.exit(0 if ok else 2)
else:
    print("[export] --ref 가 없어 검증은 생략했다.")
