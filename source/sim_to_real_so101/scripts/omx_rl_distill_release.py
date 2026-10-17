# SPDX-License-Identifier: Apache-2.0
#
# '놓기' 규칙(컵 위에서 --release_dwell 스텝 머물면 그리퍼 열기)을 정책에 가르친다 (DAgger 식 지도학습).
#   - 정책(결정론)+규칙으로 굴려 (관측, 목표행동)을 모은다.
#       규칙이 발동한 뒤 상태: 목표 그리퍼 행동 = +1 (열기)
#       그 외 상태: 목표 = 정책 자신의 현재 출력 (다른 동작이 망가지지 않게 붙잡아 두는 앵커)
#   - actor 평균을 MSE 로 맞춘 뒤 다시 굴리는 과정을 --rounds 번 반복하고 새 체크포인트로 저장한다.
#   - 끝난 뒤 --resume 으로 omx_rl_train.py 를 이어서 돌리면 PPO 가 다듬는다.
#
#   python source/sim_to_real_so101/scripts/omx_rl_distill_release.py --headless \
#     --checkpoint outputs/rl/omx_cube_to_cup/<run>/model_2100.pt --out outputs/rl/omx_cube_to_cup/distill_release.pt
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Distill the release rule into the PPO actor.")
parser.add_argument("--task", type=str, default="Lerobot-Omx-Cube-To-Cup-RL")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--steps", type=int, default=360, help="라운드마다 굴리는 스텝 수 (30 Hz, 360 = 한 에피소드)")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--release_tol", type=float, default=0.03)
parser.add_argument("--release_dwell", type=int, default=15)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from sim_to_real_so101.tasks.omx_cube_to_cup_env_cfg import CUP_POS

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.tasks.omx_rl_agent_cfg import OmxCubeToCupPPORunnerCfg


def actor_mean_fn(runner):
    """(관측 -> 결정론 행동) 함수. 기울기가 흐르는 버전. rsl_rl 버전 차이를 흡수한다."""
    alg = runner.alg
    pol = getattr(alg, "policy", None) or getattr(alg, "actor_critic", None)
    norm = getattr(runner, "obs_normalizer", None)
    if pol is not None and hasattr(pol, "act_inference"):
        if norm is not None:
            return lambda o: pol.act_inference(norm(o)), pol
        return lambda o: pol.act_inference(o), pol
    actor = getattr(alg, "actor", None)  # rsl_rl >= 3
    return (lambda o: actor(o)), actor


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = OmxCubeToCupPPORunnerCfg()
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    mean_fn, module = actor_mean_fn(runner)
    params = [p for n, p in module.named_parameters() if "critic" not in n and "log_std" not in n and n.split(".")[-1] not in ("std", "std_param")]
    opt = torch.optim.Adam(params, lr=args_cli.lr)

    uenv = env.unwrapped
    dev = uenv.device
    n = uenv.num_envs
    cube = uenv.scene["cube"]
    origins = uenv.scene.env_origins
    cup_t = torch.tensor(CUP_POS, device=dev)

    def get_obs():
        o = env.get_observations()
        return o[0] if isinstance(o, tuple) else o

    for rnd in range(args_cli.rounds):
        obs_buf, tgt_buf, rel_buf = [], [], []
        obs = get_obs()
        dwell = torch.zeros(n, dtype=torch.long, device=dev)
        released = torch.zeros(n, dtype=torch.bool, device=dev)
        succ = eps = 0
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                a = mean_fn(obs)
                cpos = cube.data.root_pos_w - origins
                cd = torch.norm(cpos[:, :2] - cup_t, dim=1)
                ok = (cpos[:, 2] > 0.075) & (cd < args_cli.release_tol)
                dwell = torch.where(ok, dwell + 1, torch.zeros_like(dwell))
                released |= dwell >= args_cli.release_dwell
                tgt = a.clone()
                tgt[released, 5] = 1.0
                obs_buf.append(obs.clone())
                tgt_buf.append(tgt)
                rel_buf.append(released.clone())
                obs, _, dones, _ = env.step(tgt)
                if dones.any():
                    m = dones.bool()
                    eps += int(m.sum())
                    succ += int(uenv.termination_manager.get_term("success")[m].sum())
                    dwell[m] = 0
                    released[m] = False
        X = torch.cat(obs_buf).clone()
        T = torch.cat(tgt_buf).clone()
        R = torch.cat(rel_buf).clone()
        print(f"[distill] round {rnd}: 샘플 {len(X)}, 놓기 샘플 {int(R.sum())}, 규칙+정책 성공 {succ}/{eps}")
        w = torch.ones(len(X), 6, device=dev)
        w[R, 5] = 20.0  # 놓기 상태의 그리퍼 출력에 가중
        for ep in range(args_cli.epochs):
            perm = torch.randperm(len(X), device=dev)
            tot = 0.0
            for i in range(0, len(X), 4096):
                idx = perm[i : i + 4096]
                pred = mean_fn(X[idx])
                loss = (w[idx] * (pred - T[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            if ep % 10 == 0 or ep == args_cli.epochs - 1:
                with torch.no_grad():
                    g = mean_fn(X[R])[:, 5].mean() if R.any() else float("nan")
                print(f"[distill]   epoch {ep}: loss {tot/len(X):.5f}  놓기 상태 그리퍼 평균 출력 {float(g):.3f}")
    # log_dir=None 으로 만든 runner 는 save() 가 쓰는 속성이 없을 수 있다
    for k, v in (("logger_type", "tensorboard"), ("disable_logs", True)):
        if not hasattr(runner, k):
            setattr(runner, k, v)
    runner.save(args_cli.out)
    print(f"[distill] 저장: {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
