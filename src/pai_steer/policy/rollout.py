"""評価ロールアウト (§6, フック差し込み口あり).

steer: None か, `.enabled` を持つ介入オブジェクト (steer/hooks.py). 推論ごとに gate_fn(info) で切り替える.
info はシミュレータの真の状態から作る推定フェーズ:
  t, carrying (グリッパー閉 かつ キューブが机から 2cm 以上浮いている), ever_lifted, released (運搬後に離した),
  high (運搬中かつキューブが箱の壁の上端 + 6cm より高い = 下ろし始める前), phase ∈ {"pre", "carry", "post"}
"""
import numpy as np
import torch

from pai_steer.common import CKPT, COLORS, base_parser, load_config
from pai_steer.data.dataset import obs_to_tokens
from pai_steer.env.cube_sort_env import make_env


class Policy:
    """正規化と z=0 推論をまとめたラッパー. 戻り値は元スケールの行動チャンク [K, 7]."""

    def __init__(self, model, stats, device="cpu"):
        self.m, self.st, self.dev = model, stats, device

    @torch.no_grad()
    def __call__(self, robot, cube, color):
        st = self.st
        f = lambda x: torch.tensor(x, dtype=torch.float32, device=self.dev)[None]
        a, _, _ = self.m(f((robot - st["robot_mean"]) / st["robot_std"]), f((cube - st["cube_mean"]) / st["cube_std"]),
                         f(color))
        return a[0].cpu().numpy() * st["act_std"] + st["act_mean"]


def rollout(policy, env, color, seed, cfg, steer=None, gate_fn=None, horizon=None, color_input=None,
            record_states=False):
    """1エピソード. color_input を与えると color トークンだけ別の色にできる (デバッグ用)."""
    K_exec = cfg["act"]["exec_steps"]
    horizon = horizon or cfg["env"]["horizon"]
    lift_thr = env.layout.table_top + env.layout.cube_size / 2 + cfg["labels"]["lift_threshold"]
    env.cube_color = color
    env.set_seed(seed)
    obs = env.reset()
    col_vec = np.eye(4)[COLORS.index(color_input or color)]
    rec = dict(eef=[], cube=[], phase=[], steer=[], action=[], states=[])
    ever_lifted = released = False
    settled, chunk, ci = 0, None, K_exec
    for t in range(horizon):
        cp = env.cube_pos
        carrying = (not env.gripper_open()) and cp[2] > lift_thr
        if carrying:
            ever_lifted = True
        elif ever_lifted and env.gripper_open():
            released = True
        phase = "carry" if carrying else ("post" if released else "pre")
        high = carrying and cp[2] > env.layout.wall_top + 0.06
        info = dict(t=t, carrying=carrying, ever_lifted=ever_lifted, released=released, high=high, phase=phase)
        if ci >= K_exec:
            if steer is not None:
                steer.enabled = bool(gate_fn(info)) if gate_fn else True
            r, c, _ = obs_to_tokens(obs)
            chunk = policy(r, c, col_vec)
            ci = 0
        a = np.clip(chunk[ci], -1, 1)
        ci += 1
        rec["eef"].append(env.eef_pos)
        rec["cube"].append(cp)
        rec["phase"].append(phase)
        rec["steer"].append(bool(steer is not None and steer.enabled))
        rec["action"].append(a)
        if record_states:
            rec["states"].append(env.get_state())
        obs, _, _, _ = env.step(a)
        # 早期終了: 離した後, キューブがどこかに静止 (10 ステップ)
        if released and np.linalg.norm(env.cube_pos - cp) < 1e-3 and env.eef_pos[2] > lift_thr + 0.05:
            settled += 1
            if settled >= 10:
                break
        else:
            settled = 0
    if steer is not None:
        steer.enabled = False
    out = {k: np.array(v) for k, v in rec.items()}
    if record_states:
        out["final_state"] = env.get_state()
    out.update(placed_box=env.placed_box(), success=bool(env._check_success()), grasped=ever_lifted,
               length=len(rec["eef"]), final_pos=env.cube_pos)
    return out


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--color", default="red")
    p.add_argument("--n", type=int, default=1)
    args = p.parse_args()
    cfg = load_config(args.config)
    from pai_steer.policy.act import load_policy

    model, _, stats = load_policy(args.ckpt)
    pol = Policy(model, stats)
    env = make_env(cfg, noise=True)
    for s in range(args.n):
        r = rollout(pol, env, args.color, s, cfg)
        print(f"{args.color} seed {s}: placed={r['placed_box']} success={r['success']} grasped={r['grasped']} len={r['length']}")


if __name__ == "__main__":
    main()
