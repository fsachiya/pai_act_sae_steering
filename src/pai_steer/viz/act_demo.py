"""学習済み ACT のみ (介入なし) の推論動画: 4色それぞれ1本 + 4本を 2×2 に並べた動画.

出力 artifacts/samples/act_{color}.mp4 (20fps 等倍), act_all.mp4 (2×2, 2倍速), act_summary.json
"""
import json

import imageio.v2 as imageio
import numpy as np

from pai_steer.common import CKPT, COLOR_JP, COLORS, SAMPLES, base_parser, ensure, load_config
from pai_steer.env.cube_sort_env import make_env
from pai_steer.policy.act import load_policy
from pai_steer.policy.rollout import Policy, rollout
from pai_steer.viz.render import StateRenderer, put_text

PHASE_JP = {"pre": "把持前", "carry": "運搬中", "post": "離した後"}


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(SAMPLES)
    model, _, stats = load_policy(args.ckpt)
    pol = Policy(model, stats)
    env = make_env(cfg, noise=True)
    R = StateRenderer(cfg, size=384)
    summary, clips = {}, {}
    for c in COLORS:
        r = rollout(pol, env, c, args.seed, cfg, record_states=True)
        summary[c] = dict(seed=args.seed, success=r["success"], placed_box=r["placed_box"], grasped=r["grasped"],
                          length=r["length"], final_cube_pos=np.round(r["final_pos"], 4).tolist())
        states = list(r["states"]) + [r["final_state"]] * 10
        phases = list(r["phase"]) + [r["phase"][-1]] * 10
        frames = []
        for t, (s, ph) in enumerate(zip(states, phases)):
            fr = np.concatenate([R.render(s, c, "agentview"), R.render(s, c, "topdown")], 1)
            fr = put_text(fr, f"ACT のみ (介入なし)  色: {COLOR_JP[c]} ({c})\n推定フェーズ: {PHASE_JP[ph]}", "tl", sz=20)
            fr = put_text(fr, f"step {min(t, r['length'])}/{r['length']}", "tr", sz=20)
            res = "成功" if r["success"] else "失敗"
            if t >= r["length"]:
                fr = put_text(fr, f"結果: {COLOR_JP.get(r['placed_box'], '箱外')}の箱 ({res})", "bl", sz=22, color=(255, 230, 0))
            frames.append(fr)
        imageio.mimwrite(SAMPLES / f"act_{c}.mp4", frames, fps=20, codec="libx264", quality=8, pixelformat="yuv420p",
                         macro_block_size=16)
        clips[c] = frames
        print(c, summary[c])
    # 2x2 (2倍速: 1フレームおき, 20fps)
    n = max(len(v) for v in clips.values())
    pad = {c: v + [v[-1]] * (n - len(v)) for c, v in clips.items()}
    small = lambda f: f[::2, ::2]
    grid = [np.concatenate([np.concatenate([small(pad[COLORS[0]][i]), small(pad[COLORS[1]][i])], 1),
                            np.concatenate([small(pad[COLORS[2]][i]), small(pad[COLORS[3]][i])], 1)], 0)
            for i in range(0, n, 2)]
    imageio.mimwrite(SAMPLES / "act_all.mp4", grid, fps=20, codec="libx264", quality=8, pixelformat="yuv420p",
                     macro_block_size=16)
    json.dump(summary, open(SAMPLES / "act_summary.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
