"""補正した x/y 介入 (V4: 軸ごとに較正 + 対角は対ごとの倍率 + −y を含む方向は下ろし始めで停止) の推論動画.

pattern_videos --grid と同じ 4×4 (行 = キューブの色, 列 = 行き先の箱, 対角 = 介入なし). 比較用に従来の SAE (上位3特徴) も同じ seed で作る.
出力 artifacts/videos/patterns/grid_xy_corrected_{agentview|topdown}[_2x]_seed{seed}.mp4, grid_xy_sae_old_..._seed{seed}.mp4
"""
import json

import imageio.v2 as imageio
import numpy as np

from pai_steer.common import CKPT, COLOR_JP, COLORS, FEATURES, ROLLOUTS, VIDEOS, base_parser, ensure, load_config
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs
from pai_steer.steer.specs import p1_spec
from pai_steer.analysis.direction_study import direction
from pai_steer.analysis.xy_combo import SITE
from pai_steer.analysis.xy_combo_calib import slopes
from pai_steer.analysis.xy_corrected import grid_best
from pai_steer.analysis.xy_failure import axis_spec
from pai_steer.viz.pattern_videos import FPS, clip
from pai_steer.viz.render import StateRenderer, put_text


def build(R, res, name, title, cam, speed, out_dir, seed):
    base = {s: clip(R, res[("base", s, s)], s, f"{COLOR_JP[s]}: 介入なし", speed, (cam,)) for s in COLORS}
    rows, n_ok = [], 0
    for s in COLORS:
        row = []
        for d in COLORS:
            if d == s:
                row.append(base[s]); continue
            r = res[(name, s, d)]
            ok = r["placed_box"] == d
            n_ok += ok
            tag = r["job"].get("tag", "")
            row.append(clip(R, r, s, f"{'○' if ok else '×'} {COLOR_JP[s]} → {COLOR_JP[d]}の箱{tag}", speed, (cam,)))
        rows.append(row)
    n = max(len(c) for row in rows for c in row)
    rows = [[c + [c[-1]] * (n - len(c)) for c in row] for row in rows]
    frames = []
    for i in range(n):
        f = np.concatenate([np.concatenate([c[i] for c in row], 1) for row in rows], 0)
        head = np.full((60, f.shape[1], 3), 25, np.uint8)
        head = put_text(head, f"{title}   行 = キューブの色 / 列 = 行き先の箱 (対角 = 介入なし)   成功 {n_ok}/12   seed {seed}", "tl", sz=20)
        head = put_text(head, "赤枠 = 介入中 / 下端のバー: オレンジ = 介入した区間, 白線 = 現在", "bl", sz=15, color=(255, 170, 60))
        frames.append(put_text(np.concatenate([head, f], 0), f"{speed:g}x", "br", sz=22, color=(255, 230, 0)))
    path = out_dir / f"grid_{name}_{cam}{'' if speed == 1 else f'_{speed:g}x'}_seed{seed}.mp4"
    imageio.mimwrite(path, frames, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=16)
    print("saved", path.name, f"成功 {n_ok}/12")


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()
    cfg = load_config(args.config)
    L = Layout(cfg["env"])
    combo = json.load(open(FEATURES / f"xy_combo_{SITE}.json"))
    sl, gb = slopes(), grid_best()
    out_dir = VIDEOS / "patterns"
    ensure(out_dir, ROLLOUTS)
    jobs = [dict(name="base", color=s, dst=s, seed=args.seed, record_states=True) for s in COLORS]
    for a in COLORS:
        for b in COLORS:
            if a == b:
                continue
            dr = direction(L, a, b)
            k = gb[(a, b)] if "&" in dr else (1.0, 1.0)
            neg_y = "-y" in dr
            tag = (f"\nx×{k[0]:g} y×{k[1]:g}" if "&" in dr else "") + ("\n下ろし始めで停止" if neg_y else "")
            jobs.append(dict(name="xy_corrected", color=a, dst=b, seed=args.seed, record_states=True, tag=tag,
                             gate="transport" if neg_y else "lift", steer=axis_spec(combo, sl, L, a, b, *k)))
            jobs.append(dict(name="xy_sae_old", color=a, dst=b, seed=args.seed, record_states=True, gate="lift",
                             steer=p1_spec(L, a, b, 0.5)))
    res = {(r["job"]["name"], r["job"]["color"], r["job"]["dst"]): r for r in run_jobs(jobs, args.ckpt, cfg)}
    np.save(ROLLOUTS / f"xy_corrected_rollouts_seed{args.seed}.npy", res, allow_pickle=True)
    R = StateRenderer(cfg, size=240)
    for name, title in [("xy_corrected", "x/y 移動 (補正版: 組み合わせ特徴 + 軸ごとに較正 + 対角補正 + −y は下ろし始めで停止)"),
                        ("xy_sae_old", "x/y 移動 (従来: 上位3特徴, α=0.5)")]:
        build(R, res, name, title, "agentview", 1.0, out_dir, args.seed)
        build(R, res, name, title, "topdown", 2.0, out_dir, args.seed)


if __name__ == "__main__":
    main()
