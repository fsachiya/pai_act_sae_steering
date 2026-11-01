"""組み合わせ版 x/y 特徴で, 軸ごとに強さを較正した介入 (行き先の箱への到達率).

xy_combo の「強さと移動量」から, 方向 d ごとに離した位置のずれの傾き s_d [m/α] (崩れる前の小さい α で線形近似) を求め,
必要なずれ |Δ| (元の箱→行き先の箱) を出す強さ α_d = |Δ| / s_d を軸ごとに使う. 全体に掛ける倍率 k ∈ {0.8, 1.0, 1.2}.
出力 artifacts/results/xy_combo_calib.csv
"""
import json

import numpy as np
import pandas as pd

from pai_steer.common import CKPT, COLORS, FEATURES, RESULTS, SAE_DIR, base_parser, load_config
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs, wilson
from pai_steer.steer.specs import box_offset
from pai_steer.steer.sweep import to_df
from pai_steer.analysis.xy_combo import SITE, dir_spec
from pai_steer.analysis.direction_study import direction


def slopes():
    d = pd.read_csv(RESULTS / "xy_combo_dose.csv")
    out = {}
    for dr in ["+x", "-x", "+y", "-y"]:
        g = d[(d.d == dr) & (d.alpha <= 0.5)]
        col = "rel_dx_mean" if "x" in dr else "rel_dy_mean"
        out[dr] = float(np.abs(np.polyfit(np.r_[0, g.alpha], np.r_[0, g[col]], 1)[0]))
    return out


def calib_spec(combo, sl, L, src, dst, k):
    off = box_offset(L, src, dst)
    feats, alphas = [], []
    for ax, pos, neg in [(0, "+x", "-x"), (1, "+y", "-y")]:
        if abs(off[ax]) < 1e-6:
            continue
        dr = pos if off[ax] > 0 else neg
        sp = dir_spec(combo, dr, k * abs(off[ax]) / sl[dr])
        feats += sp["feats"]; alphas += sp["alpha"]
    return dict(site=SITE, mode="add", feats=feats, alpha=alphas, sae_path=str(SAE_DIR / f"sae_{SITE}.pt"))


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    args = p.parse_args()
    cfg = load_config(args.config)
    L = Layout(cfg["env"])
    combo = json.load(open(FEATURES / f"xy_combo_{SITE}.json"))
    sl = slopes()
    print("離した位置のずれの傾き [m/α]:", {k: round(v, 3) for k, v in sl.items()},
          " 14cm に必要な α:", {k: round(0.14 / v, 2) for k, v in sl.items()})
    jobs = [dict(color=a, dst=b, strength=k, seed=s, gate="lift", steer=calib_spec(combo, sl, L, a, b, k))
            for k in [0.8, 1.0, 1.2] for a in COLORS for b in COLORS if a != b for s in range(5)]
    res = run_jobs(jobs, args.ckpt, cfg)
    df = to_df(res, L)
    df["dir"] = [direction(L, s, t) for s, t in zip(df.src, df.dst)]
    df.to_csv(RESULTS / "xy_combo_calib.csv", index=False)
    for k, g in df.groupby("strength"):
        pr, lo, hi = wilson(g.reached.sum(), len(g))
        print(f"倍率 {k}: 到達率 {pr:.0%} [{lo:.0%},{hi:.0%}], 元の箱 {g.stayed.mean():.0%}, 箱外 {g.placed_box.isna().mean():.0%}")
    print(df.pivot_table(index="strength", columns="dir", values="reached", aggfunc="mean").round(2).to_string())


if __name__ == "__main__":
    main()
