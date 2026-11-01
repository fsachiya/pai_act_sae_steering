"""補正した x/y steering の精度検証 (E1 と同じ規模: 4色 × 他の3色 × 20 エピソード).

評価用の seed はこれまでの調整 (グリッド探索 seed 0..4) と重ならない 100..119 を使う.
  V1 組み合わせ版・強さ一律         : α=0.5 (全方向共通)
  V2 軸ごとに較正                  : α_d = 14cm / 傾き_d
  V3 較正 + 補正A (全方向)          : V2 に加え, 下ろし始めたら介入を止める (gate=transport)
  V4 補正A (−y を含む方向のみ) + 補正B : 対角は 2次元グリッドで最もよかった x/y の倍率を対ごとに使う
比較: 従来の SAE (上位3特徴, α=0.5) と DiffMean (α=1) も同じ seed で回す.
出力 artifacts/results/xy_corrected.csv, 図 artifacts/figs/xy_corrected.png
"""
import json

import numpy as np
import pandas as pd

from pai_steer.common import CKPT, COLOR_JP, COLORS, FEATURES, FIGS, RESULTS, base_parser, ensure, load_config, setup_mpl
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs, wilson
from pai_steer.steer.specs import p1_dm_spec, p1_spec
from pai_steer.steer.sweep import to_df
from pai_steer.analysis.direction_study import direction
from pai_steer.analysis.xy_combo import SITE, reach_spec
from pai_steer.analysis.xy_combo_calib import slopes
from pai_steer.analysis.xy_failure import axis_spec

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402

SEEDS = range(100, 120)
VARS = ["従来 SAE (上位3特徴)", "DiffMean", "V1 組み合わせ・一律", "V2 軸ごとに較正", "V3 較正 + 下ろし始めで停止",
        "V4 補正A(−y) + 補正B(対角)"]


def grid_best():
    g = pd.read_csv(RESULTS / "xy_diag_grid.csv")
    out = {}
    for (a, b), h in g.groupby(["src", "dst"]):
        h = h.assign(tot=h.kx + h.ky).sort_values(["reached", "tot"], ascending=[False, True])
        r = h.iloc[0]
        out[(a, b)] = (float(r.kx), float(r.ky)) if r.reached > 0 else (1.0, 1.0)
    return out


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(RESULTS, FIGS)
    L = Layout(cfg["env"])
    combo = json.load(open(FEATURES / f"xy_combo_{SITE}.json"))
    sl = slopes()
    gb = grid_best()
    print("対角の倍率 (グリッドで最良):", gb)
    jobs = []
    for a in COLORS:
        for b in COLORS:
            if a == b:
                continue
            dr = direction(L, a, b)
            has_neg_y = "-y" in dr
            specs = {
                VARS[0]: (p1_spec(L, a, b, 0.5), "lift"),
                VARS[1]: (p1_dm_spec(L, a, b, 1.0), "lift"),
                VARS[2]: (reach_spec(combo, L, a, b, 0.5), "lift"),
                VARS[3]: (axis_spec(combo, sl, L, a, b, 1.0, 1.0), "lift"),
                VARS[4]: (axis_spec(combo, sl, L, a, b, 1.0, 1.0), "transport"),
                VARS[5]: (axis_spec(combo, sl, L, a, b, *(gb[(a, b)] if "&" in dr else (1.0, 1.0))),
                          "transport" if has_neg_y else "lift"),
            }
            for v, (spec, gate) in specs.items():
                jobs += [dict(var=v, color=a, dst=b, strength=0.0, seed=s, gate=gate, steer=spec) for s in SEEDS]
    print(len(jobs), "episodes")
    res = run_jobs(jobs, args.ckpt, cfg)
    df = to_df(res, L)
    df["var"] = [r["job"]["var"] for r in res]
    df["dir"] = [direction(L, s, t) for s, t in zip(df.src, df.dst)]
    df.to_csv(RESULTS / "xy_corrected.csv", index=False)
    rows = []
    for v in VARS:
        g = df[df["var"] == v]
        pr, lo, hi = wilson(g.reached.sum(), len(g))
        rows.append(dict(条件=v, 到達率=pr, ci_lo=lo, ci_hi=hi, 元の箱=g.stayed.mean(), 箱外=g.placed_box.isna().mean(),
                         **g.groupby("dir").reached.mean().to_dict()))
    sm = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(sm.round(2).to_string())
    sm.to_csv(RESULTS / "xy_corrected_summary.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(18, 5.5), gridspec_kw=dict(width_ratios=[1, 1.6]))
    a = axs[0]
    y = np.arange(len(VARS))
    a.barh(y, sm.到達率, xerr=[sm.到達率 - sm.ci_lo, sm.ci_hi - sm.到達率], capsize=3,
           color=["0.6", "0.6", "C0", "C1", "C2", "C3"])
    for i, r in sm.iterrows():
        a.text(r.ci_hi + 0.02, i, f"{r.到達率:.0%}", va="center")
    a.set_yticks(y); a.set_yticklabels(VARS); a.invert_yaxis(); a.set_xlim(0, 1.1)
    a.set_xlabel("行き先の箱への到達率 (12 対 × 20 エピソード, 95% Wilson CI)")
    a.set_title("x/y 移動の介入: 補正による精度の向上 (評価 seed 100-119)", fontsize=10)
    a = axs[1]
    dirs = ["+x", "-x", "+y", "-y", "+x&+y", "-x&+y", "+x&-y", "-x&-y"]
    M = sm.set_index("条件").reindex(columns=dirs).values
    im = a.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=9, color="w" if M[i, j] < .6 else "k")
    a.set_xticks(range(len(dirs))); a.set_xticklabels(dirs); a.set_yticks(range(len(VARS))); a.set_yticklabels(VARS, fontsize=8)
    a.set_title("方向ごとの到達率", fontsize=10); plt.colorbar(im, ax=a)
    plt.tight_layout(); plt.savefig(FIGS / "xy_corrected.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
