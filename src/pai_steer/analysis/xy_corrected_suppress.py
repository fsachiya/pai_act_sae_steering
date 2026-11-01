"""補正した x/y 介入 (V4) に色の抑制を組み合わせた検証 (E1 規模, 評価 seed 100..119).

V4 = 組み合わせ特徴 + 軸ごとに較正 + 対角は対ごとの倍率 + −y を含む方向は下ろし始めで停止.
色の抑制 (encoder_layer0 の color トークン, x/y 介入と同じ gate で入れる):
  sae_src : SAE で元の色の特徴を 0      dm_mean : DiffMean で color トークンを4色の平均に
−y を含む方向の介入の終わり方: transport (下ろし始めで停止, V4 と同じ) / lift (最後まで続ける)
比較の V4 単独は xy_corrected.csv の結果を使う (同じ seed).
出力 artifacts/results/xy_corrected_suppress.csv, 図 artifacts/figs/xy_corrected_suppress.png
"""
import json

import numpy as np
import pandas as pd

from pai_steer.common import CKPT, COLORS, FEATURES, FIGS, RESULTS, base_parser, ensure, load_config, setup_mpl
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs, wilson
from pai_steer.steer.specs import suppress_color_spec
from pai_steer.steer.sweep import to_df
from pai_steer.analysis.direction_study import direction
from pai_steer.analysis.xy_combo import SITE
from pai_steer.analysis.xy_combo_calib import slopes
from pai_steer.analysis.xy_corrected import SEEDS, grid_best
from pai_steer.analysis.xy_failure import axis_spec

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402

SUP_JP = {"sae_src": "SAE: 元の色の特徴を0", "dm_mean": "DiffMean: 色を中立化"}
GATE_JP = {"transport": "−y は下ろし始めで停止", "lift": "−y も最後まで介入"}
DIRS = ["+x", "-x", "+y", "-y", "+x&+y", "-x&+y", "+x&-y", "-x&-y"]


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(RESULTS, FIGS)
    L = Layout(cfg["env"])
    combo = json.load(open(FEATURES / f"xy_combo_{SITE}.json"))
    sl, gb = slopes(), grid_best()
    jobs = []
    for sup in ["sae_src", "dm_mean"]:
        for neg_gate in ["transport", "lift"]:
            v = f"V4 + {SUP_JP[sup]} ({GATE_JP[neg_gate]})"
            for a in COLORS:
                for b in COLORS:
                    if a == b:
                        continue
                    dr = direction(L, a, b)
                    k = gb[(a, b)] if "&" in dr else (1.0, 1.0)
                    spec = [axis_spec(combo, sl, L, a, b, *k), suppress_color_spec(a, sup)]
                    gate = neg_gate if "-y" in dr else "lift"
                    jobs += [dict(var=v, color=a, dst=b, strength=0.0, seed=s, gate=gate, steer=spec) for s in SEEDS]
    print(len(jobs), "episodes")
    res = run_jobs(jobs, args.ckpt, cfg)
    df = to_df(res, L)
    df["var"] = [r["job"]["var"] for r in res]
    df["dir"] = [direction(L, s, t) for s, t in zip(df.src, df.dst)]
    old = pd.read_csv(RESULTS / "xy_corrected.csv")
    old = old[old["var"].isin(["V2 軸ごとに較正", "V4 補正A(−y) + 補正B(対角)"])].copy()
    old["var"] = old["var"].replace({"V4 補正A(−y) + 補正B(対角)": "V4 (色の抑制なし)"})
    df = pd.concat([old[df.columns.intersection(old.columns)], df], ignore_index=True)
    df.to_csv(RESULTS / "xy_corrected_suppress.csv", index=False)
    rows = []
    for v, g in df.groupby("var", sort=False):
        pr, lo, hi = wilson(g.reached.sum(), len(g))
        rows.append(dict(条件=v, 到達率=pr, ci_lo=lo, ci_hi=hi, 元の箱=g.stayed.mean(), 箱外=g.placed_box.isna().mean(),
                         長さ=g.length.mean(), **g.groupby("dir").reached.mean().to_dict()))
    sm = pd.DataFrame(rows)
    sm.to_csv(RESULTS / "xy_corrected_suppress_summary.csv", index=False)
    pd.set_option("display.width", 260)
    print(sm.round(2).to_string())
    neg = df[df.dir.str.contains("-y")]
    print("\n−y を含む方向の結果の内訳:")
    print(neg.groupby("var", sort=False).agg(到達=("reached", "mean"), 元の箱=("stayed", "mean"),
                                            箱外=("placed_box", lambda x: x.isna().mean()), 長さ=("length", "mean")).round(2).to_string())

    fig, axs = plt.subplots(1, 2, figsize=(19, 5), gridspec_kw=dict(width_ratios=[1, 1.6]))
    a = axs[0]
    y = np.arange(len(sm))
    a.barh(y, sm.到達率, xerr=[sm.到達率 - sm.ci_lo, sm.ci_hi - sm.到達率], capsize=3)
    for i, r in sm.iterrows():
        a.text(r.ci_hi + 0.02, i, f"{r.到達率:.0%}", va="center")
    a.set_yticks(y); a.set_yticklabels(sm.条件, fontsize=8); a.invert_yaxis(); a.set_xlim(0, 1.1)
    a.set_xlabel("行き先の箱への到達率 (12 対 × 20, 95% Wilson CI)")
    a.set_title("補正版 x/y 介入 + 色の抑制 (評価 seed 100-119)", fontsize=10)
    a = axs[1]
    M = sm.set_index("条件").reindex(columns=DIRS).values
    im = a.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=9, color="w" if M[i, j] < .6 else "k")
    a.set_xticks(range(len(DIRS))); a.set_xticklabels(DIRS); a.set_yticks(range(len(sm))); a.set_yticklabels(sm.条件, fontsize=8)
    a.set_title("方向ごとの到達率", fontsize=10); plt.colorbar(im, ax=a)
    plt.tight_layout(); plt.savefig(FIGS / "xy_corrected_suppress.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
