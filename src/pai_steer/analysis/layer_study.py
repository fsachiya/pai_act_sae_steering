"""SAE をどの位置に適用するのが良いか (層比較).

各位置 (encoder_layer0..3, decoder_layer0..2, decoder_out) について:
  1. SAE の品質: R², 死んだ特徴の割合 (train_sae の metrics)
  2. SAE 置換時の成功率: その位置の活性を SAE 再構成で置き換えて色ごと n_recon エピソード
  3. 目標特徴の選択性: find_goal_features の位置ごとの最良トークンの相対選択性
  4. P2 介入 (SAE swap, β) と DiffMean (α) を同じ位置・トークンで: 行き先の箱への到達率 / 把持率
  5. P1 介入 (decoder の位置のみ, SAE add α と DiffMean α): 到達率と行き先方向のずれ
全色 × 他の3色の箱 × seed 0..n-1. 介入は gate=lift (把持後).
出力 artifacts/results/layer_study.csv, layer_study_summary.csv, 図 artifacts/figs/layer_study.png
--plot-only で既存の layer_study_summary.csv から図だけ描き直す.
"""
import json

import numpy as np
import pandas as pd

from pai_steer.common import (CKPT, COLORS, FEATURES, FIGS, RESULTS, SAE_DIR, SITES, base_parser, ensure, is_decoder,
                              load_config, setup_mpl, site_module)
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs, wilson
from pai_steer.steer.specs import p1_dm_spec, p1_spec, p2_dm_spec, p2_spec
from pai_steer.steer.sweep import to_df

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--n", type=int, default=5, help="介入条件の seed 数 (各 src→dst)")
    p.add_argument("--n-recon", type=int, default=10, help="SAE 置換条件の色ごとのエピソード数")
    p.add_argument("--p2-betas", default="1,2")
    p.add_argument("--p2-dm-alphas", default="1,2")
    p.add_argument("--p1-alphas", default="4")
    p.add_argument("--p1-dm-alphas", default="4")
    p.add_argument("--plot-only", action="store_true", help="ロールアウトせず既存の集計から図だけ描く")
    args = p.parse_args()
    if args.plot_only:
        ensure(FIGS)
        plot(pd.read_csv(RESULTS / "layer_study_summary.csv"))
        return
    cfg = load_config(args.config)
    ensure(RESULTS, FIGS)
    L = Layout(cfg["env"])
    g = json.load(open(FEATURES / "goal_features.json"))
    n = 1 if args.quick else args.n
    sites = [s for s in SITES if (SAE_DIR / f"sae_{s}.pt").exists()]
    jobs = []
    for site in sites:
        for c in COLORS:
            for s in range(1 if args.quick else args.n_recon):
                jobs.append(dict(study="recon", site=site, color=c, dst=c, strength=0.0, seed=s, gate="all",
                                 steer=dict(site=site_module(site), mode="recon", sae_path=str(SAE_DIR / f"sae_{site}.pt"))))
        conds = [("p2_sae", float(b), lambda a, b_, c_, s_=site: p2_spec(a, b_, c_, site=s_, g=g)) for b in args.p2_betas.split(",")]
        conds += [("p2_dm", float(a), lambda a, b_, c_, s_=site: p2_dm_spec(a, b_, c_, site=s_, g=g)) for a in args.p2_dm_alphas.split(",")]
        if is_decoder(site):
            conds += [("p1_sae", float(a), lambda a_, b_, c_, s_=site: p1_spec(L, a_, b_, c_, site=s_)) for a in args.p1_alphas.split(",")]
            conds += [("p1_dm", float(a), lambda a_, b_, c_, s_=site: p1_dm_spec(L, a_, b_, c_, site=s_)) for a in args.p1_dm_alphas.split(",")]
        for study, st, fn in conds:
            for src in COLORS:
                for dst in COLORS:
                    if dst == src:
                        continue
                    spec = fn(src, dst, st)
                    for s in range(n):
                        jobs.append(dict(study=study, site=site, color=src, dst=dst, strength=st, seed=s, gate="lift", steer=spec))
    res = run_jobs(jobs, args.ckpt, cfg)
    df = to_df(res, L)
    df["study"] = [r["job"]["study"] for r in res]
    df["site"] = [r["job"]["site"] for r in res]
    df["success"] = [r["success"] for r in res]
    df.to_csv(RESULTS / "layer_study.csv", index=False)

    rows = []
    for site in sites:
        m = json.load(open(SAE_DIR / f"metrics_{site}.json"))
        row = dict(site=site, r2=m["r2"], dead_frac=m["dead_frac"],
                   goal_rel_sel=g["candidates"][g["best_per_site"][site]]["rel_selectivity"],
                   goal_token=g["candidates"][g["best_per_site"][site]]["token"])
        d = df[(df.site == site) & (df.study == "recon")]
        row["recon_success"] = d.success.mean()
        for (study, st), dd in df[(df.site == site) & (df.study != "recon")].groupby(["study", "strength"]):
            row[f"{study}@{st:g}_reached"] = dd.reached.mean()
            row[f"{study}@{st:g}_grasped"] = dd.grasped.mean()
            if study.startswith("p1"):
                row[f"{study}@{st:g}_shift"] = dd.shift_toward.mean()
        rows.append(row)
    sm = pd.DataFrame(rows)
    sm.to_csv(RESULTS / "layer_study_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(sm.round(3).to_string())

    plot(sm)

def plot(sm):
    """位置ごとの集計 sm (layer_study_summary.csv) から 2×2 の図を描く."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    axs = axs.ravel()
    sites = list(sm.site)
    x = np.arange(len(sites))
    axs[0].bar(x - .2, sm.r2, .4, label="SAE の R²")
    axs[0].bar(x + .2, sm.recon_success, .4, label="SAE 置換時の成功率")
    axs[0].axhline(0.9, ls="--", c="gray", lw=.8)
    axs[1].bar(x, sm.goal_rel_sel, color="C4")
    for i, t in enumerate(sm.goal_token):
        axs[1].text(i, sm.goal_rel_sel[i] + .02, t, ha="center", fontsize=8)
    for a, prefix, title in [(axs[2], "p2", "色の介入 (P2): 行き先の箱への到達率"),
                             (axs[3], "p1", "x/y 移動の介入 (P1): 行き先の箱への到達率")]:
        cols = [c for c in sm.columns if c.startswith(prefix) and c.endswith("_reached")]
        w = 0.8 / max(1, len(cols))
        for j, c in enumerate(cols):
            method, st = c.replace("_reached", "").split("_", 1)[1].split("@")
            sym = "β" if (prefix, method) == ("p2", "sae") else "α"
            a.bar(x - .4 + w * (j + .5), sm[c].fillna(0), w, label=f"{'SAE' if method == 'sae' else 'DiffMean'} {sym}={st}")
        a.set_title(title); a.set_ylim(0, 1.05); a.legend(fontsize=8)
    axs[0].set_title("SAE の品質"); axs[0].legend(fontsize=8, loc="upper center", ncol=2); axs[0].set_ylim(0, 1.2)
    axs[0].set_yticks(np.arange(0, 1.01, 0.2))   # 上限は 1.0. 1.2 までの余白は凡例用
    axs[1].set_title("目標特徴の相対選択性 (最良トークン)"); axs[1].set_ylim(0, 1.1)
    for a in axs:
        a.set_xticks(x); a.set_xticklabels(sites, rotation=30, ha="right", fontsize=9)
    plt.suptitle("SAE を適用する位置の比較 (介入は把持後, 全色 × 他の3色の箱)", fontsize=12)
    plt.tight_layout(); plt.savefig(FIGS / "layer_study.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
