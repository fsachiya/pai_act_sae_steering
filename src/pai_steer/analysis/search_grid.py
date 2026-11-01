"""介入探索の網羅性を示すグリッド探索.

パターン (色 P2 / x/y P1 / 軌道 P3) × 手法 (SAE / DiffMean) ごとに,
  1. 介入位置 (候補の全層) × 強度 を全組み合わせで評価
  2. SAE は主な位置で, 使う特徴の数 n × 強度 も評価
各マス = 4色 × 他の3色の箱 × seed 0..n_seed-1 (gate=lift, 介入は把持後).
出力 artifacts/results/search_grid.csv, search_coverage.csv, 図 artifacts/figs/search_grid.png
"""
import numpy as np
import pandas as pd

from pai_steer.common import (CKPT, COLORS, DEC_SITES, ENC_SITES, FIGS, RESULTS, base_parser, ensure, load_config,
                              setup_mpl)
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs, wilson
from pai_steer.steer.specs import p1_dm_spec, p1_spec, p2_dm_spec, p2_spec, p3_dm_spec, p3_spec
from pai_steer.steer.sweep import to_df

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402

# (パターン, 手法, 位置リスト, 強度リスト, spec(L, src, dst, site, s, n))
GRID = [
    ("色", "SAE (3トークン共通)", ENC_SITES, [0.5, 1, 2, 3, 4], lambda L, a, b, site, s, n: p2_spec(a, b, s, site=site, n=n)),
    ("色", "SAE (色の SAE)", ENC_SITES, [0.5, 1, 2, 3, 4], lambda L, a, b, site, s, n: p2_spec(a, b, s, site=site, color_sae=True, n=n)),
    ("色", "DiffMean", ENC_SITES, [0.5, 1, 1.5, 2, 3], lambda L, a, b, site, s, n: p2_dm_spec(a, b, s, site=site)),
    ("x/y 移動", "SAE", DEC_SITES, [0.25, 0.5, 1, 2, 4], lambda L, a, b, site, s, n: p1_spec(L, a, b, s, site=site, n=n)),
    ("x/y 移動", "DiffMean", DEC_SITES, [0.25, 0.5, 1, 2, 4], lambda L, a, b, site, s, n: p1_dm_spec(L, a, b, s, site=site)),
    ("軌道", "SAE", DEC_SITES, [1, 2, 4, 8, 16], lambda L, a, b, site, s, n: p3_spec(a, b, s, site=site, per_k=True, n=n)),
    ("軌道", "DiffMean", DEC_SITES, [0.5, 1, 2, 4], lambda L, a, b, site, s, n: p3_dm_spec(a, b, s, site=site)),
]
# 特徴数の探索 (SAE のみ): (パターン, 手法, 位置, 強度リスト)
N_FEATS = [1, 3, 5, 10, 20]
NGRID = [("色", "SAE (3トークン共通)", "encoder_layer0", [1, 2]), ("x/y 移動", "SAE", "decoder_out", [0.5, 1]),
         ("軌道", "SAE", "decoder_out", [8, 16])]


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--n-seed", type=int, default=5)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(RESULTS, FIGS)
    L = Layout(cfg["env"])
    n_seed = 1 if args.quick else args.n_seed
    fn_of = {(pt, m): fn for pt, m, _, _, fn in GRID}
    cells = [(pt, m, site, s, None, "grid") for pt, m, sites, sts, _ in GRID for site in sites for s in sts]
    cells += [(pt, m, site, s, n, "n_feat") for pt, m, site, sts in NGRID for s in sts for n in N_FEATS]
    if args.quick:
        cells = cells[::25]
    jobs = []
    for ci, (pt, m, site, s, n, study) in enumerate(cells):
        for a in COLORS:
            for b in COLORS:
                if a == b:
                    continue
                spec = fn_of[(pt, m)](L, a, b, site, s, n)
                for seed in range(n_seed):
                    jobs.append(dict(cell=ci, color=a, dst=b, strength=s, seed=seed, gate="lift", steer=spec))
    print(f"{len(cells)} cells, {len(jobs)} episodes")
    res = run_jobs(jobs, args.ckpt, cfg)
    df = to_df(res, L)
    cell = [cells[r["job"]["cell"]] for r in res]
    df["pattern"], df["method"], df["site"] = [c[0] for c in cell], [c[1] for c in cell], [c[2] for c in cell]
    df["n_feat"] = [c[4] if c[4] is not None else -1 for c in cell]
    df["study"] = [c[5] for c in cell]
    df.to_csv(RESULTS / "search_grid.csv", index=False)
    summarize(df, n_seed)


def summarize(df, n_seed):
    agg = df.groupby(["study", "pattern", "method", "site", "n_feat", "strength"]).agg(
        reached=("reached", "mean"), no_box=("placed_box", lambda x: x.isna().mean()), grasped=("grasped", "mean"),
        n=("reached", "size")).reset_index()
    agg.to_csv(RESULTS / "search_grid_cells.csv", index=False)
    rows = []
    for (pt, m), g in df[df.study == "grid"].groupby(["pattern", "method"], sort=False):
        a = agg[(agg.study == "grid") & (agg.pattern == pt) & (agg.method == m)]
        best = a.loc[a.reached.idxmax()]
        nf = df[(df.study == "n_feat") & (df.pattern == pt) & (df.method == m)]
        bn = agg[(agg.study == "n_feat") & (agg.pattern == pt) & (agg.method == m)]
        k = int(best.reached * best.n)
        _, lo, hi = wilson(k, best.n)
        rows.append(dict(
            パターン=pt, 手法=m, 介入位置=f"{a.site.nunique()} ({', '.join(a.site.unique())})",
            強度=", ".join(f"{x:g}" for x in sorted(a.strength.unique())),
            特徴数=({"色": "5 / 色", "x/y 移動": "3 / 方向", "軌道": "5 / 色 × 位置k"}[pt] if m.startswith("SAE") else "—（ベクトル1本）")
            + (f" + 探索 {', '.join(map(str, sorted(nf.n_feat.unique())))}" if len(nf) else ""),
            マス数=int(len(a) + len(bn)), エピソード数=int(len(g) + len(nf)),
            最高到達率=f"{best.reached:.0%} [{lo:.0%},{hi:.0%}]",
            最高の条件=f"{best.site}, 強度 {best.strength:g}",
            特徴数探索の最高=(f"{bn.reached.max():.0%} (n={int(bn.loc[bn.reached.idxmax()].n_feat)}, 強度 {bn.loc[bn.reached.idxmax()].strength:g})" if len(bn) else "—")))
    cov = pd.DataFrame(rows)
    cov.to_csv(RESULTS / "search_coverage.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(cov.to_string())
    print(f"合計 {len(df)} エピソード (各マス = 4色 × 他の3色 × seed {n_seed})")

    groups = list(df[df.study == "grid"].groupby(["pattern", "method"], sort=False).groups.keys())
    fig, axs = plt.subplots(2, 5, figsize=(26, 9.5))
    for ax, (pt, m) in zip(axs.flat, groups):
        a = agg[(agg.study == "grid") & (agg.pattern == pt) & (agg.method == m)]
        sites = list(dict.fromkeys(a.site)); sts = sorted(a.strength.unique())
        M = np.full((len(sites), len(sts)), np.nan)
        for _, r in a.iterrows():
            M[sites.index(r.site), sts.index(r.strength)] = r.reached
        ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(len(sites)):
            for j in range(len(sts)):
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=8, color="w" if M[i, j] < .6 else "k")
        ax.set_xticks(range(len(sts))); ax.set_xticklabels([f"{x:g}" for x in sts]); ax.set_xlabel("強度")
        ax.set_yticks(range(len(sites))); ax.set_yticklabels(sites, fontsize=8)
        ax.set_title(f"{pt} / {m}", fontsize=10)
    for ax, (pt, m, site, sts) in zip(list(axs.flat)[len(groups):], NGRID):
        b = agg[(agg.study == "n_feat") & (agg.pattern == pt) & (agg.method == m)]
        for s in sts:
            bb = b[b.strength == s].sort_values("n_feat")
            ax.plot(bb.n_feat, bb.reached, marker="o", label=f"強度 {s:g}")
        ax.set_xscale("log"); ax.set_xticks(N_FEATS); ax.set_xticklabels(N_FEATS); ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("使う特徴の数 n"); ax.set_ylabel("到達率"); ax.legend(fontsize=8)
        ax.set_title(f"特徴数の探索: {pt} / {m} ({site})", fontsize=10)
    for ax in list(axs.flat)[len(groups) + len(NGRID):]:
        ax.axis("off")
    fig.suptitle(f"介入探索のグリッド (到達率, 各マス 60 エピソード = 4色 × 他の3色 × seed {n_seed}, 介入は把持後)", fontsize=12)
    plt.tight_layout(); plt.savefig(FIGS / "search_grid.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
