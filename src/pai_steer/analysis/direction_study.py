"""グリッド探索の結果を「ずらす方向」別に集計する (要因3: データに無い向きの動き).

元の箱→行き先の箱のずれ (+x, -x, +y, -y, 対角の組み合わせ) ごとに, パターン × 手法ごとの最高到達率と
そのときの位置・強度. 学習データの運搬方向の分布も添える.
出力 artifacts/results/direction_study.csv, 図 artifacts/figs/direction_study.png
"""
import numpy as np
import pandas as pd

from pai_steer.common import FIGS, RESULTS, base_parser, ensure, load_config, setup_mpl
from pai_steer.data.dataset import ChunkDataset
from pai_steer.env.layout import Layout

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402

DIRS = ["+x", "-x", "+y", "-y", "+x&+y", "+x&-y", "-x&+y", "-x&-y"]


def direction(L, s, d):
    o = L.box_xy[d] - L.box_xy[s]
    r = []
    if abs(o[0]) > 1e-6:
        r.append("+x" if o[0] > 0 else "-x")
    if abs(o[1]) > 1e-6:
        r.append("+y" if o[1] > 0 else "-y")
    return "&".join(r)


def main():
    base_parser(__doc__).parse_args()
    cfg = load_config()
    ensure(RESULTS, FIGS)
    L = Layout(cfg["env"])
    df = pd.read_csv(RESULTS / "search_grid.csv")
    df = df[df.study == "grid"].copy()
    df["dir"] = [direction(L, s, d) for s, d in zip(df.src, df.dst)]
    cell = df.groupby(["pattern", "method", "site", "strength", "dir"]).agg(reached=("reached", "mean"), n=("reached", "size")).reset_index()
    rows = []
    for (pt, m), g in cell.groupby(["pattern", "method"], sort=False):
        for dr in DIRS:
            h = g[g.dir == dr]
            if h.empty:
                continue
            b = h.loc[h.reached.idxmax()]
            rows.append(dict(pattern=pt, method=m, dir=dr, best_reached=b.reached, site=b.site, strength=b.strength, n=int(b.n)))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "direction_study.csv", index=False)
    piv = out.pivot_table(index=["pattern", "method"], columns="dir", values="best_reached", sort=False).reindex(columns=DIRS)
    pd.set_option("display.width", 250)
    print(piv.round(2).to_string())

    ds = ChunkDataset()
    m = np.isin(ds.phase, [4, 5])
    v = np.diff(ds.eef, axis=0, append=ds.eef[-1:])[m]
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw=dict(width_ratios=[1, 2.4]))
    a = axs[0]
    a.hist2d(v[:, 1] * 1000, v[:, 0] * 1000, bins=60, range=[[-15, 15], [-15, 15]], cmap="magma")
    a.set_xlabel("1ステップの dy [mm] (+y = 左 = 箱の側)"); a.set_ylabel("1ステップの dx [mm]")
    a.set_title(f"学習データ: 運搬中の手先の動き\n+y {np.mean(v[:, 1] > .001):.0%}, -y {np.mean(v[:, 1] < -.001):.0%}, "
                f"|dx| 平均 {np.abs(v[:, 0]).mean() * 1000:.1f}mm, |dy| 平均 {np.abs(v[:, 1]).mean() * 1000:.1f}mm", fontsize=9)
    a = axs[1]
    M = piv.values
    a.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                a.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=8, color="w" if M[i, j] < .6 else "k")
    a.set_xticks(range(len(DIRS))); a.set_xticklabels(DIRS)
    a.set_yticks(range(len(piv))); a.set_yticklabels([f"{p} / {m}" for p, m in piv.index], fontsize=8)
    a.set_xlabel("元の箱 → 行き先の箱のずれの方向")
    a.set_title("方向ごとの最高到達率 (グリッド探索の全ての位置 × 強度の中での最高)", fontsize=10)
    plt.tight_layout(); plt.savefig(FIGS / "direction_study.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
