"""3パターン (色, x/y, 軌道) × SAE / DiffMean の E1 行列を1枚にまとめた図.

出力 artifacts/figs/summary_matrix.png, artifacts/results/summary_e1.csv
"""
import numpy as np
import pandas as pd

from pai_steer.common import COLOR_JP, COLORS, FIGS, RESULTS, base_parser, ensure, setup_mpl
from pai_steer.steer.runner import wilson

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402

CONDS = [  # (行, 列, ファイル, 見出し)
    (0, 0, "matrix_p2", "色 (P2) / 3トークン共通 SAE swap β=2"),
    (0, 1, "matrix_p1", "x/y 移動 (P1) / SAE add α=0.5"),
    (0, 2, "matrix_p3_perk", "軌道 (P3) / SAE swap (位置k別) β=8"),
    (1, 0, "matrix_p2_dm", "色 (P2) / DiffMean α=1.5"),
    (1, 1, "matrix_p1_dm", "x/y 移動 (P1) / DiffMean α=1"),
    (1, 2, "matrix_p3_dm", "軌道 (P3) / DiffMean α=1"),
]


def main():
    base_parser(__doc__).parse_args()
    ensure(FIGS)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    rows = []
    for r, c, name, title in CONDS:
        df = pd.read_csv(RESULTS / f"{name}.csv")
        M = np.zeros((4, 4)); N = np.zeros((4, 4))
        for (s, d), g in df.groupby(["src", "dst"]):
            M[COLORS.index(s), COLORS.index(d)] = g.reached.mean(); N[COLORS.index(s), COLORS.index(d)] = len(g)
        off = df[df.src != df.dst]
        p, lo, hi = wilson(off.reached.sum(), len(off))
        rows.append(dict(condition=title, file=name, reached=p, ci_lo=lo, ci_hi=hi, no_box=off.placed_box.isna().mean(),
                         stayed=off.stayed.mean(), grasped=df.grasped.mean(),
                         **{f"reached_{k}": v for k, v in off.groupby("kind").reached.mean().items()}))
        a = axs[r, c]
        a.imshow(M, vmin=0, vmax=1, cmap="viridis")
        for i in range(4):
            for j in range(4):
                a.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=10,
                       color="w" if M[i, j] < .6 else "k", fontweight="bold" if i == j else None)
        a.set_xticks(range(4)); a.set_xticklabels([f"{COLOR_JP[x]}の箱" for x in COLORS], fontsize=8)
        a.set_yticks(range(4)); a.set_yticklabels([f"{COLOR_JP[x]}キューブ" for x in COLORS], fontsize=8)
        a.set_title(f"{title}\n介入セルの到達率 {p:.0%} [{lo:.0%},{hi:.0%}], 箱外 {off.placed_box.isna().mean():.0%}", fontsize=9)
    fig.suptitle("E1: 行き先の箱への到達率 (行=元の色, 列=行き先, 各セル 20 エピソード, 対角=介入なし)\n上段: SAE / 下段: DiffMean",
                 fontsize=11)
    plt.tight_layout(); plt.savefig(FIGS / "summary_matrix.png", dpi=120); plt.close()
    sm = pd.DataFrame(rows)
    sm.to_csv(RESULTS / "summary_e1.csv", index=False)
    print(sm.round(3).to_string())


if __name__ == "__main__":
    main()
