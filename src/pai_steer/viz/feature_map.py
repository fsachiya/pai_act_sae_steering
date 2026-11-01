"""SAE の介入で刺激した特徴を, 辞書 (2048 特徴) を一列に並べた上に色付けして示す.

各パターンの網羅探索で最も到達率が高かった設定の特徴集合:
  色   : 色の SAE (encoder_layer0 の color トークン), 色ごとに 5 特徴 (swap)
  x/y  : decoder_out の SAE, 方向ごとに上位 5 特徴 (add)
  軌道 : decoder_out の SAE, 色 × チャンク内の位置 k ごとに 5 特徴 (swap). 濃さ = その特徴を使う位置 k の数
出力 artifacts/figs/sae_feature_map.png
"""
import json
from collections import Counter

from pai_steer.common import COLOR_JP, COLORS, FEATURES, FIGS, ensure, setup_mpl

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from pai_steer.env.layout import RGB  # noqa: E402

D = 2048
BAR_W = 9          # 2048 本のうち 1 本が見えるよう, 描画上の幅を広げる
DIR_C = {"+x": "#d62728", "-x": "#ff9896", "+y": "#1f77b4", "-y": "#9ecae1"}


def strip(ax, rows, title, labels=True, note=None):
    """rows: [(行ラベル, 色, {特徴: 濃さ0..1})]. 共通する特徴は黒枠で囲む."""
    count = Counter(f for _, _, fs in rows for f in fs)
    pitch = 1.6 if labels else 1.0     # 番号を書く場合は上の行と重ならないよう行間を広げる
    for r, (name, col, fs) in enumerate(rows):
        y = (len(rows) - 1 - r) * pitch
        ax.add_patch(Rectangle((0, y + 0.15), D, 0.7, fc="#f2f2f2", ec="none"))
        for f, a in fs.items():
            shared = count[f] > 1
            ax.add_patch(Rectangle((f - BAR_W / 2, y + 0.15), BAR_W, 0.7, fc=col, alpha=0.25 + 0.75 * a,
                                   ec="black" if shared else "none", lw=0.9))
            if labels:
                ax.text(f, y + 0.9, str(f), ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_yticks([(len(rows) - 1 - r) * pitch + 0.5 for r in range(len(rows))])
    ax.set_yticklabels([n for n, _, _ in rows], fontsize=9)
    ax.set_xlim(-20, D + 20); ax.set_ylim(0, (len(rows) - 1) * pitch + (1.7 if labels else 1.0))
    ax.set_xticks(range(0, D + 1, 256)); ax.tick_params(axis="x", labelsize=8)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=10, loc="left")
    if note:
        ax.annotate(note, xy=(1.0, 0), xycoords="axes fraction", xytext=(0, -30), textcoords="offset points",
                    ha="right", va="top", fontsize=7.5, color="#444")


def main():
    ensure(FIGS)
    g = json.load(open(FEATURES / "goal_features.json"))
    color = g["candidates"][g["best_color_sae"]]["sets"]
    xy = json.load(open(FEATURES / "xy_features_decoder_out.json"))["features"]
    traj = json.load(open(FEATURES / "traj_features_decoder_out.json"))
    K = len(traj["per_k"])

    fig, axs = plt.subplots(3, 1, figsize=(13, 12), gridspec_kw=dict(hspace=0.6, height_ratios=[1.35, 1.35, 1]))
    strip(axs[0], [(COLOR_JP[c], RGB[c], {f: 1.0 for f in color[c]}) for c in COLORS],
          "① 色の介入（色の SAE, encoder 層0 の color トークン, 色ごとに 5 特徴）　到達率 95%",
          note="swap: 元の色の行の特徴を 0 にし, 行き先の色の行の特徴を「その色での平均活性 × β」にする")
    strip(axs[1], [(d, DIR_C[d], {r["feat"]: 1.0 for r in xy[d][:5]}) for d in ["+x", "-x", "+y", "-y"]],
          "② x/y 移動の介入（decoder 出力の SAE, 方向ごとに上位 5 特徴）　到達率 18%",
          note="add: 行き先の方向の行の特徴に α × 典型活性を足す（対角はx と y の両方の行）。黒枠 = 複数の方向で選ばれた特徴")
    rows = []
    for c in COLORS:
        cnt = Counter(f for d in traj["per_k"] for f in d["sets"][c])
        rows.append((COLOR_JP[c], RGB[c], {f: n / K for f, n in cnt.items()}))
    strip(axs[2], rows, f"③ 軌道の介入（decoder 出力の SAE, 色 × チャンク内の位置 k（{K} 個）ごとに 5 特徴）　到達率 22%",
          labels=False,
          note="swap（位置 k ごと）。濃さ = その特徴を使う位置 k の数（濃いほど多くの位置で使う）。4 色の間で共通する特徴はない")
    for a in axs:
        a.set_xlabel("SAE の特徴の番号（辞書 2048 個）", fontsize=8)
    fig.suptitle("SAE の介入で刺激した特徴（各パターンの網羅探索で最も到達率が高かった設定）", fontsize=12, y=0.93)
    plt.savefig(FIGS / "sae_feature_map.png", dpi=150, bbox_inches="tight"); plt.close()
    print("saved", FIGS / "sae_feature_map.png")


if __name__ == "__main__":
    main()
