"""ACT のモデル図と, 介入した位置を書き込んだモデル図.

出力 artifacts/figs/act_arch.png, act_intervention.png
"""
from pai_steer.common import FIGS, ensure, setup_mpl

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

TOK_C = {"latent": "#d9d9d9", "robot": "#9ecae1", "cube": "#fdd0a2", "color": "#c7a6dd"}
ENC_C, DEC_C, HEAD_C, CVAE_C = "#e5f0fa", "#fdeee0", "#e6f4e6", "#f4f4f4"
INT_C = {"color": "#c0392b", "xy": "#1f6fb4", "traj": "#2e8b57"}


def box(ax, x, y, w, h, text, fc, ec="#333", fs=9, ls="-", lw=1.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arrow(ax, p, q, c="#333", lw=1.1, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=c, lw=lw, ls=ls))


def draw(ax, interventions=False):
    ax.set_xlim(0, 16.8); ax.axis("off")
    # 入力 → Linear → トークン列
    toks = [("latent", "z (32)\n推論時 z=0"), ("robot", "robot (9)\n手先位置・姿勢・グリッパー"),
            ("cube", "cube (10)\nキューブ位置・姿勢・相対位置"), ("color", "color (4)\n色の one-hot")]
    ys = [8.3, 6.6, 4.9, 3.2]
    for (t, label), y in zip(toks, ys):
        box(ax, 0.2, y, 2.6, 1.2, label, TOK_C[t], fs=8)
        arrow(ax, (2.8, y + 0.6), (3.4, y + 0.6))
        box(ax, 3.4, y + 0.25, 1.1, 0.7, "Linear", "white", fs=8)
        arrow(ax, (4.5, y + 0.6), (5.0, y + 0.6))
    box(ax, 5.0, 3.2, 0.7, 6.3, "", "#fafafa")
    ax.text(5.35, 6.35, "トークン列 4 × 256（＋位置埋め込み）", rotation=90, ha="center", va="center", fontsize=8)
    # encoder
    box(ax, 6.1, 2.9, 3.0, 6.9, "", ENC_C, lw=1.3)
    arrow(ax, (5.7, 9.0), (6.4, 9.0))
    ax.text(7.6, 10.05, "Transformer encoder", ha="center", fontsize=10, weight="bold")
    ax.text(7.6, 3.15, "d_model 256 / 8 heads / FFN 1024", ha="center", fontsize=7)
    enc_y = [8.5, 7.0, 5.5, 4.0]
    for i, y in enumerate(enc_y):
        box(ax, 6.4, y, 2.4, 1.0, f"encoder 層{i}\nself-attention + FFN", "white", fs=8)
        if i:
            arrow(ax, (7.6, enc_y[i - 1]), (7.6, y + 1.0))
    # decoder
    box(ax, 10.3, 2.9, 3.0, 6.9, "", DEC_C, lw=1.3)
    ax.text(11.8, 10.05, "Transformer decoder", ha="center", fontsize=10, weight="bold")
    box(ax, 10.6, 8.9, 2.4, 0.7, "学習するクエリ 20 個", "white", fs=8)
    dec_y = [7.5, 6.1, 4.7, 3.3]
    hub = (9.5, 5.6)
    arrow(ax, (8.8, 4.5), hub, c="#777", lw=0.8, style="-")
    for i, y in enumerate(dec_y):
        box(ax, 10.6, y, 2.4, 1.0, f"decoder 層{i}\nself-attn + cross-attn + FFN", "white", fs=7.5)
        arrow(ax, (11.8, (8.9 if i == 0 else dec_y[i - 1])), (11.8, y + 1.0))
        arrow(ax, hub, (10.6, y + 0.5), c="#777", lw=0.8)
    ax.text(9.5, 2.55, "encoder の出力を\n全 decoder 層が\ncross-attention で参照", ha="center", va="top", fontsize=7, color="#555")
    # action head
    box(ax, 14.1, 5.9, 2.5, 1.0, "Linear (action head)", HEAD_C, fs=8)
    arrow(ax, (13.0, 3.8), (14.3, 5.9))
    box(ax, 14.1, 3.9, 2.5, 1.5, "行動チャンク\n20 ステップ × 7 次元\n(先頭 5 ステップを実行)", HEAD_C, fs=8)
    arrow(ax, (15.35, 5.9), (15.35, 5.4))
    # CVAE encoder (学習時のみ): z を上側の経路で渡す
    box(ax, 13.9, 8.0, 2.8, 1.8, "CVAE encoder（学習時のみ）\n[CLS, robot, a₁…a₂₀] → μ, σ\n→ z をサンプル", CVAE_C, ls="--", fs=7.5)
    ax.plot([15.3, 15.3, 1.5], [9.8, 10.55, 10.55], c="#999", lw=0.8, ls="--")
    arrow(ax, (1.5, 10.55), (1.5, 9.5), c="#999", lw=0.8, ls="--")
    ax.text(8.4, 10.62, "学習時: CVAE がサンプルした z を入力（推論時は z=0）", ha="center", va="bottom", fontsize=7, color="#777")
    if not interventions:
        return
    # SAE を学習した 8 か所 (位置の比較)
    for x, y in [(8.8, y + 0.5) for y in enc_y] + [(13.0, y + 0.5) for y in dec_y]:
        ax.plot(x + 0.12, y, "o", ms=6, mfc="white", mec="#555", mew=1.2, zorder=5)
    # ① 色: encoder 層0 出力の color トークン
    c = INT_C["color"]
    ax.plot(8.92, 9.0, "o", ms=10, mfc=c, mec="white", mew=1.2, zorder=6)
    ax.annotate("① 色の介入\nencoder 層0 出力の color トークン", xy=(8.95, 9.1), xytext=(10.4, 11.2), fontsize=8.5, color=c,
                weight="bold", ha="center", arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4))
    # ②③: decoder 出力 (action head の直前, 20 クエリすべて)
    ax.plot(13.12, 3.8, "o", ms=10, mfc=INT_C["xy"], mec="white", mew=1.2, zorder=6)
    ax.plot(13.32, 3.8, "o", ms=10, mfc=INT_C["traj"], mec="white", mew=1.2, zorder=7)
    ax.annotate("② x/y 移動の介入・③ 軌道の介入\ndecoder 出力（20 クエリすべて）", xy=(13.2, 3.65), xytext=(13.6, 1.4),
                fontsize=8.5, color="#1b4f72", weight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#1b4f72", lw=1.4))
    ax.plot([], [], "o", ms=7, mfc="white", mec="#555", label="SAE を学習した 8 か所（位置の比較）")
    ax.plot([], [], "o", ms=8, mfc=INT_C["color"], mec="white", label="① 色（SAE: 色の特徴を swap / DiffMean: 色の平均差）")
    ax.plot([], [], "o", ms=8, mfc=INT_C["xy"], mec="white", label="② x/y 移動（SAE: 方向の特徴に加算 / DiffMean: 移動方向の平均差）")
    ax.plot([], [], "o", ms=8, mfc=INT_C["traj"], mec="white", label="③ 軌道（SAE: 軌道の特徴を swap / DiffMean: 色平均の差）")
    ax.legend(loc="lower left", fontsize=8, frameon=True)


def main():
    ensure(FIGS)
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_ylim(1.4, 11.0)
    draw(ax)
    ax.set_title("ACT のモデル構成", fontsize=12)
    plt.tight_layout(); plt.savefig(FIGS / "act_arch.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.set_ylim(-0.2, 12.0)
    draw(ax, interventions=True)
    ax.set_title("介入した位置（介入はキューブを持ち上げた後のみ）", fontsize=12)
    plt.tight_layout(); plt.savefig(FIGS / "act_intervention.png", dpi=150); plt.close()
    print("saved", FIGS / "act_arch.png", FIGS / "act_intervention.png")

if __name__ == "__main__":
    main()
