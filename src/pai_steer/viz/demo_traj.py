"""4パターンのデモの手先 xy 軌跡を真上から描く (artifacts/samples/samples.hdf5 から).

出力 artifacts/samples/trajectories_topdown.png
"""
import h5py
import numpy as np

from pai_steer.common import COLOR_JP, COLORS, SAMPLES, base_parser, load_config, setup_mpl
from pai_steer.env.layout import RGB, Layout

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def plot(L, eps):
    """eps: {色: dict(eef=[T,3], cube0=[3], placed_box=str)}."""
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for c in COLORS:
        x, y = L.box_xy[c]
        h = L.inner / 2
        ax.add_patch(Rectangle((y - h, x - h), L.inner, L.inner, fill=False, ec=RGB[c], lw=3))
        # 箱の名前は軌跡と重ならないよう箱の下辺の内側に置く
        ax.text(y, x - h + 0.012, f"{COLOR_JP[c]}の箱", ha="center", va="bottom", color=RGB[c], fontsize=11)
    for c in COLORS:
        e = eps[c]["eef"]
        ax.plot(e[:, 1], e[:, 0], c=RGB[c], lw=2,
                label=f"{COLOR_JP[c]}キューブ → {COLOR_JP.get(eps[c]['placed_box'], eps[c]['placed_box'])}の箱")
    cp = eps["red"]["cube0"]
    s = L.cube_size
    ax.add_patch(Rectangle((cp[1] - s / 2, cp[0] - s / 2), s, s, fc="0.4", ec="k"))
    ax.text(cp[1], cp[0] - 0.045, "キューブ初期位置", ha="center", fontsize=11)
    pts = np.array(list(L.box_xy.values()) + [L.cube_xy])
    lo, hi = pts.min(0) - 0.12, pts.max(0) + 0.12
    ax.set_xlim(hi[1], lo[1])     # 真上から見て y は左向き (画像の左 = +y)
    ax.set_ylim(lo[0], hi[0])
    ax.set_aspect("equal"); ax.set_xlabel("y [m]（左右方向, 左が +y）"); ax.set_ylabel("x [m]（前後方向, 上が奥）")
    ax.set_title("4パターンの手先の xy 軌跡（真上から）"); ax.legend(loc="upper right", fontsize=10); ax.grid(alpha=.3)
    plt.savefig(SAMPLES / "trajectories_topdown.png", dpi=130, bbox_inches="tight"); plt.close()


def main():
    args = base_parser(__doc__).parse_args()
    L = Layout(load_config(args.config)["env"])
    eps = {}
    with h5py.File(SAMPLES / "samples.hdf5") as f:
        for g in f["data"].values():
            eps[g.attrs["cube_color"]] = dict(eef=g["obs/robot0_eef_pos"][:], cube0=g["obs/cube_pos"][0],
                                              placed_box=g.attrs["placed_box"])
    plot(L, eps)
    print("saved", SAMPLES / "trajectories_topdown.png")


if __name__ == "__main__":
    main()
