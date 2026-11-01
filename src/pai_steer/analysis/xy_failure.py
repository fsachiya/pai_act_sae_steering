"""組み合わせ版 x/y 介入で, 対角と -y がなぜ届かないか.

1. 失敗の仕方: 軸ごとに較正した強さ (xy_combo_calib, 倍率 1.0) で全 12 対 × seed 5 をロールアウトし, 各エピソードで
     下ろし始め = 運搬中に手先の z が運搬高さ (机 + 0.15) - 3cm を下回った最初の時刻 と その時の手先 xy
     離した位置 = グリッパーを開いた時の手先 xy
   を, 元の箱・行き先の箱と比べる. 下ろし始めまでの時間 (持ち上げてから) も記録.
2. 対角の 2 次元グリッド: 4 つの対角の対について, x と y の強さの倍率をそれぞれ {0.6, 0.8, 1.0, 1.2, 1.4} で振る.
   実際の離した位置のずれと, 1軸ずつ入れたときのずれの和 (加法性) を比べる.
出力 artifacts/results/xy_failure_episodes.csv, xy_diag_grid.csv, 図 artifacts/figs/xy_failure.png
"""
import json

import numpy as np
import pandas as pd

from pai_steer.common import CKPT, COLOR_JP, COLORS, FEATURES, FIGS, RESULTS, SAE_DIR, base_parser, ensure, load_config, setup_mpl
from pai_steer.env.layout import RGB, Layout
from pai_steer.steer.runner import run_jobs
from pai_steer.steer.specs import box_offset
from pai_steer.analysis.xy_combo import SITE, dir_spec
from pai_steer.analysis.xy_combo_calib import slopes
from pai_steer.analysis.direction_study import direction

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402

SCALES = [0.6, 0.8, 1.0, 1.2, 1.4]


def axis_spec(combo, sl, L, src, dst, kx, ky):
    off = box_offset(L, src, dst)
    feats, alphas = [], []
    for ax, pos, neg, k in [(0, "+x", "-x", kx), (1, "+y", "-y", ky)]:
        if abs(off[ax]) < 1e-6 or k == 0:
            continue
        dr = pos if off[ax] > 0 else neg
        sp = dir_spec(combo, dr, k * abs(off[ax]) / sl[dr])
        feats += sp["feats"]; alphas += sp["alpha"]
    if not feats:
        return None
    return dict(site=SITE, mode="add", feats=feats, alpha=alphas, sae_path=str(SAE_DIR / f"sae_{SITE}.pt"))


def events(r, L, cfg):
    eef, ph = np.asarray(r["eef"]), np.asarray(r["phase"])
    carry = np.where(ph == "carry")[0]
    t_lift = carry[0] if len(carry) else 0
    thr = L.table_top + cfg["expert"]["transport_height"] - 0.03
    low = [t for t in carry if t > t_lift + 5 and eef[t, 2] < thr and eef[t - 5:t, 2].max() >= thr] if len(carry) else []
    t_low = low[0] if low else None
    post = np.where(ph == "post")[0]
    t_rel = post[0] if len(post) else len(ph) - 1
    return dict(t_lift=t_lift, t_lower=t_low if t_low is not None else -1, lower_x=eef[t_low, 0] if t_low else np.nan,
                lower_y=eef[t_low, 1] if t_low else np.nan, t_release=t_rel, rel_x=eef[t_rel, 0], rel_y=eef[t_rel, 1],
                lowered=t_low is not None)


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(RESULTS, FIGS)
    L = Layout(cfg["env"])
    combo = json.load(open(FEATURES / f"xy_combo_{SITE}.json"))
    sl = slopes()
    pairs = [(a, b) for a in COLORS for b in COLORS if a != b]
    diag = [(a, b) for a, b in pairs if "&" in direction(L, a, b)]
    jobs = [dict(study="base", color=a, dst=a, seed=s, record_traj=True) for a in COLORS for s in range(5)]
    jobs += [dict(study="calib", color=a, dst=b, seed=s, record_traj=True, steer=axis_spec(combo, sl, L, a, b, 1.0, 1.0))
             for a, b in pairs for s in range(5)]
    for a, b in diag:
        for kx in [0] + SCALES:
            for ky in [0] + SCALES:
                if kx == ky == 0:
                    continue
                jobs += [dict(study="grid", color=a, dst=b, kx=kx, ky=ky, seed=s, record_traj=True,
                              steer=axis_spec(combo, sl, L, a, b, kx, ky)) for s in range(3)]
    print(len(jobs), "episodes")
    res = run_jobs(jobs, args.ckpt, cfg)
    rows = []
    for r in res:
        j = r["job"]
        e = events(r, L, cfg)
        rows.append(dict(study=j["study"], src=j["color"], dst=j["dst"], dir=direction(L, j["color"], j["dst"]) if j["dst"] != j["color"] else "base",
                         kx=j.get("kx", 1.0), ky=j.get("ky", 1.0), seed=j["seed"], placed_box=r["placed_box"],
                         reached=r["placed_box"] == j["dst"], length=r["length"], **e))
    df = pd.DataFrame(rows)
    base = df[df.study == "base"].groupby("src")[["t_lower", "t_release", "rel_x", "rel_y"]].mean()
    df.to_csv(RESULTS / "xy_failure_episodes.csv", index=False)

    c = df[df.study == "calib"].copy()
    c["src_y"] = [L.box_xy[s][1] for s in c.src]; c["dst_y"] = [L.box_xy[d][1] for d in c.dst]
    c["src_x"] = [L.box_xy[s][0] for s in c.src]; c["dst_x"] = [L.box_xy[d][0] for d in c.dst]
    c["lower_minus_src_y"] = c.lower_y - c.src_y
    c["delay"] = c.t_lower - c.t_lift - np.array([base.loc[s, "t_lower"] - 0 for s in c.src]) + c.t_lift * 0
    print("介入なしの下ろし始め時刻 (平均):", base.t_lower.round(1).to_dict())
    pd.set_option("display.width", 220)
    print(c.groupby("dir").agg(reached=("reached", "mean"), lowered=("lowered", "mean"), t_lower=("t_lower", "mean"),
                               lower_y=("lower_y", "mean"), src_y=("src_y", "first"), dst_y=("dst_y", "first"),
                               rel_y=("rel_y", "mean"), lower_x=("lower_x", "mean"), rel_x=("rel_x", "mean"),
                               dst_x=("dst_x", "first"), length=("length", "mean")).round(3).to_string())

    # 加法性: 1軸だけ (kx>0, ky=0 / kx=0, ky>0) のずれの和 と 両方入れたずれ (介入なしの同じ色・seed からの差)
    g = df[df.study == "grid"].copy()
    bxy = df[df.study == "base"].set_index(["src", "seed"])[["rel_x", "rel_y"]]
    g["dx"] = [r.rel_x - bxy.loc[(r.src, r.seed % 5), "rel_x"] for r in g.itertuples()]
    g["dy"] = [r.rel_y - bxy.loc[(r.src, r.seed % 5), "rel_y"] for r in g.itertuples()]
    m = g.groupby(["src", "dst", "kx", "ky"]).agg(reached=("reached", "mean"), dx=("dx", "mean"), dy=("dy", "mean"),
                                                   lowered=("lowered", "mean")).reset_index()
    add = []
    for (a, b), h in m.groupby(["src", "dst"]):
        for _, r in h[(h.kx > 0) & (h.ky > 0)].iterrows():
            sx = h[(h.kx == r.kx) & (h.ky == 0)]; sy = h[(h.kx == 0) & (h.ky == r.ky)]
            add.append(dict(src=a, dst=b, kx=r.kx, ky=r.ky, reached=r.reached, dx=r.dx, dy=r.dy,
                            pred_dx=sx.dx.iloc[0] + sy.dx.iloc[0], pred_dy=sx.dy.iloc[0] + sy.dy.iloc[0]))
    add = pd.DataFrame(add)
    add.to_csv(RESULTS / "xy_diag_grid.csv", index=False)
    print(add.groupby(["src", "dst"]).agg(best=("reached", "max")).to_string())
    for (a, b), h in add.groupby(["src", "dst"]):
        piv = h.pivot_table(index="ky", columns="kx", values="reached")
        print(f"\n{a}→{b} ({direction(L, a, b)}) 到達率 (行 = y の倍率, 列 = x の倍率):"); print(piv.round(2).to_string())
    print("\n加法性: 実際のずれ vs 1軸ずつの和  dx の相関", np.corrcoef(add.dx, add.pred_dx)[0, 1].round(3),
          " dy の相関", np.corrcoef(add.dy, add.pred_dy)[0, 1].round(3),
          " 比 (実際/和) dx", (add.dx / add.pred_dx).median().round(2), " dy", (add.dy / add.pred_dy).median().round(2))
    plot(L, c, add, df)


def plot(L, c, add, df):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.5))
    a = axs[0]
    for col in COLORS:
        x, y = L.box_xy[col]
        a.add_patch(plt.Rectangle((y - .05, x - .05), .1, .1, fill=False, ec=RGB[col], lw=2.5))
    kinds = {"+y": "o", "-y": "v", "+x": ">", "-x": "<"}
    for _, r in c.iterrows():
        mk = kinds.get(r.dir, "*")
        col = RGB[r.dst]
        if r.lowered:
            a.scatter(r.lower_y, r.lower_x, marker=mk, s=40, facecolor="none", edgecolor=col)
        a.scatter(r.rel_y, r.rel_x, marker=mk, s=30, color=col, alpha=.8)
    a.set_xlim(0.35, -0.05); a.set_ylim(-0.12, 0.27); a.set_aspect("equal")
    a.set_xlabel("y [m] (左が +y)"); a.set_ylabel("x [m]")
    a.set_title("軸ごとに較正した介入: 下ろし始めた位置 (白抜き) と離した位置 (塗り)\n色 = 行き先の箱, 記号 = 方向 (○+y ▽-y ▷+x ◁-x ☆対角)", fontsize=9)
    a = axs[1]
    t = c.groupby("dir").agg(lowered=("lowered", "mean"), reached=("reached", "mean"))
    t.plot.bar(ax=a); a.set_ylim(0, 1.05); a.set_title("方向ごとの「下ろし始めた率」と到達率", fontsize=10); a.set_xlabel("")
    a = axs[2]
    if len(add):
        a.scatter(add.pred_dx, add.dx, s=10, label="x")
        a.scatter(add.pred_dy, add.dy, s=10, label="y")
        lim = [-0.5, 0.5]
        a.plot(lim, lim, c="k", lw=.6); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel("1軸ずつ入れたときのずれの和 [m]"); a.set_ylabel("両軸を同時に入れたときのずれ [m]")
        a.set_title("対角: 2軸の効果は足し算になるか", fontsize=10); a.legend()
    plt.tight_layout(); plt.savefig(FIGS / "xy_failure.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
