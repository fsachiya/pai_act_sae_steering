"""x/y 方向の特徴の刺激: 強さ → 移動量 と, probing (相関) と steering (実際の動き) の一致.

1. 強さ → 移動量: 4方向 (+x, -x, +y, -y) の特徴群 (SAE: 上位3特徴に α×典型活性, DiffMean: ±v/2·α) を
   行き先とは無関係に単独で刺激 (gate=lift). 介入なし (同じ色・seed) からの
     - チャンク変化 (ロールアウトなし, 運搬中の入力 256 個で行動チャンクの xy 指令合計の変化)
     - 離した位置のずれ (グリッパーを開いた時点の手先 xy), 最終位置のずれ (キューブ xy)
2. probing vs steering (単一特徴): 各方向の上位5特徴について
     probing の向き  = 運搬中の活性と将来の手先変位 (dx, dy) の相関ベクトル
     steering の向き = 刺激したときのチャンク変化 (行動寄与) と, ロールアウトでの離した位置のずれ
   の角度のずれと cos.
出力 artifacts/results/xy_dose.csv, xy_single.csv, 図 artifacts/figs/xy_dose.png, xy_probe_vs_steer.png
"""
import json

import numpy as np
import pandas as pd
import torch

from pai_steer.common import CKPT, COLORS, FEATURES, FIGS, RESULTS, SAE_DIR, base_parser, ensure, load_config, setup_mpl
from pai_steer.data.dataset import ChunkDataset
from pai_steer.policy.act import load_policy
from pai_steer.steer.runner import build_steer, run_jobs

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402

SITE = "decoder_out"
DIRS = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1)}
ALPHAS = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
SEEDS = 3
OUT_MAX = 0.05


def group_spec(method, d, alpha, xy, dm, n=3):
    if alpha == 0:
        return None
    if method == "SAE":
        lst = xy[d][:n]
        return dict(site=SITE, mode="add", feats=[r["feat"] for r in lst], alpha=[alpha * r["scale"] for r in lst],
                    sae_path=str(SAE_DIR / f"sae_{SITE}.pt"))
    ax, sg = DIRS[d]
    return dict(site=SITE, mode="vector", vec=sg * dm["xy"][ax] / 2, alpha=alpha)


def single_spec(r, alpha):
    return dict(site=SITE, mode="add", feats=[r["feat"]], alpha=[alpha * r["scale"]], sae_path=str(SAE_DIR / f"sae_{SITE}.pt"))


def release_xy(r):
    ph = np.asarray(r["phase"])
    idx = np.where(ph == "post")[0]
    t = idx[0] if len(idx) else len(ph) - 1
    return np.asarray(r["eef"])[t, :2]


def chunk_delta(model, stats, ds, idx, spec):
    b = ds.batch(idx)
    inp = [torch.from_numpy(b[k]) for k in ["robot", "cube", "color"]]
    with torch.no_grad():
        base = model(*inp)[0].numpy()
        st = build_steer(model, spec)
        st.enabled = True
        try:
            out = model(*inp)[0].numpy()
        finally:
            st.remove()
    f = lambda a: np.clip(a * stats["act_std"] + stats["act_mean"], -1, 1)[..., :2].sum(1) * OUT_MAX
    return (f(out) - f(base)).mean(0)


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(RESULTS, FIGS)
    xy = json.load(open(FEATURES / f"xy_features_{SITE}.json"))["features"]
    dm = np.load(FEATURES / f"diffmean_{SITE}.npz")
    st = np.load(FEATURES / f"dec_feature_stats_{SITE}.npz")
    model, mcfg, stats = load_policy(args.ckpt)
    ds = ChunkDataset(chunk=mcfg["act"]["chunk"], stats=stats)
    idx = np.random.default_rng(0).choice(np.where(ds.phase == 4)[0], 256, replace=False)

    # ---- ロールアウト ----
    jobs = [dict(kind="base", color=c, dst=c, seed=s, record_traj=True) for c in COLORS for s in range(SEEDS)]
    for method in ["SAE", "DiffMean"]:
        for d in DIRS:
            for a in ALPHAS[1:]:
                spec = group_spec(method, d, a, xy, dm)
                jobs += [dict(kind="group", method=method, d=d, alpha=a, color=c, dst=c, seed=s, record_traj=True, steer=spec)
                         for c in COLORS for s in range(SEEDS)]
    singles = [(d, r) for d in DIRS for r in xy[d][:5]]
    for d, r in singles:
        for a in [1.0, 2.0]:
            jobs += [dict(kind="single", d=d, feat=r["feat"], alpha=a, color=c, dst=c, seed=s, record_traj=True,
                          steer=single_spec(r, a)) for c in COLORS for s in range(SEEDS)]
    print(len(jobs), "episodes")
    res = run_jobs(jobs, args.ckpt, cfg)
    base = {(r["job"]["color"], r["job"]["seed"]): r for r in res if r["job"]["kind"] == "base"}
    rows = []
    for r in res:
        j = r["job"]
        if j["kind"] == "base":
            continue
        b = base[(j["color"], j["seed"])]
        drel, dfin = release_xy(r) - release_xy(b), np.asarray(r["final_pos"])[:2] - np.asarray(b["final_pos"])[:2]
        rows.append(dict(kind=j["kind"], method=j.get("method", "SAE"), d=j["d"], feat=j.get("feat", -1), alpha=j["alpha"],
                         color=j["color"], seed=j["seed"], rel_dx=drel[0], rel_dy=drel[1], fin_dx=dfin[0], fin_dy=dfin[1],
                         placed_box=r["placed_box"], grasped=r["grasped"]))
    df = pd.DataFrame(rows)

    # ---- チャンク変化 (ロールアウトなし) ----
    crow = []
    for method in ["SAE", "DiffMean"]:
        for d in DIRS:
            for a in ALPHAS:
                cd = np.zeros(2) if a == 0 else chunk_delta(model, stats, ds, idx, group_spec(method, d, a, xy, dm))
                crow.append(dict(method=method, d=d, alpha=a, chunk_dx=cd[0], chunk_dy=cd[1]))
    cdf = pd.DataFrame(crow)
    g = df[df.kind == "group"].groupby(["method", "d", "alpha"])[["rel_dx", "rel_dy", "fin_dx", "fin_dy"]].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    dose = cdf.merge(g.reset_index(), on=["method", "d", "alpha"], how="left").fillna(0)
    dose.to_csv(RESULTS / "xy_dose.csv", index=False)

    # 単一特徴: probing (相関) vs steering (チャンク寄与, ロールアウト)
    srow = []
    for d, r in singles:
        f = r["feat"]
        for a in [1.0, 2.0]:
            s = df[(df.kind == "single") & (df.feat == f) & (df.alpha == a)]
            cd = chunk_delta(model, stats, ds, idx, single_spec(r, a))
            srow.append(dict(d=d, feat=f, alpha=a, probe_x=st["cx"][f], probe_y=st["cy"][f], chunk_dx=cd[0], chunk_dy=cd[1],
                             roll_dx=s.rel_dx.mean(), roll_dy=s.rel_dy.mean()))
    sdf = pd.DataFrame(srow)
    ang = lambda x, y: np.degrees(np.arctan2(y, x))
    for k in ["chunk", "roll"]:
        sdf[f"cos_probe_{k}"] = (sdf.probe_x * sdf[f"{k}_dx"] + sdf.probe_y * sdf[f"{k}_dy"]) / (
            np.hypot(sdf.probe_x, sdf.probe_y) * np.hypot(sdf[f"{k}_dx"], sdf[f"{k}_dy"]) + 1e-12)
        sdf[f"angdiff_{k}"] = ((ang(sdf[f"{k}_dx"], sdf[f"{k}_dy"]) - ang(sdf.probe_x, sdf.probe_y) + 180) % 360) - 180
    sdf.to_csv(RESULTS / "xy_single.csv", index=False)
    pd.set_option("display.width", 220)
    print(sdf.round(3).to_string())
    plot(dose, sdf)


def plot(dose, sdf):
    cols = {"+x": "C0", "-x": "C1", "+y": "C2", "-y": "C3"}
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for ri, method in enumerate(["SAE", "DiffMean"]):
        for ci, (pre, title) in enumerate([("chunk_d", "1回の推論の行動チャンク (ロールアウトなし)"),
                                          ("rel_d", "離した位置のずれ (ロールアウト)"), ("fin_d", "キューブの最終位置のずれ")]):
            a = axs[ri, ci]
            for d in DIRS:
                g = dose[(dose.method == method) & (dose.d == d)].sort_values("alpha")
                ax_i = DIRS[d][0]
                main = f"{pre}{'xy'[ax_i]}" + ("" if pre == "chunk_d" else "_mean")
                off = f"{pre}{'yx'[ax_i]}" + ("" if pre == "chunk_d" else "_mean")
                a.plot(g.alpha, g[main], marker="o", c=cols[d], label=f"{d}: 狙った軸 ({'xy'[ax_i]})")
                a.plot(g.alpha, g[off], marker="x", ls="--", c=cols[d], alpha=.6, label=f"{d}: 直交する軸 ({'yx'[ax_i]})")
                if pre != "chunk_d":
                    sd = g[f"{pre}{'xy'[ax_i]}_std"]
                    a.fill_between(g.alpha, g[main] - sd, g[main] + sd, color=cols[d], alpha=.12)
            a.axhline(0, c="k", lw=.5)
            for v in (0.14, -0.14):
                a.axhline(v, c="gray", ls=":", lw=.8)
            a.set_xlabel("強さ α"); a.set_ylabel("移動量 [m]"); a.set_title(f"{method}: {title}", fontsize=10)
            if ri == 0 and ci == 0:
                a.legend(fontsize=7, ncol=2)
            if pre != "chunk_d":
                a.set_ylim(-0.5, 0.5)
    fig.suptitle("x/y 方向の特徴の刺激: 強さと移動量 (decoder_out, 4色 × seed 3 の平均 ± 標準偏差, 点線 = 隣の箱までの距離 14cm)", fontsize=11)
    plt.tight_layout(); plt.savefig(FIGS / "xy_dose.png", dpi=120); plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    s1 = sdf[sdf.alpha == 1.0]
    for ax, (k, t) in zip(axs[:2], [("chunk", "行動チャンクの変化 (ロールアウトなし)"), ("roll", "離した位置のずれ (ロールアウト)")]):
        for _, r in s1.iterrows():
            pv = np.array([r.probe_x, r.probe_y]); pv = pv / (np.linalg.norm(pv) + 1e-12)
            sv = np.array([r[f"{k}_dx"], r[f"{k}_dy"]]); sv = sv / (np.linalg.norm(sv) + 1e-12)
            c = cols[r.d]
            ax.annotate("", xy=pv, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=1.2, ls="--", alpha=.6))
            ax.annotate("", xy=sv, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
        for d, c in cols.items():
            ax.plot([], [], c=c, lw=2, label=f"{d} の特徴: steering")
        ax.plot([], [], c="k", ls="--", label="probing (相関の向き)")
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal"); ax.axhline(0, c="k", lw=.4); ax.axvline(0, c="k", lw=.4)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"単一特徴: probing (破線) と steering (実線) の向き\n{t}, α=1", fontsize=10)
    a = axs[2]
    x = np.arange(4)
    for j, (k, lab) in enumerate([("chunk", "行動チャンク"), ("roll", "ロールアウト")]):
        m = s1.groupby("d", sort=False)[f"cos_probe_{k}"].mean().reindex(DIRS.keys())
        a.bar(x + (j - .5) * .38, m.values, .38, label=lab)
        for _, r in s1.iterrows():
            a.scatter(list(DIRS).index(r.d) + (j - .5) * .38, r[f"cos_probe_{k}"], c="k", s=8, zorder=5)
    a.set_xticks(x); a.set_xticklabels([f"{d} の特徴" for d in DIRS]); a.axhline(0, c="k", lw=.5); a.set_ylim(-1.05, 1.05)
    a.set_ylabel("cos(probing の向き, steering の向き)"); a.legend(fontsize=8)
    a.set_title("probing と steering の一致 (各方向の上位5特徴, 点 = 各特徴)", fontsize=10)
    plt.tight_layout(); plt.savefig(FIGS / "xy_probe_vs_steer.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
