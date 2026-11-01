"""x/y 移動特徴を「組み合わせ」で選ぶ: 直交する成分を打ち消して軸に沿った動きを作る.

1個で x 軸に沿う特徴は SAE に無い (±x は probing・steering とも ±15° 以内の特徴が 0) ので,
運搬中に発火する特徴の行動寄与 Δa_i (7次元: 位置3, 回転3, グリッパー1, 典型活性ぶん, 摂動法) を非負の重みで足し,
  Σ w_i Δa_i ≈ (目標の軸の単位ベクトル × C, それ以外の6成分 = 0),   w_i ≥ 0
(最初は位置だけを条件にしたが, −x の組み合わせがグリッパーを開く成分を持ち運搬中にキューブを落としたため, 7次元すべてを条件にした)
を NNLS で解く (解は高々3特徴の組み合わせになる). 重みが極端に大きくならないよう, 候補は行動寄与が
|Δa_xy| ≥ MIN_DA の特徴に限る. 介入は特徴 i に α · w_i · 典型活性_i を足す (α=1 で行動寄与の合計が C).
その後 (1) 強さと移動量 (4方向単独, 4色 × seed 3) と (2) 行き先の箱への到達率 (4色 × 他の3色 × seed 5, 強さ6段階) を評価.
出力 artifacts/features/xy_combo_decoder_out.json, artifacts/results/xy_combo_dose.csv, xy_combo_reach.csv, 図 artifacts/figs/xy_combo.png
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from pai_steer.common import CKPT, COLORS, FEATURES, FIGS, RESULTS, SAE_DIR, base_parser, ensure, load_config, setup_mpl
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs, wilson
from pai_steer.steer.specs import box_offset, box_span
from pai_steer.steer.sweep import to_df

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402

SITE = "decoder_out"
C = 0.3          # α=1 での行動寄与の合計 (行動の単位)
MIN_DA = 0.05    # 候補にする特徴の行動寄与の下限
DIRS = {"+x": np.eye(7)[0], "-x": -np.eye(7)[0], "+y": np.eye(7)[1], "-y": -np.eye(7)[1]}
DIM_W = np.array([1, 1, 1, 1, 1, 1, 1.0])   # 各成分の重み (行動はどれも [-1, 1] の範囲)


def select(st):
    dA, fire, scale = st["dA"], st["fire"], st["scale"]
    act = np.where((fire > 0.01) & (np.linalg.norm(dA[:, :2], axis=1) >= MIN_DA))[0]
    A = dA[act].T * DIM_W[:, None]                           # [7, n]
    out = {}
    for d, u in DIRS.items():
        w, _ = nnls(A, u * C * DIM_W)
        keep = np.where(w > 1e-4)[0]
        v = dA[act][keep].T @ w[keep]
        out[d] = dict(feats=act[keep].tolist(), weights=w[keep].round(4).tolist(), scale=scale[act[keep]].round(4).tolist(),
                      achieved=v.round(4).tolist(), off_axis_deg=float(np.degrees(np.arctan2(np.linalg.norm(v - (v @ u) * u), v @ u))),
                      residual=float(np.linalg.norm(v - u * C)))
        print(f"{d}: 候補 {len(act)} 個中 特徴 {len(keep)} 個 {act[keep].tolist()}, 重み {w[keep].round(2).tolist()}, "
              f"作れる行動寄与 (7次元) {v.round(3).tolist()}, "
              f"軸からのずれ {out[d]['off_axis_deg']:.1f}°")
    return out


def dir_spec(combo, d, alpha):
    c = combo[d]
    return dict(site=SITE, mode="add", feats=c["feats"], alpha=[alpha * w * s for w, s in zip(c["weights"], c["scale"])],
                sae_path=str(SAE_DIR / f"sae_{SITE}.pt"))


def reach_spec(combo, L, src, dst, alpha):
    if src == dst:
        return None
    off, span = box_offset(L, src, dst), box_span(L)
    feats, alphas = [], []
    for ax, pos, neg in [(0, "+x", "-x"), (1, "+y", "-y")]:
        if abs(off[ax]) < 1e-6:
            continue
        sp = dir_spec(combo, pos if off[ax] > 0 else neg, alpha * abs(off[ax]) / span[ax])
        feats += sp["feats"]; alphas += sp["alpha"]
    return dict(site=SITE, mode="add", feats=feats, alpha=alphas, sae_path=str(SAE_DIR / f"sae_{SITE}.pt"))


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(RESULTS, FIGS, FEATURES)
    L = Layout(cfg["env"])
    combo = select(np.load(FEATURES / f"dec_feature_stats_{SITE}.npz"))
    json.dump(combo, open(FEATURES / f"xy_combo_{SITE}.json", "w"), indent=1)

    from pai_steer.analysis.xy_dose import ALPHAS, SEEDS, chunk_delta, release_xy
    from pai_steer.data.dataset import ChunkDataset
    from pai_steer.policy.act import load_policy

    jobs = [dict(kind="base", color=c, dst=c, seed=s, record_traj=True) for c in COLORS for s in range(SEEDS)]
    for d in DIRS:
        for a in ALPHAS[1:]:
            jobs += [dict(kind="dose", d=d, alpha=a, color=c, dst=c, seed=s, record_traj=True, steer=dir_spec(combo, d, a))
                     for c in COLORS for s in range(SEEDS)]
    strengths = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    for a in strengths:
        jobs += [dict(kind="reach", alpha=a, color=s_, dst=t_, strength=a, seed=sd, gate="lift", steer=reach_spec(combo, L, s_, t_, a))
                 for s_ in COLORS for t_ in COLORS if s_ != t_ for sd in range(5)]
    print(len(jobs), "episodes")
    res = run_jobs(jobs, args.ckpt, cfg)

    base = {(r["job"]["color"], r["job"]["seed"]): r for r in res if r["job"]["kind"] == "base"}
    rows = []
    for r in res:
        j = r["job"]
        if j["kind"] != "dose":
            continue
        b = base[(j["color"], j["seed"])]
        dr = release_xy(r) - release_xy(b)
        rows.append(dict(d=j["d"], alpha=j["alpha"], rel_dx=dr[0], rel_dy=dr[1]))
    dose = pd.DataFrame(rows).groupby(["d", "alpha"]).agg(["mean", "std"])
    dose.columns = ["_".join(c) for c in dose.columns]
    dose = dose.reset_index()
    model, mcfg, stats = load_policy(args.ckpt)
    ds = ChunkDataset(chunk=mcfg["act"]["chunk"], stats=stats)
    idx = np.random.default_rng(0).choice(np.where(ds.phase == 4)[0], 256, replace=False)
    ch = [dict(d=d, alpha=a, **dict(zip(["chunk_dx", "chunk_dy"], chunk_delta(model, stats, ds, idx, dir_spec(combo, d, a)))))
          for d in DIRS for a in ALPHAS[1:]]
    dose = dose.merge(pd.DataFrame(ch), on=["d", "alpha"])
    dose.to_csv(RESULTS / "xy_combo_dose.csv", index=False)

    reach = to_df([r for r in res if r["job"]["kind"] == "reach"], L)
    reach.to_csv(RESULTS / "xy_combo_reach.csv", index=False)
    from pai_steer.analysis.direction_study import direction
    reach["dir"] = [direction(L, s, t) for s, t in zip(reach.src, reach.dst)]
    pd.set_option("display.width", 220)
    print(dose.round(3).to_string())
    print(reach.groupby("strength").agg(reached=("reached", "mean"), stayed=("stayed", "mean"),
                                       nobox=("placed_box", lambda x: x.isna().mean())).round(2).to_string())
    print(reach.pivot_table(index="strength", columns="dir", values="reached", aggfunc="mean").round(2).to_string())
    plot(dose, reach)


def plot(dose, reach):
    old = pd.read_csv(RESULTS / "xy_dose.csv")
    cols = {"+x": "C0", "-x": "C1", "+y": "C2", "-y": "C3"}
    fig, axs = plt.subplots(1, 4, figsize=(24, 5.2))
    for ai, (pre, title) in enumerate([("chunk_d", "行動チャンク (ロールアウトなし)"), ("rel_d", "離した位置のずれ (ロールアウト)")]):
        a = axs[ai]
        for d in DIRS:
            ax_i = 0 if "x" in d else 1
            suf = "" if pre == "chunk_d" else "_mean"
            g = dose[dose.d == d].sort_values("alpha")
            a.plot([0] + g.alpha.tolist(), [0] + g[f"{pre}{'xy'[ax_i]}{suf}"].tolist(), marker="o", c=cols[d], label=f"{d}: 狙った軸")
            a.plot([0] + g.alpha.tolist(), [0] + g[f"{pre}{'yx'[ax_i]}{suf}"].tolist(), marker="x", ls="--", c=cols[d], alpha=.6,
                   label=f"{d}: 直交する軸")
        for v in (0.14, -0.14):
            a.axhline(v, c="gray", ls=":", lw=.8)
        a.axhline(0, c="k", lw=.5); a.set_xlabel("強さ α"); a.set_ylabel("移動量 [m]"); a.set_ylim(-0.5, 0.5)
        a.set_title(f"組み合わせ版: {title}", fontsize=10)
        if ai == 0:
            a.legend(fontsize=7, ncol=2)
    # 直交軸への漏れの比較 (チャンク, α=0.5)
    a = axs[2]
    x = np.arange(4); w = .38
    for j, (lab, df_, key) in enumerate([("上位3特徴 (従来)", old[(old.method == "SAE") & (old.alpha == 0.5)], "old"),
                                        ("組み合わせ", dose[dose.alpha == 0.5], "new")]):
        vals = []
        for d in DIRS:
            r = df_[df_.d == d].iloc[0]
            m, o = (r.chunk_dx, r.chunk_dy) if "x" in d else (r.chunk_dy, r.chunk_dx)
            vals.append(np.degrees(np.arctan2(abs(o), abs(m))))
        a.bar(x + (j - .5) * w, vals, w, label=lab)
    a.set_xticks(x); a.set_xticklabels(list(DIRS)); a.set_ylabel("狙った軸からのずれ [度]")
    a.set_title("行動チャンクの向きのずれ (α=0.5)", fontsize=10); a.legend(fontsize=8)
    a = axs[3]
    g = reach.groupby(["strength", "kind"]).reached.mean().unstack("kind")
    for k in ["adj_x", "adj_y", "diag"]:
        if k in g:
            a.plot(g.index, g[k], marker="o", label={"adj_x": "隣 (x のみ)", "adj_y": "隣 (y のみ)", "diag": "対角"}[k])
    a.set_ylim(-0.03, 1.03); a.set_xlabel("強さ α"); a.set_ylabel("行き先の箱への到達率"); a.legend(fontsize=8)
    a.set_title("組み合わせ版: 行き先の箱への到達率 (各点 = 4色 × 2行き先 × seed 5)", fontsize=10)
    plt.tight_layout(); plt.savefig(FIGS / "xy_combo.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
