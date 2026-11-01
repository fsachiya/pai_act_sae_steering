"""図: E0 棒グラフ, E1 行列ヒートマップ, E2 スイープ曲線, E3 上面軌跡.

  --e3 で E3 用ロールアウト (同じ seed で 介入なし + 行き先3通り, P1 と P2, P2 の β 連続変化) を実行して保存・描画.
"""
import json

import numpy as np
import pandas as pd

from pai_steer.common import (CKPT, COLOR_JP, COLORS, FIGS, RESULTS, ROLLOUTS, base_parser, ensure, load_config,
                              setup_mpl)

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from pai_steer.env.layout import RGB, Layout  # noqa: E402
from pai_steer.steer.runner import wilson  # noqa: E402

KIND_JP = {"adj_x": "隣 (x のみ)", "adj_y": "隣 (y のみ)", "diag": "対角 (x と y)"}
PAT_JP = {"p1": "P1 (SAE): xy 移動特徴", "p2": "P2 (3トークン共通 SAE): 色特徴 swap",
          "p2_colorsae": "P2 (色の SAE: color トークン専用): 色特徴 swap", "p3": "P3 (軌道の SAE): 軌道パターン特徴 swap",
          "p1_dm": "P1 (DiffMean): 将来変位の平均差", "p2_dm": "P2 (DiffMean): 色ごとの平均差",
          "p3_dm": "P3 (DiffMean): 位置 k ごとの色平均差"}


def pattern_of(stem):
    parts = stem.split("_")
    if "colorsae" in parts:
        return "p2_colorsae"
    return f"{parts[1]}_dm" if len(parts) > 2 and parts[2] == "dm" else parts[1]


def plot_e0():
    for f in sorted(RESULTS.glob("e0*.csv")):
        df = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        xs = np.arange(4)
        for j, (col, lab) in enumerate([("success", "成功率"), ("grasped", "把持成功率")]):
            v = [wilson(df[df.color == c][col].sum(), (df.color == c).sum()) for c in COLORS]
            p = np.array([x[0] for x in v])
            ax.bar(xs + (j - .5) * .38, p, .38, yerr=[p - [x[1] for x in v], np.array([x[2] for x in v]) - p],
                   capsize=3, label=lab, color=[RGB[c] for c in COLORS] if j == 0 else "0.6", alpha=1 if j == 0 else .7)
        ax.set_xticks(xs); ax.set_xticklabels([COLOR_JP[c] for c in COLORS]); ax.set_ylim(0, 1.05)
        ax.axhline(0.9, ls="--", c="gray", lw=.8)
        ax.set_title(f"E0 ({f.stem}, 各色 {len(df) // 4} エピソード, 95% Wilson CI)", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout(); plt.savefig(FIGS / f"{f.stem}.png", dpi=130); plt.close()


def plot_matrix():
    for f in sorted(RESULTS.glob("matrix_*.csv")):
        df = pd.read_csv(f)
        M, N = np.zeros((4, 4)), np.zeros((4, 4))
        for (s, d), g in df.groupby(["src", "dst"]):
            M[COLORS.index(s), COLORS.index(d)] = g.reached.mean(); N[COLORS.index(s), COLORS.index(d)] = len(g)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        for i in range(4):
            for j in range(4):
                p, lo, hi = wilson(M[i, j] * N[i, j], N[i, j])
                ax.text(j, i, f"{p:.0%}\n[{lo:.0%},{hi:.0%}]", ha="center", va="center", fontsize=7,
                        color="w" if p < .6 else "k", fontweight="bold" if i == j else None)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{COLOR_JP[c]}の箱" for c in COLORS])
        ax.set_yticks(range(4)); ax.set_yticklabels([f"{COLOR_JP[c]}キューブ" for c in COLORS])
        ax.set_xlabel("行き先"); plt.colorbar(im, label="行き先の箱への到達率")
        pat = pattern_of(f.stem)
        st = df[df.strength > 0].strength.iloc[0] if (df.strength > 0).any() else 0
        ax.set_title(f"E1 {PAT_JP.get(pat, pat)}\n{f.stem}, 強度={st:g} (対角=介入なし)", fontsize=9)
        plt.tight_layout(); plt.savefig(FIGS / f"{f.stem}.png", dpi=130); plt.close()


def plot_sweep():
    for f in sorted(RESULTS.glob("sweep_*.csv")):
        df = pd.read_csv(f)
        pat = pattern_of(f.stem)
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
        for kind in ["adj_x", "adj_y", "diag"]:
            g = df[df.kind == kind]
            if g.empty:
                continue
            agg = g.groupby("strength")
            s = np.array(sorted(g.strength.unique()))
            r = [wilson(agg.get_group(x).reached.sum(), len(agg.get_group(x))) for x in s]
            p = np.array([v[0] for v in r])
            axs[0].errorbar(s, p, yerr=[p - [v[1] for v in r], np.array([v[2] for v in r]) - p], marker="o",
                            capsize=3, label=KIND_JP[kind])
            axs[1].plot(s, agg.shift_toward.mean().values, marker="o", label=KIND_JP[kind])
            fail = agg.apply(lambda x: ((~x.grasped) | x.placed_box.isna()).mean())
            axs[2].plot(s, fail.values, marker="o", label=KIND_JP[kind])
        sym = "β" if pat in ("p2", "p3", "p2_colorsae") else "α"
        axs[0].set_ylabel("行き先の箱への到達率"); axs[0].set_ylim(-0.03, 1.03)
        axs[1].set_ylabel("行き先方向への置き位置のずれ [m]"); axs[1].axhline(0, c="k", lw=.5)
        axs[2].set_ylabel("把持失敗 or 箱外の率"); axs[2].set_ylim(-0.03, 1.03)
        for a in axs:
            a.set_xlabel(f"{sym}"); a.legend(fontsize=8)
        fig.suptitle(f"E2 強度スイープ: {PAT_JP.get(pat, pat)} ({f.stem})", fontsize=10)
        plt.tight_layout(); plt.savefig(FIGS / f"{f.stem}.png", dpi=130); plt.close()


def draw_boxes(ax, L):
    for c in COLORS:
        x, y = L.box_xy[c]
        h = L.inner / 2
        ax.add_patch(Rectangle((y - h, x - h), L.inner, L.inner, fill=False, ec=RGB[c], lw=2.5))
    s = L.cube_size
    ax.add_patch(Rectangle((L.cube_xy[1] - s / 2, L.cube_xy[0] - s / 2), s, s, fc="0.5", ec="k"))


def set_topdown(ax, L):
    pts = np.array(list(L.box_xy.values()) + [L.cube_xy])
    lo, hi = pts.min(0) - 0.09, pts.max(0) + 0.09
    ax.set_xlim(hi[1], lo[1]); ax.set_ylim(lo[0], hi[0]); ax.set_aspect("equal")
    ax.set_xlabel("y [m] (左が +y)"); ax.set_ylabel("x [m] (上が奥)")


def run_e3(cfg, ckpt, p1_alpha, p2_beta, p2_all, srcs, seed):
    from pai_steer.steer.runner import run_jobs
    from pai_steer.steer.specs import p1_spec, p2_spec

    L = Layout(cfg["env"])
    jobs = []
    for src in srcs:
        jobs.append(dict(color=src, dst=src, pattern="base", seed=seed, record_traj=True))
        for dst in COLORS:
            if dst == src:
                continue
            jobs.append(dict(color=src, dst=dst, pattern="p1", seed=seed, record_traj=True, steer=p1_spec(L, src, dst, p1_alpha)))
            jobs.append(dict(color=src, dst=dst, pattern="p2", seed=seed, record_traj=True,
                             steer=p2_spec(src, dst, p2_beta, all_tokens=p2_all)))
        diag = [d for d in COLORS if d != src and all(abs(L.box_xy[d] - L.box_xy[src]) > 1e-6)][0]
        for b in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            jobs.append(dict(color=src, dst=diag, pattern=f"p2_beta{b}", seed=seed, record_traj=True,
                             steer=p2_spec(src, diag, b, all_tokens=p2_all)))
        for a in [0.5, 1, 2, 4, 8]:
            jobs.append(dict(color=src, dst=diag, pattern=f"p1_alpha{a}", seed=seed, record_traj=True,
                             steer=p1_spec(L, src, diag, a)))
    res = run_jobs(jobs, ckpt, cfg)
    ensure(ROLLOUTS)
    out = [dict(src=r["job"]["color"], dst=r["job"]["dst"], pattern=r["job"]["pattern"], eef=r["eef"], steer_on=r["steer_on"],
                placed_box=r["placed_box"], final_pos=r["final_pos"]) for r in res]
    np.save(ROLLOUTS / "e3.npy", np.array(out, dtype=object), allow_pickle=True)
    json.dump(dict(p1_alpha=p1_alpha, p2_beta=p2_beta, p2_all=p2_all, seed=seed), open(ROLLOUTS / "e3_meta.json", "w"))


def plot_e3(cfg):
    p = ROLLOUTS / "e3.npy"
    if not p.exists():
        return
    L = Layout(cfg["env"])
    meta = json.load(open(ROLLOUTS / "e3_meta.json"))
    res = list(np.load(p, allow_pickle=True))
    for src in sorted({r["src"] for r in res}, key=COLORS.index):
        fig, axs = plt.subplots(1, 4, figsize=(19, 5))
        for a, pat, title in [(axs[0], "p1", f"P1 (α={meta['p1_alpha']:g})"), (axs[1], "p2", f"P2 (β={meta['p2_beta']:g})")]:
            draw_boxes(a, L)
            for r in res:
                if r["src"] != src or r["pattern"] not in ("base", pat):
                    continue
                e = r["eef"]
                base = r["pattern"] == "base"
                col = "k" if base else RGB[r["dst"]]
                lab = "介入なし" if base else f"→ {COLOR_JP[r['dst']]}の箱 (結果: {COLOR_JP.get(r['placed_box'], '箱外')})"
                a.plot(e[:, 1], e[:, 0], c=col, lw=1.6 if base else 1.2, ls="--" if base else "-", label=lab)
                a.scatter(r["final_pos"][1], r["final_pos"][0], marker="*", s=130, color=col, zorder=5, ec="k", lw=.5)
            set_topdown(a, L); a.set_title(f"{title}: {COLOR_JP[src]}キューブ"); a.legend(fontsize=7, loc="lower left")
        for a, prefix, sym in [(axs[2], "p2_beta", "β"), (axs[3], "p1_alpha", "α")]:
            draw_boxes(a, L)
            rs = [r for r in res if r["src"] == src and r["pattern"].startswith(prefix)]
            cm = plt.get_cmap("plasma")
            for i, r in enumerate(rs):
                col = cm(i / max(1, len(rs) - 1) * .9)
                e = r["eef"]
                a.plot(e[:, 1], e[:, 0], color=col, lw=1, label=f"{sym}={r['pattern'].replace(prefix, '')} → {COLOR_JP.get(r['placed_box'], '箱外')}")
                a.scatter(r["final_pos"][1], r["final_pos"][0], marker="*", s=110, color=col, zorder=5, ec="k", lw=.5)
            set_topdown(a, L)
            dst = rs[0]["dst"] if rs else "?"
            a.set_title(f"{'P2' if sym == 'β' else 'P1'}: {sym} の連続変化 (行き先 {COLOR_JP.get(dst, dst)}の箱)")
            a.legend(fontsize=7, loc="lower left")
        plt.tight_layout(); plt.savefig(FIGS / f"e3_traj_{src}.png", dpi=120); plt.close()


def main():
    p = base_parser(__doc__)
    p.add_argument("--e3", action="store_true")
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--p1-alpha", type=float, default=4.0)
    p.add_argument("--p2-beta", type=float, default=1.0)
    p.add_argument("--p2-all", action="store_true")
    p.add_argument("--e3-srcs", default="red")
    p.add_argument("--e3-seed", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(FIGS)
    if args.e3:
        run_e3(cfg, args.ckpt, args.p1_alpha, args.p2_beta, args.p2_all, args.e3_srcs.split(","), args.e3_seed)
    plot_e0(); plot_matrix(); plot_sweep(); plot_e3(cfg)
    print("figs:", sorted(x.name for x in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
