"""x/y 移動 (P1) と軌道 (P3) の介入の精度が低い要因の切り分け.

仮説 (ユーザー): 色の情報と動作が強く結びついているため, x/y・軌道の特徴を刺激しても元の色が行動を引き戻す.
→ x/y・軌道の特徴を刺激すると同時に色の特徴を抑えれば精度が上がる.
あわせて「下ろす・離すタイミングが元の目標で決まる」(要因2) も確かめる. P1 / P3 (SAE, DiffMean) に次を掛ける:
  gate     : lift (持ち上げた後ずっと) / transport (運搬中で下ろし始める前だけ)
  suppress : なし / sae_src (SAE で元の色の特徴を 0) / sae_all (SAE で4色すべての色特徴を 0) /
             dm_mean (DiffMean で color トークンを4色の平均に)  ※色の抑制は同じ gate で入れる
対照: 色の抑制だけ (P1/P3 なし) で, キューブがどこへ置かれるか.
各マス = 4色 × 他の3色の箱 × seed 0..n-1.
出力 artifacts/results/factor_study.csv, factor_study_summary.csv, 図 artifacts/figs/factor_study.png
"""
import numpy as np
import pandas as pd

from pai_steer.common import CKPT, COLORS, FIGS, RESULTS, base_parser, ensure, load_config, setup_mpl
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs, wilson
from pai_steer.steer.specs import p1_dm_spec, p1_spec, p3_dm_spec, p3_spec, suppress_color_spec
from pai_steer.steer.sweep import rel_kind, to_df

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402

CONDS = [  # (ラベル, 強度, spec(L, src, dst, s))
    ("x/y SAE", 0.5, lambda L, a, b, s: p1_spec(L, a, b, s)),
    ("x/y SAE", 1.0, lambda L, a, b, s: p1_spec(L, a, b, s)),
    ("x/y DiffMean", 1.0, lambda L, a, b, s: p1_dm_spec(L, a, b, s)),
    ("x/y DiffMean", 2.0, lambda L, a, b, s: p1_dm_spec(L, a, b, s)),
    ("軌道 SAE", 8.0, lambda L, a, b, s: p3_spec(a, b, s, per_k=True)),
    ("軌道 SAE", 16.0, lambda L, a, b, s: p3_spec(a, b, s, per_k=True)),
    ("軌道 DiffMean", 1.0, lambda L, a, b, s: p3_dm_spec(a, b, s)),
    ("軌道 DiffMean", 2.0, lambda L, a, b, s: p3_dm_spec(a, b, s)),
]
SUPPRESS = [None, "sae_src", "sae_all", "dm_mean"]
SUP_JP = {None: "色の抑制なし", "sae_src": "SAE: 元の色の特徴を0", "sae_all": "SAE: 全色の特徴を0", "dm_mean": "DiffMean: 色を中立化"}
GATE_JP = {"lift": "持ち上げ後ずっと", "transport": "運搬中だけ"}
VARIANTS = [(g, sp) for g in ["lift", "transport"] for sp in SUPPRESS]
VAR_JP = {(g, sp): f"{GATE_JP[g]} / {SUP_JP[sp]}" for g, sp in VARIANTS}


def as_list(spec):
    return [] if spec is None else (spec if isinstance(spec, list) else [spec])


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(RESULTS, FIGS)
    L = Layout(cfg["env"])
    n = 1 if args.quick else args.n
    jobs = []
    for ci, (lab, st, fn) in enumerate(CONDS):
        for gate, sp in VARIANTS:
            for a in COLORS:
                for b in COLORS:
                    if a == b:
                        continue
                    spec = as_list(fn(L, a, b, st)) + (as_list(suppress_color_spec(a, sp)) if sp else [])
                    for seed in range(n):
                        jobs.append(dict(cond=lab, strength=st, gate=gate, suppress=sp or "none", color=a, dst=b, seed=seed, steer=spec))
    # 対照: 色の抑制だけ (dst は記録用に元の色)
    for gate in ["lift", "transport"]:
        for sp in SUPPRESS[1:]:
            for a in COLORS:
                for seed in range(2 if args.quick else 10):
                    jobs.append(dict(cond="色の抑制のみ", strength=0.0, gate=gate, suppress=sp, color=a, dst=a, seed=seed,
                                     steer=as_list(suppress_color_spec(a, sp))))
    print(len(jobs), "episodes")
    res = run_jobs(jobs, args.ckpt, cfg)
    df = to_df(res, L)
    for k in ["cond", "gate", "suppress"]:
        df[k] = [r["job"][k] for r in res]
    df.to_csv(RESULTS / "factor_study.csv", index=False)

    ctrl = df[df.cond == "色の抑制のみ"]
    print("対照 (色の抑制のみ): 置かれた箱 (元の色の箱 / 他の箱 / 箱外)")
    for (gate, sp), g in ctrl.groupby(["gate", "suppress"]):
        print(f"  {GATE_JP[gate]} / {SUP_JP[sp]}: 元の箱 {g.stayed.mean():.0%}, 箱外 {g.placed_box.isna().mean():.0%}, "
              f"他の箱 {(~g.stayed & g.placed_box.notna()).mean():.0%}")
    d = df[df.cond != "色の抑制のみ"]
    rows = []
    for (lab, st, gate, sp), g in d.groupby(["cond", "strength", "gate", "suppress"], sort=False):
        sp = None if sp == "none" else sp
        pr, lo, hi = wilson(g.reached.sum(), len(g))
        rows.append(dict(条件=f"{lab} {st:g}", 変化=VAR_JP[(gate, sp)], 到達率=pr, ci_lo=lo, ci_hi=hi,
                         箱外=g.placed_box.isna().mean(), 元の箱=g.stayed.mean(),
                         **{f"到達率_{k}": v for k, v in g.groupby("kind").reached.mean().items()}))
    sm = pd.DataFrame(rows)
    sm.to_csv(RESULTS / "factor_study_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(sm.round(2).to_string())

    conds = list(dict.fromkeys(sm.条件))
    fig, axs = plt.subplots(3, 2, figsize=(20, 13))
    x = np.arange(len(conds)); w = 0.2
    cols = {None: "0.5", "sae_src": "C0", "sae_all": "C2", "dm_mean": "C1"}
    for gi, gate in enumerate(["lift", "transport"]):
        for j, sp in enumerate(SUPPRESS):
            sub = sm[sm.変化 == VAR_JP[(gate, sp)]].set_index("条件").reindex(conds)
            for ri, col in enumerate(["到達率", "箱外", "元の箱"]):
                axs[ri, gi].bar(x - 0.3 + w * j, sub[col], w, color=cols[sp], label=SUP_JP[sp])
        for ri, t in enumerate(["行き先の箱への到達率", "どの箱にも入らない率", "元の色の箱に戻った率"]):
            a = axs[ri, gi]
            a.set_xticks(x); a.set_xticklabels(conds, rotation=25, ha="right", fontsize=8); a.set_ylim(0, 1.05)
            a.set_title(f"{t} (介入: {GATE_JP[gate]})", fontsize=10); a.legend(fontsize=7)
    fig.suptitle("x/y 移動・軌道の介入 + 色の特徴の抑制 (各棒 60 エピソード = 4色 × 他の3色 × seed 5)", fontsize=12)
    plt.tight_layout(); plt.savefig(FIGS / "factor_study.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
