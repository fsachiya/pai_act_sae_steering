"""因果的に選んだ decoder 側の軌道特徴でロールアウト介入 (全色 × 他の3色 × seed n, gate=lift).

  cf    (診断用): 各推論で反実仮想 (color だけ dst 色) の値を計算し, 上位 n 特徴だけ差し替える (状態に応じた値)
  meank (実用形): 上位 n 特徴を dst 色での平均値 (チャンク内の位置 k ごと) に固定する
出力 artifacts/results/traj_causal_rollout.csv
"""
import json

import pandas as pd

from pai_steer.common import CKPT, COLORS, FEATURES, RESULTS, SAE_DIR, base_parser, load_config, site_module
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs, wilson
from pai_steer.steer.sweep import to_df

CONDS = [("cf", "decoder_out", 10), ("cf", "decoder_out", 50), ("cf", "decoder_out", 200), ("cf", "decoder_layer2", 50),
         ("cf", "decoder_layer2", 200), ("meank", "decoder_out", 10), ("meank", "decoder_out", 20),
         ("meank", "decoder_layer2", 20)]


def meank_spec(tc, site, src, dst, n):
    pr = tc["pairs"][f"{src}->{dst}"]
    feats, vals = pr["ranked"][:n], pr["mean_dst_k"]
    return [dict(site=site_module(site), mode="clamp", feats=feats, value=[v[:n] for v in vals][k], tokens=[k],
                 sae_path=str(SAE_DIR / f"sae_{site}.pt")) for k in range(len(vals))]


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()
    cfg = load_config(args.config)
    L = Layout(cfg["env"])
    tcs = {s: json.load(open(FEATURES / f"traj_causal_{s}.json")) for s in {c[1] for c in CONDS}}
    jobs = []
    for kind, site, n in CONDS:
        for a in COLORS:
            for b in COLORS:
                if a == b:
                    continue
                for seed in range(1 if args.quick else args.n):
                    j = dict(kind=kind, site=site, nfeat=n, color=a, dst=b, strength=0.0, seed=seed, gate="lift")
                    if kind == "cf":
                        j["cf_patch"] = dict(site=site, feats=tcs[site]["pairs"][f"{a}->{b}"]["ranked"][:n],
                                             sae_path=str(SAE_DIR / f"sae_{site}.pt"))
                    else:
                        j["steer"] = meank_spec(tcs[site], site, a, b, n)
                    jobs.append(j)
    res = run_jobs(jobs, args.ckpt, cfg)
    df = to_df(res, L)
    for k in ["kind", "site", "nfeat"]:
        df[k] = [r["job"][k] for r in res]
    df.to_csv(RESULTS / "traj_causal_rollout.csv", index=False)
    for (kind, site, n), g in df.groupby(["kind", "site", "nfeat"], sort=False):
        pr, lo, hi = wilson(g.reached.sum(), len(g))
        print(f"{kind:5s} {site:15s} n={n:3d}: 到達率 {pr:.0%} [{lo:.0%},{hi:.0%}], 元の箱 {g.stayed.mean():.0%}, "
              f"箱外 {g.placed_box.isna().mean():.0%}, 方向別 {g.groupby('kind').reached.mean().round(2).to_dict()}")


if __name__ == "__main__":
    main()
