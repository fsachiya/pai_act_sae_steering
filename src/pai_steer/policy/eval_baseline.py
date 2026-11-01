"""E0: ベースライン評価 (§9). 色ごとに seed 0..N-1 (初期状態ノイズあり).

--sae-recon decoder_out | encoder_layer{i} で, その位置の活性を SAE 再構成に置き換えた条件も測る.
出力 artifacts/results/e0[_recon_{site}].csv
"""
import pandas as pd

from pai_steer.common import CKPT, COLORS, RESULTS, SAE_DIR, base_parser, ensure, load_config, site_module
from pai_steer.steer.runner import run_jobs, wilson


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--sae-recon", default=None)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(RESULTS)
    n = args.n or (4 if args.quick else cfg["eval"]["n_episodes"])
    steer = None
    if args.sae_recon:
        steer = dict(site=site_module(args.sae_recon), mode="recon", sae_path=str(SAE_DIR / f"sae_{args.sae_recon}.pt"))
    jobs = [dict(color=c, seed=s, steer=steer, gate="all") for c in COLORS for s in range(n)]
    res = run_jobs(jobs, args.ckpt, cfg)
    df = pd.DataFrame([dict(color=r["job"]["color"], seed=r["job"]["seed"], placed_box=r["placed_box"],
                            success=r["success"], grasped=r["grasped"], length=r["length"]) for r in res])
    name = "e0" + (f"_recon_{args.sae_recon}" if args.sae_recon else "") + args.tag
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    for c in COLORS:
        d = df[df.color == c]
        pr, lo, hi = wilson(d.success.sum(), len(d))
        print(f"{c:7s} success {pr:.0%} [{lo:.0%},{hi:.0%}]  grasp {d.grasped.mean():.0%}  placed {d.placed_box.value_counts(dropna=False).to_dict()}")
    print(f"ALL success {df.success.mean():.1%}")


if __name__ == "__main__":
    main()
