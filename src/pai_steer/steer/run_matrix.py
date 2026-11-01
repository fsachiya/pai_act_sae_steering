"""E1: 4×4 行列 (§9). 元の色 a × 行き先の箱 b. 各セル seed 0..n-1 (セル間で seed 共通). 対角は介入なし.

強度は E2 の結果から --strength で与える.
出力 artifacts/results/matrix_{pattern}[_all][_start][tag].csv
"""
import pandas as pd

from pai_steer.common import COLORS, RESULTS, base_parser, ensure, load_config
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs
from pai_steer.steer.sweep import add_args, make_spec, out_name, to_df


def main():
    p = base_parser(__doc__)
    add_args(p)
    p.add_argument("--strength", type=float, required=True)
    p.add_argument("--n", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(RESULTS)
    layout = Layout(cfg["env"])
    n = args.n or (2 if args.quick else cfg["steer"]["n_episodes"])
    jobs = [dict(color=src, dst=dst, strength=0.0 if src == dst else args.strength, seed=seed, gate=args.gate,
                 steer=make_spec(args.pattern, layout, src, dst, args.strength, args))
            for src in COLORS for dst in COLORS for seed in range(n)]
    df = to_df(run_jobs(jobs, args.ckpt, cfg), layout)
    df.to_csv(RESULTS / f"{out_name('matrix', args)}.csv", index=False)
    print(pd.pivot_table(df, index="src", columns="dst", values="reached", aggfunc="mean").reindex(index=COLORS, columns=COLORS).round(2))


if __name__ == "__main__":
    main()
