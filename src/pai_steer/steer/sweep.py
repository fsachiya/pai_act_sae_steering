"""E2: 強度スイープ (§9). P1: α, P2/P3: β. DiffMean ベースラインは --pattern p1_dm / p2_dm / p3_dm (強度は α).

  p1  : x/y 移動特徴 (decoder)          p2  : 色の特徴 (encoder, --color-sae で color トークン専用の色の SAE)
  p3  : 軌道パターン特徴 (decoder の軌道の SAE, --per-k で位置 k 別)

各色 src について, 他の3色の箱 (隣 x / 隣 y / 対角) を行き先に, seed 0..n-1 で評価.
出力 artifacts/results/sweep_{pattern}[_all][_start][tag].csv
"""
import numpy as np
import pandas as pd

from pai_steer.common import CKPT, COLORS, RESULTS, base_parser, ensure, load_config
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs
from pai_steer.steer.specs import box_offset, p1_dm_spec, p1_spec, p2_dm_spec, p2_spec, p3_dm_spec, p3_spec, rel_kind


def make_spec(pattern, layout, src, dst, s, args):
    if s == 0 or src == dst:
        return None
    site = args.site
    if pattern == "p1":
        return p1_spec(layout, src, dst, s, site=site or "decoder_out")
    if pattern == "p1_dm":
        return p1_dm_spec(layout, src, dst, s, site=site or "decoder_out")
    if pattern == "p2":
        return p2_spec(src, dst, s, key=args.key, site=site, all_tokens=args.all_tokens, color_sae=args.color_sae)
    if pattern == "p3":
        return p3_spec(src, dst, s, site=site or "decoder_out", per_k=args.per_k)
    if pattern == "p3_dm":
        return p3_dm_spec(src, dst, s, site=site or "decoder_out")
    if pattern == "p2_dm":
        return p2_dm_spec(src, dst, s, key=args.key, site=site, all_tokens=args.all_tokens)
    raise ValueError(pattern)


def to_df(res, layout):
    rows = []
    for r in res:
        j = r["job"]
        src, dst = j["color"], j["dst"]
        d = box_offset(layout, src, dst)
        u = d / (np.linalg.norm(d) + 1e-12)
        fp = np.asarray(r["final_pos"])[:2]
        rows.append(dict(src=src, dst=dst, kind=rel_kind(layout, src, dst), strength=j["strength"], seed=j["seed"],
                         gate=j.get("gate"), placed_box=r["placed_box"], reached=r["placed_box"] == dst,
                         stayed=r["placed_box"] == src, grasped=r["grasped"], length=r["length"],
                         final_x=fp[0], final_y=fp[1],
                         shift_toward=float(np.dot(fp - layout.box_xy[src], u)) if src != dst else 0.0))
    return pd.DataFrame(rows)


def add_args(p):
    p.add_argument("--pattern", default="p1", choices=["p1", "p2", "p3", "p1_dm", "p2_dm", "p3_dm"])
    p.add_argument("--color-sae", action="store_true", help="P2: color トークン専用の色の SAE を使う")
    p.add_argument("--per-k", action="store_true", help="P3: チャンク内の位置 k ごとの軌道パターン特徴")
    p.add_argument("--site", default=None, help="介入位置 (既定: P1 は decoder_out, P2 は goal_features の best)")
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--gate", default="lift", choices=["lift", "all"])
    p.add_argument("--key", default=None, help="P2: goal_features の候補キー 'site/token' (既定 best)")
    p.add_argument("--all-tokens", action="store_true", help="P2: 同じ encoder 層の3トークン全てに介入")
    p.add_argument("--srcs", default=None, help="元の色 (カンマ区切り, 既定 全色)")
    p.add_argument("--tag", default="")


def out_name(prefix, args):
    site = f"_{args.site}" if args.site else ""
    opts = ("_colorsae" if getattr(args, "color_sae", False) else "") + ("_perk" if getattr(args, "per_k", False) else "")
    return f"{prefix}_{args.pattern}{site}{opts}{'_all' if args.all_tokens else ''}{'_start' if args.gate == 'all' else ''}{args.tag}"


def main():
    p = base_parser(__doc__)
    add_args(p)
    p.add_argument("--strengths", default=None, help="カンマ区切り")
    p.add_argument("--n", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(RESULTS)
    layout = Layout(cfg["env"])
    st = [float(x) for x in args.strengths.split(",")] if args.strengths else \
        cfg["steer"]["strengths"][args.pattern]
    n = args.n or (2 if args.quick else cfg["steer"]["n_sweep"])
    srcs = args.srcs.split(",") if args.srcs else COLORS
    jobs = [dict(color=src, dst=dst, strength=s, seed=seed, gate=args.gate,
                 steer=make_spec(args.pattern, layout, src, dst, s, args))
            for src in srcs for dst in COLORS if dst != src for s in st for seed in range(n)]
    df = to_df(run_jobs(jobs, args.ckpt, cfg), layout)
    df.to_csv(RESULTS / f"{out_name('sweep', args)}.csv", index=False)
    print(df.groupby(["kind", "strength"])[["reached", "stayed", "grasped", "shift_toward"]].mean().round(3))


if __name__ == "__main__":
    main()
