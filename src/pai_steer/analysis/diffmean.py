"""DiffMean ベースライン (§8.4 を拡張): SAE を使わず, 平均活性の差ベクトルで介入する.

各位置について次を保存する (artifacts/features/diffmean_{site}.npz):
  goal_mu [n_tok, 4, d]: lift〜lower 区間 (phase 3–5) の色ごとの平均活性
     (encoder はトークン robot/cube/color ごと, decoder は全チャンクトークンをまとめて n_tok=1)
     → P2 の DiffMean: h' = h + α (μ_dst - μ_src)
  xy [2, d] (decoder のみ): 運搬中 (phase 4–5) の, 将来の手先変位 dx (dy) が上位 25% のトークンの平均
     - 下位 25% のトークンの平均 → P1 の DiffMean: h' = h + α · sign · v_x · (|Δx| / 箱間隔) / 2
"""
import numpy as np

from pai_steer.common import ACTS, FEATURES, SITES, base_parser, ensure, is_decoder, load_config


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    ensure(FEATURES)
    lab = np.load(ACTS / "labels.npz")
    t_enc = np.load(ACTS / "enc_meta.npz")["t"]
    meta = np.load(ACTS / "dec_meta.npz")
    t_dec, k_dec = meta["t"], meta["k"]
    for site in SITES:
        path = ACTS / f"{site}.npy"
        if not path.exists():
            continue
        x = np.load(path, mmap_mode="r")
        t = t_dec if is_decoder(site) else t_enc
        rows = np.where(np.isin(lab["phase"][t], [3, 4, 5]))[0]
        if args.quick:
            rows = rows[:20000]
        xs = np.asarray(x[rows], dtype=np.float32)
        if not is_decoder(site):
            xs = xs.transpose(1, 0, 2)          # [3, N, d]
        else:
            xs = xs[None]                       # [1, N, d]
        cid = lab["color_id"][t[rows]]
        mu = np.stack([np.stack([xt[cid == c].mean(0) for c in range(4)]) for xt in xs])
        out = dict(goal_mu=mu)
        if is_decoder(site):
            tk = t + k_dec
            v = np.isin(lab["phase"][t], [4, 5]) & (tk + 1 < lab["end_of"][t])
            r2 = np.where(v)[0]
            if args.quick:
                r2 = r2[:20000]
            h = np.asarray(x[r2], dtype=np.float32)
            disp = lab["eef"][tk[r2] + 1] - lab["eef"][tk[r2]]
            xy = []
            for ax in range(2):
                lo, hi = np.quantile(disp[:, ax], [0.25, 0.75])
                xy.append(h[disp[:, ax] >= hi].mean(0) - h[disp[:, ax] <= lo].mean(0))
            out["xy"] = np.stack(xy)
        np.savez(FEATURES / f"diffmean_{site}.npz", **out)
        d = np.linalg.norm(mu[:, :, None] - mu[:, None], axis=-1).mean((1, 2))
        extra = f"  |v_x|={np.linalg.norm(out['xy'][0]):.2f} |v_y|={np.linalg.norm(out['xy'][1]):.2f}" if "xy" in out else ""
        print(f"{site}: mean |μ_a-μ_b| per token = {d.round(2)}{extra}")


if __name__ == "__main__":
    main()
