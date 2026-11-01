"""P3: decoder 側の軌道パターン特徴の同定 (色ごとの4パターンの軌道に対応する特徴).

decoder の各位置 (decoder_layer0..2, decoder_out) で, lift〜lower 区間 (phase 3–5) の活性を
cube_color (= 行き先の箱 = 軌道パターン) ごとに平均し, 選択性 s_{i,c} = E[f_i|c] - max_{c'≠c} E[f_i|c'] で
各色 c の特徴集合 S_c (上位 n 個) を選ぶ. 2通り:
  pooled: チャンク内の全トークンをまとめて1組
  per_k : チャンク内の位置 k ごとに別の組 (軌道の時間方向の形を位置ごとに持つ)
DiffMean 用に位置 k ごとの色平均 μ_c^k (元の活性空間) も保存する.
出力 artifacts/features/traj_features_{site}.json, traj_diffmean_{site}.npz, 図 artifacts/figs/traj_selectivity.png
"""
import json

import numpy as np
import torch

from pai_steer.common import ACTS, COLORS, DEC_SITES, FEATURES, FIGS, SAE_DIR, base_parser, ensure, load_config, setup_mpl
from pai_steer.sae.topk_sae import load_sae

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def encode(sae, x, bs=65536):
    return np.concatenate([sae.encode(torch.from_numpy(np.asarray(x[s:s + bs], dtype=np.float32))).numpy()
                           for s in range(0, len(x), bs)])


N_RANK = 20


def select(f, cid, n_sel):
    means = np.stack([f[cid == c].mean(0) for c in range(4)])
    s = np.stack([means[c] - np.delete(means, c, 0).max(0) for c in range(4)])
    rk = [np.argsort(-s[c])[:N_RANK] for c in range(4)]
    sets = [r[:n_sel] for r in rk]
    rel = float(np.mean([s[c, sets[c]].sum() / (means[c, sets[c]].sum() + 1e-9) for c in range(4)]))
    return dict(sets={COLORS[c]: sets[c].tolist() for c in range(4)},
                dst_means={COLORS[c]: means[c, sets[c]].round(4).tolist() for c in range(4)}, rel_selectivity=rel,
                ranked={COLORS[c]: rk[c].tolist() for c in range(4)},
                ranked_dst_means={COLORS[c]: means[c, rk[c]].round(4).tolist() for c in range(4)})


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(FEATURES, FIGS)
    n_sel = cfg["steer"]["n_select_goal"]
    lab = np.load(ACTS / "labels.npz")
    meta = np.load(ACTS / "dec_meta.npz")
    t, k = meta["t"], meta["k"]
    K = int(k.max()) + 1
    rows = np.where(np.isin(lab["phase"][t], [3, 4, 5]))[0]
    if args.quick:
        rows = rows[:40000]
    cid = lab["color_id"][t[rows]]
    kk = k[rows]
    summary = {}
    for site in DEC_SITES:
        if not (SAE_DIR / f"sae_{site}.pt").exists():
            continue
        sae, _ = load_sae(SAE_DIR / f"sae_{site}.pt")
        x = np.load(ACTS / f"{site}.npy", mmap_mode="r")
        xr = np.asarray(x[rows], dtype=np.float32)
        f = encode(sae, xr)
        out = dict(site=site, K=K, pooled=select(f, cid, n_sel), per_k=[select(f[kk == j], cid[kk == j], n_sel) for j in range(K)])
        mu = np.stack([np.stack([xr[(kk == j) & (cid == c)].mean(0) for c in range(4)]) for j in range(K)])   # [K, 4, d]
        np.savez(FEATURES / f"traj_diffmean_{site}.npz", mu=mu)
        json.dump(out, open(FEATURES / f"traj_features_{site}.json", "w"), indent=1)
        relk = [d["rel_selectivity"] for d in out["per_k"]]
        summary[site] = (out["pooled"]["rel_selectivity"], relk)
        print(f"{site}: pooled rel_sel={out['pooled']['rel_selectivity']:.3f}  per_k rel_sel mean={np.mean(relk):.3f} "
              f"(k=0 {relk[0]:.3f}, k={K - 1} {relk[-1]:.3f})")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for site, (pooled, relk) in summary.items():
        l, = ax.plot(range(len(relk)), relk, marker="o", ms=3, label=f"{site} (位置 k 別)")
        ax.axhline(pooled, ls="--", c=l.get_color(), lw=.8)
    ax.set_xlabel("チャンク内の位置 k"); ax.set_ylabel("相対選択性"); ax.set_ylim(0, 1.05)
    ax.set_title("decoder 側の軌道パターン特徴の選択性 (破線 = 全トークンまとめ)", fontsize=10); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIGS / "traj_selectivity.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
