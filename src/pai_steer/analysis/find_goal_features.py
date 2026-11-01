"""P2: 目標 (色→箱) 特徴の同定 (§8.3). 全ての位置 (encoder_layer0..3 × robot/cube/color, decoder 各層 × 全トークン).

lift〜lower 区間 (phase 3–5) の活性を cube_color ごとに平均し, 選択性
  s_{i,c} = E[f_i | c] - max_{c'≠c} E[f_i | c']
で各色 c の特徴集合 S_c (上位 n 個) を選ぶ. 指標は相対選択性 (S_c の選択性 / S_c の平均活性, 1 なら他の色で全く発火しない).
best は仕様どおり encoder 側で最も相対選択性が高い (層, トークン).
色の SAE (color トークンだけで学習した sae_{site}_color.pt) の候補はキー "{site}+colorSAE/color" で別に記録し,
best_color_sae にその中の最良を入れる.
出力 artifacts/features/goal_features.json, 図 artifacts/figs/goal_selectivity_grid.png, goal_selectivity.png
"""
import json

import numpy as np
import torch

from pai_steer.common import (ACTS, COLORS, ENC_SITES, FEATURES, FIGS, SAE_DIR, SITES, base_parser, ensure,
                              is_decoder, load_config, setup_mpl)
from pai_steer.sae.topk_sae import load_sae

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402

TOKENS = ["robot", "cube", "color"]


@torch.no_grad()
def encode(sae, x, bs=65536):
    return np.concatenate([sae.encode(torch.from_numpy(np.asarray(x[s:s + bs], dtype=np.float32))).numpy()
                           for s in range(0, len(x), bs)])


N_RANK = 20   # 特徴数を振る探索のため, 選択性の上位 20 位まで保存する


def selectivity(f, cid, n_sel):
    means = np.stack([f[cid == c].mean(0) for c in range(4)])
    s = np.stack([means[c] - np.delete(means, c, 0).max(0) for c in range(4)])
    sets = {c: np.argsort(-s[c])[:n_sel].tolist() for c in range(4)}
    rel = np.mean([s[c, sets[c]].sum() / (means[c, sets[c]].sum() + 1e-9) for c in range(4)])
    absel = np.mean([s[c, sets[c]].mean() for c in range(4)])
    return means, sets, float(rel), float(absel)


def ranked(f, cid):
    """選択性の上位 N_RANK 特徴と, その各色での平均活性."""
    means = np.stack([f[cid == c].mean(0) for c in range(4)])
    s = np.stack([means[c] - np.delete(means, c, 0).max(0) for c in range(4)])
    rk = {c: np.argsort(-s[c])[:N_RANK] for c in range(4)}
    return dict(ranked={COLORS[c]: rk[c].tolist() for c in range(4)},
                ranked_dst_means={COLORS[c]: means[c, rk[c]].round(4).tolist() for c in range(4)})


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(FEATURES, FIGS)
    n_sel = cfg["steer"]["n_select_goal"]
    lab = np.load(ACTS / "labels.npz")
    t_enc = np.load(ACTS / "enc_meta.npz")["t"]
    t_dec = np.load(ACTS / "dec_meta.npz")["t"]
    out = dict(candidates={})
    maps = {}
    for site in SITES:
        path = SAE_DIR / f"sae_{site}.pt"
        if not path.exists():
            continue
        sae, _ = load_sae(path)
        x = np.load(ACTS / f"{site}.npy", mmap_mode="r")
        t = t_dec if is_decoder(site) else t_enc
        rows = np.where(np.isin(lab["phase"][t], [3, 4, 5]))[0]
        if args.quick:
            rows = rows[:5000]
        cid = lab["color_id"][t[rows]]
        toks = [("all", None, None)] if is_decoder(site) else [(tk, ti, ti + 1) for ti, tk in enumerate(TOKENS)]
        for tok, ti, tidx in toks:
            f = encode(sae, x[rows] if ti is None else x[rows, ti])
            means, sets, rel, absel = selectivity(f, cid, n_sel)
            key = f"{site}/{tok}"
            out["candidates"][key] = dict(**ranked(f, cid),
                site=site, layer=int(site[-1]) if site[-1].isdigit() else 3, token=tok, token_index=tidx,
                rel_selectivity=rel, abs_selectivity=absel,
                sets={COLORS[c]: sets[c] for c in range(4)},
                means_all_sets={COLORS[c]: {COLORS[c2]: means[c2, sets[c]].round(4).tolist() for c2 in range(4)}
                                for c in range(4)})
            maps[key] = (means, sets)
            print(f"{key:28s} rel_sel={rel:.3f} abs_sel={absel:.3f}")
    # 色の SAE (color トークン専用)
    for site in ENC_SITES:
        path = SAE_DIR / f"sae_{site}_color.pt"
        if not path.exists():
            continue
        sae, _ = load_sae(path)
        x = np.load(ACTS / f"{site}.npy", mmap_mode="r")
        rows = np.where(np.isin(lab["phase"][t_enc], [3, 4, 5]))[0]
        cid = lab["color_id"][t_enc[rows]]
        f = encode(sae, x[rows, 2])
        means, sets, rel, absel = selectivity(f, cid, n_sel)
        key = f"{site}+colorSAE/color"
        out["candidates"][key] = dict(**ranked(f, cid),
            site=site, sae=f"{site}_color", layer=int(site[-1]), token="color", token_index=3,
            rel_selectivity=rel, abs_selectivity=absel, sets={COLORS[c]: sets[c] for c in range(4)},
            means_all_sets={COLORS[c]: {COLORS[c2]: means[c2, sets[c]].round(4).tolist() for c2 in range(4)} for c in range(4)})
        print(f"{key:28s} rel_sel={rel:.3f} abs_sel={absel:.3f}")
    color_keys = [k for k in out["candidates"] if "+colorSAE" in k]
    if color_keys:
        out["best_color_sae"] = max(color_keys, key=lambda k: out["candidates"][k]["rel_selectivity"])
    enc_keys = [k for k in out["candidates"] if out["candidates"][k]["site"] in ENC_SITES and "+colorSAE" not in k]
    out["best"] = max(enc_keys, key=lambda k: out["candidates"][k]["rel_selectivity"])
    out["best_per_site"] = {s: max([k for k in out["candidates"] if out["candidates"][k]["site"] == s and "+colorSAE" not in k],
                                   key=lambda k: out["candidates"][k]["rel_selectivity"])
                            for s in {c["site"] for c in out["candidates"].values()}}
    json.dump(out, open(FEATURES / "goal_features.json", "w"), indent=1)
    print("best (encoder):", out["best"])

    sites = [s for s in SITES if any(c["site"] == s for c in out["candidates"].values())]
    cols = TOKENS + ["all"]
    Hm = np.full((len(sites), 4), np.nan)
    for k, c in out["candidates"].items():
        if "+colorSAE" not in k:
            Hm[sites.index(c["site"]), cols.index(c["token"])] = c["rel_selectivity"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(Hm, vmin=0, vmax=1, cmap="magma")
    for i in range(Hm.shape[0]):
        for j in range(4):
            if np.isfinite(Hm[i, j]):
                ax.text(j, i, f"{Hm[i, j]:.2f}", ha="center", va="center", color="w" if Hm[i, j] < .6 else "k")
    ax.set_xticks(range(4)); ax.set_xticklabels(["robot", "cube", "color", "全トークン\n(decoder)"])
    ax.set_yticks(range(len(sites))); ax.set_yticklabels(sites); plt.colorbar(im, label="相対選択性")
    ax.set_title("目標特徴 S_c の選択性 (位置 × トークン)", fontsize=10)
    plt.tight_layout(); plt.savefig(FIGS / "goal_selectivity_grid.png", dpi=130); plt.close()
    means, sets = maps[out["best"]]
    ids = sum([sets[c] for c in range(4)], [])
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(means[:, ids], aspect="auto", cmap="viridis")
    ax.set_yticks(range(4)); ax.set_yticklabels(COLORS)
    ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, rotation=90, fontsize=7)
    ax.set_title(f"目標特徴 S_c の運搬中平均活性 ({out['best']}, 行=キューブの色)"); plt.colorbar(im)
    plt.tight_layout(); plt.savefig(FIGS / "goal_selectivity.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
