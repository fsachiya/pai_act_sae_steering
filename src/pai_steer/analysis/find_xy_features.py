"""P1: xy 移動特徴の同定 (§8.2). --site decoder_out (既定) | decoder_layer{i} | all.

1. 各特徴のデコーダ方向 d_i (元スケール) を「典型活性 (正の活性の 90% 分位) ぶん」その位置の全チャンクトークンに足し,
   運搬中の入力 B 個で行動チャンクの平均変化 Δa_i (元スケール) を測る (摂動法).
   decoder_out では Δa_i = W_head d_i × 典型活性 と一致する (action head が線形のため).
2. 運搬中 (phase 4 transport, 5 lower) の, 特徴活性と手先の将来変位 (そのチャンク位置の1ステップ変位) の相関.
3. +x, -x, +y, -y の各方向で, 寄与の符号と相関の符号が一致し運搬中によく発火する特徴を,
   「寄与 × 軸の純度 × |相関|」の大きい順に上位 n 個選ぶ.
出力 artifacts/features/xy_features_{site}.json (decoder_out は xy_features.json にも), 図 artifacts/figs/xy_contrib_{site}.png
"""
import json
import shutil

import numpy as np
import torch

from pai_steer.common import (ACTS, CKPT, DEC_SITES, FEATURES, FIGS, SAE_DIR, base_parser, ensure, get_device, load_config,
                              setup_mpl, site_module)
from pai_steer.data.dataset import ChunkDataset
from pai_steer.policy.act import load_policy
from pai_steer.sae.topk_sae import load_sae

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def encode_all(sae, x, bs=65536):
    return np.concatenate([sae.encode(torch.from_numpy(np.asarray(x[s:s + bs], dtype=np.float32))).numpy()
                           for s in range(0, len(x), bs)])


@torch.no_grad()
def perturb_effect(model, stats, site, D, ds, idx, chunk=32):
    """D [F, d] の各方向を位置 site の全トークンに足したときの行動チャンク平均変化 [F, 7] (元スケール).
    特徴 chunk 個ぶんの入力を複製して1回の forward で測る."""
    dev = next(model.parameters()).device
    b = ds.batch(idx)
    inp = [torch.from_numpy(b[k]).to(dev) for k in ["robot", "cube", "color"]]
    B = len(idx)
    base = model(*inp)[0].mean((0, 1))
    vec = {"v": None}
    h = getattr(model, site_module(site)).register_forward_hook(lambda m, i, o: o + vec["v"] if vec["v"] is not None else None)
    out = np.zeros((len(D), 7), dtype=np.float32)
    Dt = torch.from_numpy(D).to(dev)
    try:
        for s in range(0, len(D), chunk):
            dc = Dt[s:s + chunk]                                   # [c, d]
            c = len(dc)
            vec["v"] = dc.repeat_interleave(B, 0)[:, None, :]      # [c*B, 1, d]
            a = model(*(x.repeat(c, *[1] * (x.dim() - 1)) for x in inp))[0]   # [c*B, K, 7]
            out[s:s + c] = (a.view(c, B, *a.shape[1:]).mean((1, 2)) - base).cpu().numpy()
    finally:
        h.remove()
        vec["v"] = None
    return out * stats["act_std"]


def find(site, cfg, model, stats, ds, quick=False):
    n_sel = cfg["steer"]["n_select_xy"]
    sae, _ = load_sae(SAE_DIR / f"sae_{site}.pt")
    lab = np.load(ACTS / "labels.npz")
    meta = np.load(ACTS / "dec_meta.npz")
    t, k = meta["t"], meta["k"]
    x = np.load(ACTS / f"{site}.npy", mmap_mode="r")
    tk = t + k
    valid = np.isin(lab["phase"][t], [4, 5]) & (tk + 1 < lab["end_of"][t])
    rows = np.where(valid)[0]
    if quick:
        rows = rows[:20000]
    f = encode_all(sae, x[rows])
    disp = lab["eef"][tk[rows] + 1] - lab["eef"][tk[rows]]
    fire = (f > 0).mean(0)
    scale = np.array([np.quantile(f[:, i][f[:, i] > 0], 0.9) if (f[:, i] > 0).any() else 0.0 for i in range(f.shape[1])])
    D = ((sae.W_dec * sae.std).detach().numpy() * scale[:, None]).astype(np.float32)   # 典型活性ぶんの方向
    tr = np.unique(t[rows])
    idx = np.random.default_rng(0).choice(tr, size=min(64, len(tr)), replace=False)
    dA = perturb_effect(model, stats, site, D, ds, idx)          # [F, 7] 典型活性1単位あたり
    fc = f - f.mean(0)

    def corr(v):
        vc = v - v.mean()
        return (fc * vc[:, None]).sum(0) / (np.sqrt((fc ** 2).sum(0) * (vc ** 2).sum()) + 1e-12)

    cx, cy = corr(disp[:, 0]), corr(disp[:, 1])
    res = {}
    for name, ax, sign, c in [("+x", 0, 1, cx), ("-x", 0, -1, cx), ("+y", 1, 1, cy), ("-y", 1, -1, cy)]:
        purity = np.abs(dA[:, ax]) / (np.linalg.norm(dA[:, :3], axis=1) + 1e-12)
        ok = (sign * dA[:, ax] > 0) & (sign * c > 0) & (fire > 0.01)
        score = np.where(ok, sign * dA[:, ax] * purity * np.abs(c), -np.inf)
        top = [i for i in np.argsort(-score)[:20] if np.isfinite(score[i])]   # 特徴数を振る探索のため上位 20 まで
        res[name] = [dict(feat=int(i), dA=dA[i, :3].round(4).tolist(), corr=float(c[i]), fire=float(fire[i]),
                          scale=float(scale[i]), purity=float(purity[i])) for i in top]
        print(site, name, [(r["feat"], np.round(r["dA"], 3).tolist(), round(r["corr"], 2)) for r in res[name][:n_sel]])
    json.dump(dict(site=site, features=res, n_select=n_sel, n_tokens=int(len(rows))), open(FEATURES / f"xy_features_{site}.json", "w"), indent=1)
    if site == "decoder_out":
        shutil.copy(FEATURES / f"xy_features_{site}.json", FEATURES / "xy_features.json")
    np.savez(FEATURES / f"dec_feature_stats_{site}.npz", dA=dA, scale=scale, fire=fire, cx=cx, cy=cy)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    m = fire > 0.01
    cols = {"+x": "C0", "-x": "C1", "+y": "C2", "-y": "C3"}
    a = axs[0]
    a.scatter(dA[m, 0], dA[m, 1], s=4, c="0.7", label="運搬中に発火する特徴")
    for n, lst in res.items():
        ids = [r["feat"] for r in lst[:n_sel]]
        a.scatter(dA[ids, 0], dA[ids, 1], s=40, c=cols[n], label=n)
    a.set_xlabel("Δa_x (典型活性ぶん)"); a.set_ylabel("Δa_y (典型活性ぶん)"); a.axhline(0, c="k", lw=.5); a.axvline(0, c="k", lw=.5)
    a.legend(fontsize=8); a.set_title(f"{site} 特徴の行動寄与 (摂動法)")
    a = axs[1]
    a.scatter(cx[m], cy[m], s=4, c="0.7")
    for n, lst in res.items():
        ids = [r["feat"] for r in lst[:n_sel]]
        a.scatter(cx[ids], cy[ids], s=40, c=cols[n], label=n)
    a.set_xlabel("corr(f, 将来 dx)"); a.set_ylabel("corr(f, 将来 dy)"); a.set_title("運搬中の活性と手先変位の相関")
    a.axhline(0, c="k", lw=.5); a.axvline(0, c="k", lw=.5)
    plt.tight_layout(); plt.savefig(FIGS / f"xy_contrib_{site}.png", dpi=130); plt.close()


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--site", default="decoder_out")
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(FEATURES, FIGS)
    model, mcfg, stats = load_policy(args.ckpt, get_device())
    ds = ChunkDataset(chunk=mcfg["act"]["chunk"], stats=stats)
    for site in (DEC_SITES if args.site == "all" else [args.site]):
        if (SAE_DIR / f"sae_{site}.pt").exists():
            find(site, cfg, model, stats, ds, args.quick)


if __name__ == "__main__":
    main()
