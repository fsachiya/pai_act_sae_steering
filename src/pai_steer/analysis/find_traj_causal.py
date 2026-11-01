"""decoder 側の軌道特徴を因果的な効果で特定する (反実仮想の対 + attribution patching).

src 色のキューブの持ち上げ〜下ろす区間 (phase 3–5) の入力 x について,
  元の実行      : 方策 (x)                    → decoder の位置 site の SAE 特徴 f_src, チャンク a_src
  反実仮想の実行 : color トークンだけ dst 色にした x' → f_dst, チャンク a_ref (= 行き先の箱へ向かうチャンク)
状態は同じなので f_dst - f_src は「行き先だけ」による特徴の差. 寄与
  attr_i = Σ_{サンプル, トークン} (f_dst - f_src)_i · ∂J/∂f_i,   J = -||a(f) - a_ref||²   (f = f_src で評価)
の大きい順に特徴を選ぶ. 特定の判定として, 上位 n 特徴だけを f_dst の値に差し替えたときの
「参照チャンクへの近づき」 1 - |a_patch - a_ref| / |a_src - a_ref| を n ごとに測る (学習用と別のサンプルで評価).
介入用に, 選んだ特徴の dst 色での平均値 (チャンク内の位置 k ごと) を保存する.
出力 artifacts/features/traj_causal_{site}.json, artifacts/results/traj_causal_patch.csv, 図 artifacts/figs/traj_causal.png
"""
import json

import numpy as np
import pandas as pd
import torch

from pai_steer.common import (CKPT, COLORS, DEC_SITES, FEATURES, FIGS, RESULTS, SAE_DIR, base_parser, ensure, load_config,
                              setup_mpl, site_module)
from pai_steer.data.dataset import ChunkDataset
from pai_steer.policy.act import load_policy
from pai_steer.sae.topk_sae import load_sae

setup_mpl()
import matplotlib.pyplot as plt  # noqa: E402

NS = [1, 3, 5, 10, 20, 50, 100, 200]


class SiteHook:
    """位置 site の出力を記録 / 置換する."""

    def __init__(self, model, site):
        self.mode, self.out, self.repl = "record", None, None
        self.h = getattr(model, site_module(site)).register_forward_hook(self._f)

    def _f(self, m, i, o):
        if self.mode == "record":
            self.out = o
            return None
        return self.repl(o)

    def remove(self):
        self.h.remove()


def run(model, b, color=None):
    c = torch.from_numpy(b["color"] if color is None else color)
    return model(torch.from_numpy(b["robot"]), torch.from_numpy(b["cube"]), c)[0]


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--sites", default=",".join(DEC_SITES))
    p.add_argument("--n-fit", type=int, default=192)
    p.add_argument("--n-eval", type=int, default=128)
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(FEATURES, FIGS, RESULTS)
    model, mcfg, stats = load_policy(args.ckpt)
    for q in model.parameters():
        q.requires_grad_(False)
    ds = ChunkDataset(chunk=mcfg["act"]["chunk"], stats=stats)
    rng = np.random.default_rng(0)
    rows = []
    for site in args.sites.split(","):
        sae, _ = load_sae(SAE_DIR / f"sae_{site}.pt")
        hook = SiteHook(model, site)
        out = dict(site=site, pairs={})
        for src in COLORS:
            pool = np.where((ds.color_id == COLORS.index(src)) & np.isin(ds.phase, [3, 4, 5]))[0]
            demo = np.searchsorted(np.array([e for _, e in ds.demo_range]), pool, side="right")
            fit_idx = rng.choice(pool[demo % 5 != 0], args.n_fit, replace=False)
            ev_idx = rng.choice(pool[demo % 5 == 0], args.n_eval, replace=False)
            for dst in COLORS:
                if dst == src:
                    continue
                onehot = np.eye(4, dtype=np.float32)[COLORS.index(dst)]

                def pair_feats(idx):
                    b = ds.batch(idx)
                    hook.mode = "record"
                    with torch.no_grad():
                        a_src = run(model, b); h_src = hook.out.clone()
                        a_ref = run(model, b, np.tile(onehot, (len(idx), 1))); h_dst = hook.out.clone()
                        f_src, f_dst = sae.encode(h_src), sae.encode(h_dst)
                    return b, a_src, a_ref, h_src, f_src, f_dst

                # 寄与 (学習用サンプル)
                b, a_src, a_ref, h_src, f_src, f_dst = pair_feats(fit_idx)
                f = f_src.clone().requires_grad_(True)
                hook.mode = "replace"
                hook.repl = lambda o: h_src + sae.delta(f - f_src)
                J = -((run(model, b) - a_ref) ** 2).sum()
                J.backward()
                attr = ((f_dst - f_src) * f.grad).sum((0, 1)).detach().numpy()
                order = np.argsort(-attr)
                # 平均値 (位置 k ごと, 学習用サンプル): 介入用
                mean_dst_k = f_dst.mean(0).numpy()          # [K, F]
                out["pairs"][f"{src}->{dst}"] = dict(ranked=order[:200].tolist(), attr=attr[order[:200]].round(5).tolist(),
                                                    mean_dst_k=mean_dst_k[:, order[:200]].round(4).tolist())
                # 評価: 上位 n だけ f_dst の値 (サンプルごと) に差し替え / 位置 k ごとの平均値に固定
                b, a_src, a_ref, h_src, f_src, f_dst = pair_feats(ev_idx)
                gap0 = (a_src - a_ref).norm(dim=(1, 2)).mean().item()
                for n in NS:
                    sel = torch.from_numpy(order[:n].copy())
                    for kind in ["exact", "mean_k"]:
                        f2 = f_src.clone()
                        f2[..., sel] = f_dst[..., sel] if kind == "exact" else torch.from_numpy(mean_dst_k[:, order[:n]]).float()[None]
                        hook.mode, hook.repl = "replace", (lambda o, f2=f2: h_src + sae.delta(f2 - f_src))
                        with torch.no_grad():
                            a = run(model, b)
                        gap = (a - a_ref).norm(dim=(1, 2)).mean().item()
                        rows.append(dict(site=site, src=src, dst=dst, n=n, kind=kind, gap_reduction=1 - gap / gap0))
                hook.mode = "record"
            print(site, src, "done")
        hook.remove()
        json.dump(out, open(FEATURES / f"traj_causal_{site}.json", "w"))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "traj_causal_patch.csv", index=False)
    piv = df.groupby(["site", "kind", "n"]).gap_reduction.mean().unstack("n").round(3)
    pd.set_option("display.width", 200)
    print(piv.to_string())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, kind, t in [(axs[0], "exact", "上位 n 特徴をサンプルごとの反実仮想の値に差し替え"),
                        (axs[1], "mean_k", "上位 n 特徴を行き先の色での平均値 (位置 k ごと) に固定")]:
        for site in df.site.unique():
            g = df[(df.site == site) & (df.kind == kind)].groupby("n").gap_reduction.mean()
            ax.plot(g.index, g.values, marker="o", label=site)
        ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS); ax.axhline(0, c="k", lw=.5)
        ax.set_xlabel("差し替える特徴の数 n (寄与の大きい順)"); ax.set_title(t, fontsize=10); ax.legend(fontsize=8)
    axs[0].set_ylabel("参照チャンク (行き先の箱へ向かう) への近づき")
    fig.suptitle("decoder 側の軌道特徴の因果的な特定 (評価用サンプル, 全色 × 他の3色の平均)", fontsize=11)
    plt.tight_layout(); plt.savefig(FIGS / "traj_causal.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
