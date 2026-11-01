"""介入が「1回の推論で出す行動チャンク」をどう変えるか (ロールアウトなしの直接比較).

学習データの運搬中 (phase 4) の入力に介入を入れ, 行動チャンク (K ステップ) の手先移動の合計 Σ dpos が
介入なしからどれだけ・どの向きに変わるかを測る. 行き先の方向 u = (箱_dst − 箱_src)/|…| に対し
  toward = Δ·u (行き先方向の変化 [m 相当]), cos = Δ と u のなす角の cos, side = 直交方向の大きさ
また, 行き先の色のデモが同じ運搬進捗で出す実際のチャンクにどれだけ近づくか (gap の減少率) も測る.
出力 artifacts/results/chunk_effect.csv
"""
import numpy as np
import pandas as pd
import torch

from pai_steer.common import CKPT, COLORS, RESULTS, base_parser, load_config
from pai_steer.data.dataset import ChunkDataset
from pai_steer.env.layout import Layout
from pai_steer.policy.act import load_policy
from pai_steer.steer.runner import build_steer
from pai_steer.steer.specs import p1_dm_spec, p1_spec, p2_dm_spec, p2_spec, p3_dm_spec, p3_spec

OUT_MAX = 0.05   # OSC の位置指令: 行動 1 あたり 5cm


def chunk_dpos(model, stats, b):
    with torch.no_grad():
        a = model(*(torch.from_numpy(b[k]) for k in ["robot", "cube", "color"]))[0].numpy()
    a = a * stats["act_std"] + stats["act_mean"]
    return np.clip(a[..., :3], -1, 1).sum(1)[:, :2] * OUT_MAX          # [B, 2] チャンク内の指令移動の合計 (x, y)


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    args = p.parse_args()
    cfg = load_config(args.config)
    L = Layout(cfg["env"])
    model, mcfg, stats = load_policy(args.ckpt)
    ds = ChunkDataset(chunk=mcfg["act"]["chunk"], stats=stats)
    conds = [("色 SAE β=2", lambda a, b: p2_spec(a, b, 2.0)), ("色 DiffMean α=1.5", lambda a, b: p2_dm_spec(a, b, 1.5)),
             ("x/y SAE α=0.5", lambda a, b: p1_spec(L, a, b, 0.5)), ("x/y DiffMean α=1", lambda a, b: p1_dm_spec(L, a, b, 1.0)),
             ("軌道 SAE β=8", lambda a, b: p3_spec(a, b, 8.0, per_k=True)), ("軌道 DiffMean α=1", lambda a, b: p3_dm_spec(a, b, 1.0))]
    rng = np.random.default_rng(0)
    # 運搬進捗 (運搬区間内の相対時刻) ごとの, 各色の実際のチャンク (目標の参照)
    prog = np.zeros(len(ds))
    for s, e in ds.demo_range:
        idx = np.arange(s, e)[ds.phase[s:e] == 4]
        if len(idx):
            prog[idx] = np.linspace(0, 1, len(idx))
    rows = []
    for src in COLORS:
        idx_src = np.where((ds.color_id == COLORS.index(src)) & (ds.phase == 4))[0]
        idx = rng.choice(idx_src, size=min(256, len(idx_src)), replace=False)
        b = ds.batch(idx)
        base = chunk_dpos(model, stats, b)
        for dst in COLORS:
            if dst == src:
                continue
            u = L.box_xy[dst] - L.box_xy[src]
            u = u / np.linalg.norm(u)
            # 参照: 行き先の色のデモで, 同じ運搬進捗の時刻のチャンク (色だけ入れ替えた入力に対する方策の出力)
            bd = dict(b); bd["color"] = np.tile(np.eye(4, dtype=np.float32)[COLORS.index(dst)], (len(idx), 1))
            ref = chunk_dpos(model, stats, bd)
            for name, fn in conds:
                st = build_steer(model, fn(src, dst))
                st.enabled = True
                try:
                    out = chunk_dpos(model, stats, b)
                finally:
                    st.remove()
                d = out - base
                toward = d @ u
                gap0 = np.linalg.norm(ref - base, axis=1).mean()
                gap1 = np.linalg.norm(ref - out, axis=1).mean()
                rows.append(dict(cond=name, src=src, dst=dst, toward=toward.mean(),
                                 side=np.abs(d @ np.array([-u[1], u[0]])).mean(),
                                 cos=float(np.mean((d @ u) / (np.linalg.norm(d, axis=1) + 1e-9))),
                                 gap_reduction=1 - gap1 / (gap0 + 1e-9), ref_toward=((ref - base) @ u).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "chunk_effect.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.groupby("cond", sort=False)[["toward", "side", "cos", "gap_reduction", "ref_toward"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
