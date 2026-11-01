"""4パターンの軌道が「パターン固有の特徴」で表現されているか, 「x の行き先 × y の行き先」の組み合わせで表現されているか.

箱は 2×2 (x: 手前/奥, y: 近い/遠い) なので, 色ごとの平均活性 m_c を 2×2 の表とみなし
  m = 全体平均 + x の行き先の効果 + y の行き先の効果 + 交互作用 (= そのパターン固有の部分)
に分解する. 色によって変わる特徴 (色ごとの平均の分散が上位の特徴) について, 分散のうち交互作用が占める割合を見る.
  交互作用が小さい → x と y の組み合わせで表現 (パターン固有の特徴は無い)
  交互作用が大きい → パターンごとに固有の特徴
位置: decoder 各層 (全トークン), encoder_layer0 の color トークン (対照).
あわせて, 関節角度 (states から) が 4パターンでどれだけ違うかも, 同じ分解で示す.
出力 artifacts/results/pattern_factor.csv
"""
import h5py
import numpy as np
import pandas as pd
import torch

from pai_steer.common import ACTS, COLORS, DATA_HDF5, SAE_DIR, base_parser, is_decoder, load_config
from pai_steer.env.layout import Layout
from pai_steer.sae.topk_sae import load_sae


def decompose(M, xlev, ylev):
    """M [4, F] (色ごとの平均). xlev, ylev: 各色の x / y の行き先 (0/1). 分散を x, y, 交互作用に分ける."""
    g = M.mean(0)
    ex = np.stack([M[xlev == v].mean(0) for v in (0, 1)]) - g      # [2, F]
    ey = np.stack([M[ylev == v].mean(0) for v in (0, 1)]) - g
    pred = g + ex[xlev] + ey[ylev]
    inter = M - pred
    vx, vy, vi = (ex[xlev] ** 2).sum(0), (ey[ylev] ** 2).sum(0), (inter ** 2).sum(0)
    return vx, vy, vi


@torch.no_grad()
def encode(sae, x):
    return np.concatenate([sae.encode(torch.from_numpy(np.asarray(x[s:s + 65536], dtype=np.float32))).numpy()
                           for s in range(0, len(x), 65536)])


def main():
    base_parser(__doc__).parse_args()
    cfg = load_config()
    L = Layout(cfg["env"])
    xs = np.array([L.box_xy[c][0] for c in COLORS]); ys = np.array([L.box_xy[c][1] for c in COLORS])
    xlev, ylev = (xs > xs.mean()).astype(int), (ys > ys.mean()).astype(int)
    print("x の行き先 (奥=1):", dict(zip(COLORS, xlev)), " y の行き先 (遠い=1):", dict(zip(COLORS, ylev)))
    lab = np.load(ACTS / "labels.npz")
    rows = []
    sites = ["encoder_layer0/color", "decoder_layer0", "decoder_layer1", "decoder_layer2", "decoder_out"]
    for s in sites:
        site, tok = (s.split("/") + [None])[:2]
        meta = np.load(ACTS / ("dec_meta.npz" if is_decoder(site) else "enc_meta.npz"))
        t = meta["t"]
        rr = np.where(np.isin(lab["phase"][t], [3, 4, 5]))[0]
        rr = rr[:: max(1, len(rr) // 60000)]
        x = np.load(ACTS / f"{site}.npy", mmap_mode="r")
        xr = np.asarray(x[rr] if tok is None else x[rr, 2], dtype=np.float32)
        f = encode(load_sae(SAE_DIR / f"sae_{site}.pt")[0], xr)
        cid = lab["color_id"][t[rr]]
        M = np.stack([f[cid == c].mean(0) for c in range(4)])
        vx, vy, vi = decompose(M, xlev, ylev)
        tot = vx + vy + vi
        top = np.argsort(-tot)[:50]                                    # 色によって最も変わる 50 特徴
        share = lambda v: float(v[top].sum() / tot[top].sum())
        frac_i = vi[top] / (tot[top] + 1e-12)
        rows.append(dict(site=s, x_share=share(vx), y_share=share(vy), interaction_share=share(vi),
                         n_pattern_specific=int((frac_i > 0.5).sum()), n_x=int((vx[top] / tot[top] > 0.5).sum()),
                         n_y=int((vy[top] / tot[top] > 0.5).sum())))
    # 関節角度 (運搬中の平均) の 4 パターン差
    with h5py.File(DATA_HDF5, "r") as fh:
        g = fh["data"]
        J = {c: [] for c in COLORS}
        for n in g:
            if not n.startswith("demo_"):
                continue
            d = g[n]
            ph = d["phase"][:]
            st = d["states"][ph == 4][:, 1:8]                          # qpos の先頭 7 = Panda の関節角
            J[d.attrs["cube_color"]].append(st.mean(0))
    MJ = np.stack([np.mean(J[c], 0) for c in COLORS])                  # [4, 7]
    vx, vy, vi = decompose(MJ, xlev, ylev)
    tot = vx + vy + vi
    rows.append(dict(site="関節角度 (運搬中の平均, 7関節)", x_share=float(vx.sum() / tot.sum()), y_share=float(vy.sum() / tot.sum()),
                     interaction_share=float(vi.sum() / tot.sum()), n_pattern_specific=int((vi / tot > 0.5).sum()),
                     n_x=int((vx / tot > 0.5).sum()), n_y=int((vy / tot > 0.5).sum())))
    print("運搬中の平均関節角 [rad] (行=色):"); print(pd.DataFrame(MJ, index=COLORS, columns=[f"j{i}" for i in range(7)]).round(3))
    df = pd.DataFrame(rows)
    df.to_csv("artifacts/results/pattern_factor.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
