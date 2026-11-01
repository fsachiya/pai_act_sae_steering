"""pick 後に「4パターンのうちどの軌道で動くか」(= 行き先) の情報が, 各位置の活性にどれだけ・どんな形で入っているか.

lift〜lower (phase 3–5) の活性から行き先 (4クラス) を線形分類器で読み出す (デモ単位で学習/評価を分ける).
  raw      : 生の活性 (d=256)
  sae_all  : SAE の全特徴 (2048)
  sae_sel  : 選択性で選んだ特徴だけ (色ごと上位5 → 計20)
  sae_top1 : 最良の1特徴だけ (特徴ごとに色ごとの平均活性で分類したときの最高精度)
さらに運搬の進み具合で3区間 (lift / transport 前半 / 後半〜lower) に分けた精度も出す.
出力 artifacts/results/traj_probe.csv
"""
import numpy as np
import pandas as pd
import torch

from pai_steer.common import ACTS, COLORS, FEATURES, SAE_DIR, SITES, base_parser, is_decoder, load_config
from pai_steer.sae.topk_sae import load_sae


def fit_probe(xtr, ytr, xte, yte, epochs=300, lr=0.05, wd=1e-4):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = torch.tensor((xtr - mu) / sd), torch.tensor((xte - mu) / sd)
    ytr_t = torch.tensor(ytr)
    W = torch.zeros(xtr.shape[1], 4, requires_grad=True)
    b = torch.zeros(4, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        loss = torch.nn.functional.cross_entropy(xtr @ W + b, ytr_t)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float(((xte @ W + b).argmax(1).numpy() == yte).mean())


@torch.no_grad()
def encode(sae, x):
    return np.concatenate([sae.encode(torch.from_numpy(np.asarray(x[s:s + 65536], dtype=np.float32))).numpy()
                           for s in range(0, len(x), 65536)])


def main():
    base_parser(__doc__).parse_args()
    lab = np.load(ACTS / "labels.npz")
    end_of = lab["end_of"]
    demo_of = np.searchsorted(np.unique(end_of), end_of, side="right")   # データセット時刻 → デモ番号
    import json
    g = json.load(open(FEATURES / "goal_features.json"))
    rows = []
    rng = np.random.default_rng(0)
    for site in SITES:
        meta = np.load(ACTS / ("dec_meta.npz" if is_decoder(site) else "enc_meta.npz"))
        t = meta["t"]
        m = np.isin(lab["phase"][t], [3, 4, 5])
        if is_decoder(site):
            m &= meta["k"] == 0          # チャンクの先頭トークンだけ (時刻ごと1サンプル)
        rows_i = np.where(m)[0]
        if len(rows_i) > 30000:
            rows_i = np.sort(rng.choice(rows_i, 30000, replace=False))
        x = np.load(ACTS / f"{site}.npy", mmap_mode="r")
        toks = [("all", None)] if is_decoder(site) else [("robot", 0), ("cube", 1), ("color", 2)]
        tt = t[rows_i]
        y = lab["color_id"][tt]
        dm = demo_of[tt]
        test = (dm % 5) == 0
        ph = lab["phase"][tt]
        sae, _ = load_sae(SAE_DIR / f"sae_{site}.pt")
        for tok, ti in toks:
            key = f"{site}/{tok}"
            if key not in g["candidates"]:
                continue
            xr = np.asarray(x[rows_i] if ti is None else x[rows_i, ti], dtype=np.float32)
            f = encode(sae, xr)
            sel = sorted({i for c in COLORS for i in g["candidates"][key]["sets"][c]})
            r = dict(site=site, token=tok, n=len(y), rel_selectivity=g["candidates"][key]["rel_selectivity"])
            r["raw"] = fit_probe(xr[~test], y[~test], xr[test], y[test])
            r["sae_all"] = fit_probe(f[~test], y[~test], f[test], y[test])
            r["sae_sel20"] = fit_probe(f[~test][:, sel], y[~test], f[test][:, sel], y[test])
            # 最良の1特徴: 学習側で色ごとの平均活性を求め, 評価側で最も近い平均の色に分類
            means = np.stack([f[~test & (y == c)].mean(0) for c in range(4)])           # [4, F]
            pred = np.abs(f[test][:, None, :] - means[None]).argmin(1)                    # [N, F]
            acc1 = (pred == y[test][:, None]).mean(0)
            r["sae_top1"] = float(acc1.max())
            for pn, pm in [("lift", ph == 3), ("transport", ph == 4), ("lower", ph == 5)]:
                tr, te = ~test & pm, test & pm
                r[f"raw_{pn}"] = fit_probe(xr[tr], y[tr], xr[te], y[te]) if te.sum() > 20 else np.nan
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    df = pd.DataFrame(rows)
    df.to_csv("artifacts/results/traj_probe.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
