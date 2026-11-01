"""介入仕様 (Steerer 引数) を作る. 行き先は色名 (その色の箱).

  P1 (SAE)      : p1_spec       xy 移動特徴に α×典型活性 を足す (decoder 各層)
  P1 (DiffMean) : p1_dm_spec    将来変位の平均差ベクトル v_x, v_y を足す
  P2 (SAE)      : p2_spec       目標(色)特徴の swap. color_sae=True で color トークン専用の「色の SAE」を使う
  P2 (DiffMean) : p2_dm_spec    色ごとの平均差 μ_dst - μ_src を足す
  P3 (SAE)      : p3_spec       decoder の「軌道の SAE」で軌道パターン特徴を swap (全トークンまとめ / 位置 k 別)
  P3 (DiffMean) : p3_dm_spec    decoder の位置 k ごとの色平均差 μ_dst^k - μ_src^k を足す
"""
import json

import numpy as np

from pai_steer.common import FEATURES, SAE_DIR, is_decoder, site_module

TOKENS = ["robot", "cube", "color"]


def box_offset(layout, src, dst):
    return layout.box_xy[dst] - layout.box_xy[src]


def box_span(layout):
    return np.array([np.ptp([v[0] for v in layout.box_xy.values()]), np.ptp([v[1] for v in layout.box_xy.values()])])


def rel_kind(layout, src, dst):
    if src == dst:
        return "base"
    d = box_offset(layout, src, dst)
    nx, ny = abs(d[0]) > 1e-6, abs(d[1]) > 1e-6
    return "diag" if nx and ny else ("adj_x" if nx else "adj_y")


def p1_spec(layout, src, dst, alpha, site="decoder_out", n=None):
    """元の箱→行き先の箱のずれの符号に応じて x/y 特徴に α×典型活性 を足す (箱の間隔で正規化).
    n: 各方向で使う特徴数 (既定 = config の n_select_xy)."""
    if src == dst:
        return None
    js = json.load(open(FEATURES / f"xy_features_{site}.json"))
    xy, n = js["features"], n or js.get("n_select", 3)
    d, span = box_offset(layout, src, dst), box_span(layout)
    feats, alphas = [], []
    for ax, pos, neg in [(0, "+x", "-x"), (1, "+y", "-y")]:
        if abs(d[ax]) < 1e-6:
            continue
        for r in xy[pos if d[ax] > 0 else neg][:n]:
            feats.append(r["feat"])
            alphas.append(alpha * r["scale"] * abs(d[ax]) / span[ax])
    if not feats:
        return None
    return dict(site=site_module(site), mode="add", feats=feats, alpha=alphas, sae_path=str(SAE_DIR / f"sae_{site}.pt"))


def p1_dm_spec(layout, src, dst, alpha, site="decoder_out"):
    """DiffMean: h + α · Σ_axis sign · v_axis · (|Δ|/間隔) / 2  (v は将来変位 上位25% − 下位25% の平均差)."""
    if src == dst:
        return None
    z = np.load(FEATURES / f"diffmean_{site}.npz")
    d, span = box_offset(layout, src, dst), box_span(layout)
    vec = sum(np.sign(d[ax]) * z["xy"][ax] * abs(d[ax]) / span[ax] / 2 for ax in range(2))
    return dict(site=site_module(site), mode="vector", vec=vec, alpha=alpha)


def goal_features():
    return json.load(open(FEATURES / "goal_features.json"))


def p2_keys(g, key=None, site=None, all_tokens=False, color_sae=False):
    if color_sae:
        key = key or (f"{site}+colorSAE/color" if site else g["best_color_sae"])
        return [key]
    if site is not None:
        key = g["best_per_site"][site]
    key = key or g["best"]
    c = g["candidates"][key]
    if all_tokens and c["token_index"] is not None:
        return [f"{c['site']}/{t}" for t in TOKENS]
    return [key]


def pick(c, src, dst, n, sets_key="sets"):
    """特徴集合 (src, dst) と dst の平均活性. n があれば上位 n 位 (ranked) から取る."""
    if n is None:
        dv = c["means_all_sets"][dst][dst] if "means_all_sets" in c else c["dst_means"][dst]
        return c[sets_key][src], c[sets_key][dst], dv
    return c["ranked"][src][:n], c["ranked"][dst][:n], c["ranked_dst_means"][dst][:n]


def p2_spec(src, dst, beta, key=None, site=None, all_tokens=False, g=None, color_sae=False, n=None):
    """swap: S_src を0, S_dst を「色 dst の運搬時平均活性 × β」に.
    site 指定時はその位置で最も選択性の高いトークン. all_tokens なら同じ encoder 層の3トークン全て."""
    if src == dst:
        return None
    g = g or goal_features()
    specs = []
    for k in p2_keys(g, key, site, all_tokens, color_sae):
        c = g["candidates"][k]
        sf, df, dv = pick(c, src, dst, n)
        specs.append(dict(site=site_module(c["site"]), mode="swap",
                          tokens=None if c["token_index"] is None else [c["token_index"]],
                          src_feats=sf, dst_feats=df, dst_values=dv, beta=beta,
                          sae_path=str(SAE_DIR / f"sae_{c.get('sae', c['site'])}.pt")))
    return specs


def p2_dm_spec(src, dst, alpha, key=None, site=None, all_tokens=False, g=None):
    """DiffMean: P2 と同じ位置・トークンに μ_dst - μ_src を α 倍して足す."""
    if src == dst:
        return None
    from pai_steer.common import COLORS

    g = g or goal_features()
    specs = []
    for k in p2_keys(g, key, site, all_tokens):
        c = g["candidates"][k]
        z = np.load(FEATURES / f"diffmean_{c['site']}.npz")["goal_mu"]
        ti = 0 if c["token_index"] is None else c["token_index"] - 1
        v = z[ti, COLORS.index(dst)] - z[ti, COLORS.index(src)]
        specs.append(dict(site=site_module(c["site"]), mode="vector", vec=v, alpha=alpha,
                          tokens=None if c["token_index"] is None else [c["token_index"]]))
    return specs


def p3_spec(src, dst, beta, site="decoder_out", per_k=False, n=None):
    """軌道の SAE (decoder) で, 元の色の軌道パターン特徴を 0, 行き先の色の特徴を「その色での平均活性 × β」に.
    per_k: チャンク内の位置 k ごとに別の特徴集合 (位置 k のトークンだけに入れる)."""
    if src == dst:
        return None
    tf = json.load(open(FEATURES / f"traj_features_{site}.json"))
    sae_path = str(SAE_DIR / f"sae_{site}.pt")
    groups = [(None, tf["pooled"])] if not per_k else [([j], d) for j, d in enumerate(tf["per_k"])]
    out = []
    for tok, d in groups:
        sf, df, dv = pick(d, src, dst, n)
        out.append(dict(site=site_module(site), mode="swap", tokens=tok, src_feats=sf, dst_feats=df, dst_values=dv,
                        beta=beta, sae_path=sae_path))
    return out


def p3_dm_spec(src, dst, alpha, site="decoder_out"):
    """DiffMean: decoder の位置 k ごとに α (μ_dst^k - μ_src^k) を足す."""
    if src == dst:
        return None
    from pai_steer.common import COLORS

    mu = np.load(FEATURES / f"traj_diffmean_{site}.npz")["mu"]     # [K, 4, d]
    vec = mu[:, COLORS.index(dst)] - mu[:, COLORS.index(src)]       # [K, d] → [B, K, d] にブロードキャスト
    return dict(site=site_module(site), mode="vector", vec=vec, alpha=alpha)


def neutral_color_spec(src, layer=0):
    """元の色の影響を消す: encoder_layer{layer} の color トークンを4色の平均 (中立) に置き換える (DiffMean の平均を使う).
    h' = h + (mean_c μ_c - μ_src)"""
    from pai_steer.common import COLORS

    mu = np.load(FEATURES / f"diffmean_encoder_layer{layer}.npz")["goal_mu"][2]    # color トークン [4, d]
    v = mu.mean(0) - mu[COLORS.index(src)]
    return dict(site=f"encoder_layer{layer}_out", mode="vector", vec=v, alpha=1.0, tokens=[3])


def suppress_color_spec(src, kind, layer=0):
    """色の情報を抑える (x/y 移動・軌道の介入と同時に入れる).
      sae_src : SAE で元の色の特徴 S_src を 0 にする (encoder_layer{layer} の color トークン, 3トークン共通 SAE)
      sae_all : SAE で4色すべての色特徴 (S_赤 ∪ S_緑 ∪ S_青 ∪ S_黄) を 0 にする
      dm_mean : DiffMean で color トークンを4色の平均 (中立) に置き換える
    """
    if kind == "dm_mean":
        return neutral_color_spec(src, layer)
    from pai_steer.common import COLORS

    g = goal_features()
    c = g["candidates"][f"encoder_layer{layer}/color"]
    feats = c["sets"][src] if kind == "sae_src" else sorted({f for col in COLORS for f in c["sets"][col]})
    return dict(site=f"encoder_layer{layer}_out", mode="clamp", feats=feats, value=0.0, tokens=[3],
                sae_path=str(SAE_DIR / f"sae_encoder_layer{layer}.pt"))
