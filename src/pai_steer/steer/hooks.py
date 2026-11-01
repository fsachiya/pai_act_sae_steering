"""forward hook による介入 (誤差保存, §8.1).

h' = h + W_dec (f' - f)   (正規化空間で計算し元スケールへ: sae.delta)
SAE が再構成できない残差は壊さない.

mode:
  add    : feats の活性に alpha (feats ごとに配列可) を足す
  clamp  : feats の活性を value に固定
  swap   : src_feats を 0 に, dst_feats を dst_values * beta に
  recon  : h を SAE 再構成 decode(encode(h)) で置き換える (SAE 置換評価用, 誤差保存しない)
  vector : h に vec * alpha を足す (SAE を使わない平均差ベクトル操作)
tokens: 介入するトークン位置 (None=全トークン). encoder なら 1=robot, 2=cube, 3=color.
"""
import numpy as np
import torch


class Steerer:
    def __init__(self, model, module_name, sae=None, mode="add", feats=None, alpha=0.0, value=0.0,
                 src_feats=None, dst_feats=None, dst_values=None, beta=1.0, vec=None, tokens=None):
        self.sae, self.mode, self.tokens = sae, mode, tokens
        self.enabled = False
        dev = next(model.parameters()).device
        t = lambda v: None if v is None else torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev)
        self.feats = None if feats is None else torch.as_tensor(feats, dtype=torch.long, device=dev)
        self.alpha, self.value = t(alpha), t(value)
        self.src = None if src_feats is None else torch.as_tensor(src_feats, dtype=torch.long, device=dev)
        self.dst = None if dst_feats is None else torch.as_tensor(dst_feats, dtype=torch.long, device=dev)
        self.dst_values, self.beta, self.vec = t(dst_values), beta, t(vec)
        self.handle = getattr(model, module_name).register_forward_hook(self._hook)

    def remove(self):
        self.handle.remove()

    def _edit(self, h):
        if self.mode == "vector":
            return h + self.vec * self.alpha
        f = self.sae.encode(h)
        if self.mode == "recon":
            return self.sae.decode(f)
        f2 = f.clone()
        if self.mode == "add":
            f2[..., self.feats] = f2[..., self.feats] + self.alpha
        elif self.mode == "clamp":
            f2[..., self.feats] = self.value
        elif self.mode == "swap":
            f2[..., self.src] = 0.0
            f2[..., self.dst] = self.dst_values * self.beta
        else:
            raise ValueError(self.mode)
        return h + self.sae.delta(f2 - f)

    def _hook(self, mod, inp, out):
        if not self.enabled:
            return None
        h = out.clone()
        if self.tokens is None:
            h = self._edit(h)
        else:
            idx = list(self.tokens)
            h[:, idx] = self._edit(h[:, idx])
        return h
