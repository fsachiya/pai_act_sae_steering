"""TopK SAE (既存の別プロジェクトの TopKSAE を移植).

f = TopK(W_enc (norm(h) - b_pre) + b_enc),  norm(h)^ = W_dec f + b_dec
入力の正規化統計をバッファに持ち, デコーダ列は単位ノルムに保つ. 任意で OrtSAE の直交化損失.
"""
import torch
import torch.nn.functional as F
from torch import nn


def topk_activations(pre, k):
    vals, idx = pre.topk(k, dim=-1)
    return torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))


def orthogonality_loss(W_dec, eps=1e-8):
    """OrtSAE (arXiv:2509.22033): 各特徴の最類似な他特徴との cos 類似度の2乗平均."""
    N = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(eps)
    cos = N @ N.t() - 2.0 * torch.eye(N.shape[0], device=N.device)
    return (cos.max(dim=1).values ** 2).mean()


class TopKSAE(nn.Module):
    def __init__(self, d_in, d_dict, k, aux_k=32):
        super().__init__()
        self.d_in, self.d_dict, self.k, self.aux_k = d_in, d_dict, k, aux_k
        W = torch.randn(d_dict, d_in)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("mean", torch.zeros(d_in))
        self.register_buffer("std", torch.ones(d_in))
        self.register_buffer("last_active", torch.zeros(d_dict, dtype=torch.long))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def set_norm(self, mean, std):
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-6))

    def norm(self, h):
        return (h - self.mean) / self.std

    def denorm(self, hn):
        return hn * self.std + self.mean

    def preacts(self, h):
        return (self.norm(h) - self.b_pre) @ self.W_enc + self.b_enc

    def encode(self, h):
        return topk_activations(self.preacts(h), self.k)

    def decode(self, f):
        return self.denorm(f @ self.W_dec + self.b_dec)

    def forward(self, h):
        f = self.encode(h)
        return self.decode(f), f

    def delta(self, df):
        """特徴の変化 df に対応する元スケールの活性変化 (誤差保存介入用)."""
        return (df @ self.W_dec) * self.std

    @torch.no_grad()
    def set_decoder_unit_norm(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def update_dead(self, f):
        self.step += 1
        self.last_active[(f > 0).any(0)] = self.step
        return self.step - self.last_active

    def aux_loss(self, h, recon, dead_mask):
        n_dead = int(dead_mask.sum())
        if n_dead == 0:
            return h.new_zeros(())
        resid = self.norm(h) - self.norm(recon)
        pre = self.preacts(h).masked_fill(~dead_mask, float("-inf")).nan_to_num(neginf=0.0)
        f_aux = topk_activations(pre, min(self.aux_k, n_dead))
        return F.mse_loss(f_aux @ self.W_dec, resid)


def load_sae(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = TopKSAE(ck["d_in"], ck["d_dict"], ck["k"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck
