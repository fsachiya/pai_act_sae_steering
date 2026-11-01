"""TopK SAE の学習 (§7.2). --site decoder_out | decoder_layer{i} | encoder_layer{i} (encoder_out = encoder_layer3).

encoder 層では robot / object / id の3トークンをまとめて [T*3, d] として学習する.
出力 artifacts/sae/sae_{site}.pt, 指標 (R², dead 割合, L0) を表示・保存.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pai_steer.common import ACTS, SAE_DIR, base_parser, ensure, get_device, load_config, set_seed
from pai_steer.sae.topk_sae import TopKSAE, orthogonality_loss


TOKENS = ["robot", "cube", "color"]


def load_site(site, token=None):
    if site.startswith("decoder"):
        return np.load(ACTS / f"{site}.npy", mmap_mode="r")
    x = np.load(ACTS / f"{site}.npy", mmap_mode="r")
    if token:
        return x[:, TOKENS.index(token)]
    return x.reshape(-1, x.shape[-1])


@torch.no_grad()
def evaluate(sae, x, dev, bs=65536):
    resid = tot = 0.0
    mean = x.mean(0)
    fired = torch.zeros(sae.d_dict, dtype=torch.bool, device=dev)
    l0 = 0.0
    for s in range(0, len(x), bs):
        xb = x[s:s + bs].to(dev)
        r, f = sae(xb)
        resid += ((xb - r) ** 2).sum().item()
        tot += ((xb - mean.to(dev)) ** 2).sum().item()
        fired |= (f > 0).any(0)
        l0 += (f > 0).float().sum().item()
    return dict(r2=1 - resid / tot, dead_frac=1 - fired.float().mean().item(), l0=l0 / len(x))


def main():
    p = base_parser(__doc__)
    p.add_argument("--site", default="decoder_out")
    p.add_argument("--token", default=None, help="encoder の1トークンだけで学習 (例: color)")
    args = p.parse_args()
    name = args.site + (f"_{args.token}" if args.token else "")
    cfg = load_config(args.config)
    sc = cfg["sae"]
    set_seed(cfg["seed"])
    ensure(SAE_DIR)
    dev = get_device()
    x = torch.from_numpy(np.asarray(load_site(args.site, args.token), dtype=np.float32))
    if args.quick:
        x = x[:50000]
    d = x.shape[1]
    k = sc["k_decoder"] if args.site.startswith("decoder") else sc["k_encoder"]
    sae = TopKSAE(d, sc["dict_mult"] * d, k).to(dev)
    sae.set_norm(x.mean(0).to(dev), x.std(0).to(dev))
    sae.set_decoder_unit_norm()
    opt = torch.optim.Adam(sae.parameters(), lr=sc["lr"])
    n = len(x)
    epochs = 1 if args.quick else sc["epochs"]
    steps = max(1, epochs * n // sc["batch"])
    g = torch.Generator().manual_seed(cfg["seed"])
    print(f"site={name} N={n} d={d} dict={sae.d_dict} k={k} steps={steps}")
    for step in tqdm(range(steps), mininterval=10):
        xb = x[torch.randint(0, n, (sc["batch"],), generator=g)].to(dev)
        recon, f = sae(xb)
        loss = F.mse_loss(sae.norm(recon), sae.norm(xb))
        dead = sae.update_dead(f)
        loss = loss + sc["aux_coef"] * sae.aux_loss(xb, recon, dead > sc["dead_steps"])
        if sc["ortho_gamma"] > 0:
            loss = loss + sc["ortho_gamma"] * orthogonality_loss(sae.W_dec)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.set_decoder_unit_norm()
    sae.eval()
    m = evaluate(sae, x, dev)
    print(json.dumps(m))
    torch.save(dict(model=sae.state_dict(), d_in=d, d_dict=sae.d_dict, k=k, site=args.site, token=args.token, metrics=m),
               SAE_DIR / f"sae_{name}{'_quick' if args.quick else ''}.pt")
    with open(SAE_DIR / f"metrics_{name}.json", "w") as fp:
        json.dump(m, fp, indent=1)


if __name__ == "__main__":
    main()
