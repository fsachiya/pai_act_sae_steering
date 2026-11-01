"""ACT の学習."""
import time

import numpy as np
import torch

from pai_steer.common import CKPT, DATA_HDF5, LOGS, STATS, base_parser, ensure, get_device, load_config, set_seed
from pai_steer.data.dataset import ChunkDataset
from pai_steer.policy.act import act_loss, build_act


def main():
    p = base_parser(__doc__)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--name", default="act")
    p.add_argument("--data", default=None, help="cube_sort.hdf5 以外を使う場合")
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure(CKPT, LOGS)
    dev = get_device()
    a = cfg["act"]
    steps = args.steps or (a["quick_steps"] if args.quick else a["steps"])
    ds = ChunkDataset(path=args.data or DATA_HDF5, chunk=a["chunk"], max_demos=40 if args.quick else None)
    ds.save_stats(STATS)
    print(f"dataset: {len(ds)} steps, {len(ds.demo_range)} demos, device={dev}")
    model = build_act(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a["lr"], weight_decay=a["wd"])
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(ds), size=a["batch"])
        b = {k: torch.from_numpy(v).to(dev) for k, v in ds.batch(idx).items()}
        pred, mu, logvar = model(b["robot"], b["cube"], b["color"], b["actions"], b["is_pad"])
        loss, l1, kl = act_loss(pred, b["actions"], b["is_pad"], mu, logvar, a["kl_weight"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append([loss.item(), l1.item(), kl.item()])
        if step % 100 == 0 or step == 1:
            m = np.mean(hist[-100:], 0)
            print(f"step {step} loss {m[0]:.4f} l1 {m[1]:.4f} kl {m[2]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step % 5000 == 0 or step == steps:
            torch.save(dict(model=model.state_dict(), cfg=cfg, stats=ds.stats, step=step),
                       CKPT / f"{args.name}{'_quick' if args.quick else ''}.pt")
    np.save(LOGS / f"{args.name}_loss.npy", np.array(hist))


if __name__ == "__main__":
    main()
