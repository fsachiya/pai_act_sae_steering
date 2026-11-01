"""フック位置の活性を学習データ全時刻 (z=0) について収集する (§7.1).

出力 artifacts/acts/:
  decoder_out.npy, decoder_layer{0,1,2}.npy [N_dec*K, d] float16, dec_meta.npz (t: データセット時刻, k: チャンク内位置)
  encoder_layer{i}.npy [N_enc, 3, d] float16 (トークン robot, cube, color), enc_meta.npz (t)
  (encoder_out は encoder_layer{L-1} と同一の値)
  labels.npz: データセット時刻ごとの color_id, phase, action(元スケール), eef, end_of
"""
import numpy as np
import torch
from tqdm import tqdm

from pai_steer.common import ACTS, CKPT, DATA_HDF5, base_parser, ensure, get_device, load_config, site_module
from pai_steer.data.dataset import ChunkDataset
from pai_steer.policy.act import load_policy


class Capture:
    def __init__(self, model, names):
        self.out = {}
        self.hs = [getattr(model, n).register_forward_hook(self._hook(n)) for n in names]

    def _hook(self, name):
        def f(mod, inp, out):
            self.out[name] = out.detach()
        return f


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--data", default=str(DATA_HDF5))
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure(ACTS)
    dev = get_device()
    model, mcfg, stats = load_policy(args.ckpt, dev)
    ds = ChunkDataset(path=args.data, chunk=mcfg["act"]["chunk"], stats=stats, max_demos=20 if args.quick else None)
    n = len(ds)
    np.savez(ACTS / "labels.npz", color_id=ds.color_id, phase=ds.phase, action=ds.act, eef=ds.eef, end_of=ds.end_of)
    nl = mcfg["act"]["enc_layers"]
    dec_sites = [f"decoder_layer{i}" for i in range(mcfg["act"]["dec_layers"] - 1)] + ["decoder_out"]
    names = [site_module(x) for x in dec_sites] + [f"encoder_layer{i}_out" for i in range(nl)]
    cap = Capture(model, names)
    sc = cfg["sae"]
    t_dec, t_enc = np.arange(0, n, sc["stride_dec"]), np.arange(0, n, sc["stride_enc"])
    K, d = mcfg["act"]["chunk"], mcfg["act"]["d_model"]
    decs = {x: np.lib.format.open_memmap(ACTS / f"{x}.npy", mode="w+", dtype=np.float16, shape=(len(t_dec) * K, d))
            for x in dec_sites}
    encs = [np.lib.format.open_memmap(ACTS / f"encoder_layer{i}.npy", mode="w+", dtype=np.float16,
                                      shape=(len(t_enc), 3, d)) for i in range(nl)]
    pos_dec = -np.ones(n, dtype=np.int64); pos_dec[t_dec] = np.arange(len(t_dec))
    pos_enc = -np.ones(n, dtype=np.int64); pos_enc[t_enc] = np.arange(len(t_enc))
    ts = np.union1d(t_dec, t_enc)
    with torch.no_grad():
        for s in tqdm(range(0, len(ts), 2048), mininterval=10):
            idx = ts[s:s + 2048]
            b = ds.batch(idx)
            model(*(torch.from_numpy(b[k]).to(dev) for k in ["robot", "cube", "color"]))
            m = pos_dec[idx] >= 0
            qs = pos_dec[idx][m]
            rows = (qs[:, None] * K + np.arange(K)[None]).reshape(-1)
            for x in dec_sites:
                dout = cap.out[site_module(x)][torch.from_numpy(np.where(m)[0]).to(dev)].cpu().numpy().astype(np.float16)
                decs[x][rows] = dout.reshape(-1, d)
            m = pos_enc[idx] >= 0
            for li in range(nl):
                encs[li][pos_enc[idx][m]] = cap.out[f"encoder_layer{li}_out"][torch.from_numpy(np.where(m)[0]).to(dev), 1:].cpu().numpy().astype(np.float16)
    [x.flush() for x in decs.values()]; [e.flush() for e in encs]
    np.savez(ACTS / "dec_meta.npz", t=np.repeat(t_dec, K), k=np.tile(np.arange(K), len(t_dec)))
    np.savez(ACTS / "enc_meta.npz", t=t_enc, tokens=np.array(["robot", "cube", "color"]))
    print("decoder sites", list(decs), decs["decoder_out"].shape, "encoder layers", encs[0].shape)


if __name__ == "__main__":
    main()
