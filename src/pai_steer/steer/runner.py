"""並列ロールアウト実行器 (E0〜E3 共通).

job = dict(color, seed, steer (None / Steerer 引数 dict / そのリスト. 'sae_path', 'site' を含む),
           gate ('lift'|'all'), record_traj, record_states, その他の識別子は結果の job にそのまま残る)
"""
import multiprocessing as mp
import os

import numpy as np

_W = {}

GATES = {
    "lift": lambda info: info["ever_lifted"],   # 把持して持ち上げた後だけ
    "all": lambda info: True,                   # エピソードの最初から
    "transport": lambda info: info["high"],     # 運搬中で下ろし始める前だけ (下ろす・離すときは止める)
}


def _init(ckpt, cfg):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from pai_steer.env.cube_sort_env import make_env
    from pai_steer.policy.act import load_policy
    from pai_steer.policy.rollout import Policy

    model, _, stats = load_policy(ckpt, "cpu")
    _W.update(model=model, pol=Policy(model, stats), cfg=cfg, env=make_env(cfg, noise=True), saes={})


def _sae(path):
    from pai_steer.sae.topk_sae import load_sae

    _W.setdefault("saes", {})
    if path not in _W["saes"]:
        _W["saes"][path] = load_sae(path, "cpu")[0]
    return _W["saes"][path]


class MultiSteer:
    def __init__(self, steers):
        self.steers = steers

    @property
    def enabled(self):
        return self.steers[0].enabled

    @enabled.setter
    def enabled(self, v):
        for s in self.steers:
            s.enabled = v

    def remove(self):
        for s in self.steers:
            s.remove()


def build_steer(model, spec):
    from pai_steer.steer.hooks import Steerer

    if not spec:
        return None
    steers = []
    for s in (spec if isinstance(spec, list) else [spec]):
        s = dict(s)
        path = s.pop("sae_path", None)
        steers.append(Steerer(model, s.pop("site"), sae=_sae(path) if path else None, **s))
    return MultiSteer(steers)


class CFPatch:
    """診断用: 各推論で color だけ dst 色にした反実仮想の forward を回し, 位置 site の SAE 特徴のうち feats だけを
    その値に差し替える (状態に応じた値を使う). `.enabled` は rollout の gate で切り替わる."""

    def __init__(self, model, pol, site, feats, dst, sae):
        import torch

        from pai_steer.common import COLORS, site_module

        self.enabled, self.model, self.pol, self.sae = False, model, pol, sae
        self.feats = torch.as_tensor(feats, dtype=torch.long)
        self.dst = np.eye(4)[COLORS.index(dst)]
        self.mod = getattr(model, site_module(site))
        self.mode, self.cache = "off", None
        self.h = self.mod.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        if self.mode == "record":
            self.cache = o.detach().clone()
            return None
        if self.mode == "patch":
            f_src, f_dst = self.sae.encode(o), self.sae.encode(self.cache)
            f2 = f_src.clone()
            f2[..., self.feats] = f_dst[..., self.feats]
            return o + self.sae.delta(f2 - f_src)
        return None

    def __call__(self, robot, cube, color):
        if not self.enabled:
            return self.pol(robot, cube, color)
        self.mode = "record"
        self.pol(robot, cube, self.dst)
        self.mode = "patch"
        out = self.pol(robot, cube, color)
        self.mode = "off"
        return out

    def remove(self):
        self.h.remove()


def run_job(job):
    from pai_steer.policy.rollout import rollout

    cfg = _W["cfg"]
    if job.get("cf_patch"):
        c = job["cf_patch"]
        cf = CFPatch(_W["model"], _W["pol"], c["site"], c["feats"], job["dst"], _sae(c["sae_path"]))
        try:
            r = rollout(cf, _W["env"], job["color"], job["seed"], cfg, steer=cf, gate_fn=GATES[job.get("gate", "lift")])
        finally:
            cf.remove()
        out = {k: r[k] for k in ["placed_box", "success", "grasped", "length", "final_pos"]}
        out["job"] = {k: v for k, v in job.items() if k not in ("steer", "cf_patch")}
        return out
    steer = build_steer(_W["model"], job.get("steer"))
    try:
        r = rollout(_W["pol"], _W["env"], job["color"], job["seed"], cfg, steer=steer,
                    gate_fn=GATES[job.get("gate", "lift")], record_states=job.get("record_states", False),
                    color_input=job.get("color_input"))
    finally:
        if steer is not None:
            steer.remove()
    out = {k: r[k] for k in ["placed_box", "success", "grasped", "length", "final_pos"]}
    if job.get("record_traj") or job.get("record_states"):
        out.update(eef=r["eef"], cube=r["cube"], steer_on=r["steer"], phase=r["phase"])
    if job.get("record_states"):
        out.update(states=r["states"], final_state=r["final_state"])
    out["job"] = {k: v for k, v in job.items() if k != "steer"}
    return out


def run_jobs(jobs, ckpt, cfg, workers=None):
    workers = workers or min(8, os.cpu_count() or 4)
    ctx = mp.get_context("spawn")
    from tqdm import tqdm

    with ctx.Pool(workers, initializer=_init, initargs=(ckpt, cfg)) as pool:
        return list(tqdm(pool.imap(run_job, jobs, chunksize=1), total=len(jobs), mininterval=10))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)
