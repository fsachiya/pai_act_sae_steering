"""3パターン (色, x/y, 軌道) × SAE / DiffMean の推論動画.

同じ初期状態 (seed) で, 元の色のキューブを「介入なし」と「他の3色の箱へ介入」の4本並べる (上段 agentview, 下段 topdown).
出力 artifacts/videos/patterns/{条件}_{src}.mp4, overview_{src}_to_{dst}.mp4 (6条件 + 介入なしを並べた topdown)
   --grid: 4色すべてを1本で比較する 4×4 動画 (行 = キューブの色, 列 = 行き先の箱, 対角 = 介入なし, topdown)
           → artifacts/videos/patterns/grid_{条件}.mp4
"""
import imageio.v2 as imageio
import numpy as np

from pai_steer.common import CKPT, COLOR_JP, COLORS, ROLLOUTS, VIDEOS, base_parser, ensure, load_config
from pai_steer.env.layout import Layout
from pai_steer.steer.runner import run_jobs
from pai_steer.steer.specs import p1_dm_spec, p1_spec, p2_dm_spec, p2_spec, p3_dm_spec, p3_spec
from pai_steer.viz.render import StateRenderer, put_text

FPS, SIM_HZ = 30, 20


def conditions(L):
    """(名前, 見出し, spec 関数(src, dst))  強度は E1 と同じ."""
    return [
        ("color_sae", "色 / SAE swap β=2", lambda s, d: p2_spec(s, d, 2.0)),
        ("color_dm", "色 / DiffMean α=1.5", lambda s, d: p2_dm_spec(s, d, 1.5)),
        ("xy_sae", "x/y 移動 / SAE α=0.5", lambda s, d: p1_spec(L, s, d, 0.5)),
        ("xy_dm", "x/y 移動 / DiffMean α=1", lambda s, d: p1_dm_spec(L, s, d, 1.0)),
        ("traj_sae", "軌道 / SAE swap β=8", lambda s, d: p3_spec(s, d, 8.0, per_k=True)),
        ("traj_dm", "軌道 / DiffMean α=1", lambda s, d: p3_dm_spec(s, d, 1.0)),
    ]


STEER_RGB = (230, 40, 40)      # 介入中の枠と文字
BAR_ON = (255, 150, 0)         # 時間バー: 介入区間
BAR_OFF = (90, 90, 90)


def mark_steer(fr, on_now, on_all, t, n_steps, bar_h=10, border=5):
    """介入のタイミングを描く: 介入中は赤枠, 下端に時間バー (オレンジ = 介入区間, 白線 = 現在)."""
    fr = fr.copy()
    H, W = fr.shape[:2]
    if on_now:
        fr[:border] = STEER_RGB; fr[-border:] = STEER_RGB; fr[:, :border] = STEER_RGB; fr[:, -border:] = STEER_RGB
    y0 = H - border - bar_h if on_now else H - bar_h
    y1 = y0 + bar_h
    xs = np.linspace(0, W, n_steps + 1).astype(int)
    bar = np.empty((bar_h, W, 3), np.uint8)
    bar[:] = BAR_OFF
    for i in np.where(on_all)[0]:
        bar[:, xs[i]:xs[i + 1]] = BAR_ON
    cx = min(W - 2, xs[min(t, n_steps - 1)])
    bar[:, max(0, cx - 1):cx + 2] = 255
    fr[y0:y1] = bar
    return fr


def clip(R, rec, color, label, speed, cams):
    n_steps = len(rec["states"])
    states = list(rec["states"]) + [rec["final_state"]] * 12
    on_all = np.asarray(rec["steer_on"], bool)
    on = list(on_all) + [False] * 12
    t_on = int(np.argmax(on_all)) if on_all.any() else None
    n = int(len(states) / SIM_HZ / speed * FPS)
    out = []
    for i in range(n):
        t = min(len(states) - 1, int(i * speed * SIM_HZ / FPS))
        fr = np.concatenate([R.render(states[t], color, c) for c in cams], 0)
        txt = label
        if t >= n_steps:
            txt += f"\n結果: {COLOR_JP.get(rec['placed_box'], '箱外')}" + ("の箱" if rec["placed_box"] else "")
        fr = put_text(fr, txt, "tl", sz=17)
        if on[t]:
            fr = put_text(fr, f"● 介入中 (step {t_on}〜)", "bl", sz=17, color=STEER_RGB)
        elif t_on is None:
            fr = put_text(fr, "介入なし", "bl", sz=15, color=(200, 200, 200))
        fr = put_text(fr, f"step {min(t, n_steps)}", "tr", sz=14)
        out.append(mark_steer(fr, on[t], on_all, min(t, n_steps - 1), n_steps))
    return out


def tile(clips):
    n = max(len(c) for c in clips)
    clips = [c + [c[-1]] * (n - len(c)) for c in clips]
    return [np.concatenate(fs, 1) for fs in zip(*clips)]


def grid_videos(cfg, args, L, conds, out_dir):
    cache = ROLLOUTS / f"grid_rollouts_seed{args.seed}.npy"
    if args.reuse and cache.exists():
        res = np.load(cache, allow_pickle=True).item()
    else:
        jobs = [dict(name="base", color=s, dst=s, seed=args.seed, record_states=True) for s in COLORS]
        for name, _, fn in conds:
            for s in COLORS:
                for d in COLORS:
                    if d != s:
                        jobs.append(dict(name=name, color=s, dst=d, seed=args.seed, record_states=True, steer=fn(s, d)))
        res = {(r["job"]["name"], r["job"]["color"], r["job"]["dst"]): r for r in run_jobs(jobs, args.ckpt, cfg)}
        ensure(ROLLOUTS)
        np.save(cache, res, allow_pickle=True)
    R = StateRenderer(cfg, size=240)
    cam = (args.cam,)
    base = {s: clip(R, res[("base", s, s)], s, f"{COLOR_JP[s]}: 介入なし", args.speed, cam) for s in COLORS}
    for name, title, _ in conds:
        rows, n_ok = [], 0
        for s in COLORS:
            row = []
            for d in COLORS:
                if d == s:
                    row.append(base[s])
                    continue
                r = res[(name, s, d)]
                n_ok += r["placed_box"] == d
                mark = "○" if r["placed_box"] == d else "×"
                row.append(clip(R, r, s, f"{mark} {COLOR_JP[s]} → {COLOR_JP[d]}の箱", args.speed, cam))
            rows.append(row)
        n = max(len(c) for row in rows for c in row)
        rows = [[c + [c[-1]] * (n - len(c)) for c in row] for row in rows]
        frames = []
        for i in range(n):
            f = np.concatenate([np.concatenate([c[i] for c in row], 1) for row in rows], 0)
            head = np.full((60, f.shape[1], 3), 25, np.uint8)
            head = put_text(head, f"{title}   行 = キューブの色 / 列 = 行き先の箱 (対角 = 介入なし)   成功 {n_ok}/12", "tl", sz=20)
            head = put_text(head, "赤枠 = 介入中 / 下端のバー: オレンジ = 介入した区間, 白線 = 現在", "bl", sz=15, color=(255, 170, 60))
            frames.append(put_text(np.concatenate([head, f], 0), f"{args.speed:g}x", "br", sz=22, color=(255, 230, 0)))
        imageio.mimwrite(out_dir / f"grid_{name}_{args.cam}{'' if args.speed == 1 else f'_{args.speed:g}x'}.mp4", frames, fps=FPS, codec="libx264", quality=7,
                         pixelformat="yuv420p", macro_block_size=16)
        print(name, f"success {n_ok}/12")


def main():
    p = base_parser(__doc__)
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--src", default="red")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.5)
    p.add_argument("--overview-dst", default="yellow")
    p.add_argument("--grid", action="store_true", help="4色すべての 4×4 比較動画を作る")
    p.add_argument("--cam", default="agentview", help="--grid のカメラ (agentview / topdown)")
    p.add_argument("--reuse", action="store_true", help="--grid: 保存済みのロールアウトを使い回す")
    args = p.parse_args()
    cfg = load_config(args.config)
    L = Layout(cfg["env"])
    out_dir = VIDEOS / "patterns"
    ensure(out_dir)
    if args.grid:
        grid_videos(cfg, args, L, conditions(L), out_dir)
        return
    src, dsts = args.src, [c for c in COLORS if c != args.src]
    conds = conditions(L)
    jobs = [dict(name="base", color=src, dst=src, seed=args.seed, record_states=True)]
    for name, _, fn in conds:
        for d in dsts:
            jobs.append(dict(name=name, color=src, dst=d, seed=args.seed, record_states=True, steer=fn(src, d)))
    res = {(r["job"]["name"], r["job"]["dst"]): r for r in run_jobs(jobs, args.ckpt, cfg)}
    R = StateRenderer(cfg, size=320)
    base = res[("base", src)]
    print("base", base["placed_box"])
    base_clip = clip(R, base, src, f"介入なし: {COLOR_JP[src]}キューブ", args.speed, ("agentview", "topdown"))
    for name, title, _ in conds:
        clips = [base_clip]
        for d in dsts:
            r = res[(name, d)]
            print(name, d, "→", r["placed_box"])
            clips.append(clip(R, r, src, f"{title}\n{COLOR_JP[src]} → {COLOR_JP[d]}の箱", args.speed, ("agentview", "topdown")))
        frames = tile(clips)
        frames = [put_text(f, f"{args.speed:g}x", "br", sz=20, color=(255, 230, 0)) for f in frames]
        imageio.mimwrite(out_dir / f"{name}_{src}.mp4", frames, fps=FPS, codec="libx264", quality=7,
                         pixelformat="yuv420p", macro_block_size=16)
    # 概観: 介入なし + 6条件 (行き先 overview_dst), topdown のみ 2 行 × 4 列
    d = args.overview_dst
    tops = [clip(R, base, src, f"介入なし: {COLOR_JP[src]}キューブ", args.speed, ("topdown",))]
    for name, title, _ in conds:
        tops.append(clip(R, res[(name, d)], src, f"{title}\n→ {COLOR_JP[d]}の箱", args.speed, ("topdown",)))
    n = max(len(c) for c in tops)
    tops = [c + [c[-1]] * (n - len(c)) for c in tops]
    blank = [np.zeros_like(tops[0][0])] * n
    grid = [np.concatenate([np.concatenate([tops[0][i], tops[1][i], tops[3][i], tops[5][i]], 1),
                            np.concatenate([blank[i], tops[2][i], tops[4][i], tops[6][i]], 1)], 0) for i in range(n)]
    grid = [put_text(f, f"{args.speed:g}x", "br", sz=20, color=(255, 230, 0)) for f in grid]
    imageio.mimwrite(out_dir / f"overview_{src}_to_{d}.mp4", grid, fps=FPS, codec="libx264", quality=7,
                     pixelformat="yuv420p", macro_block_size=16)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
