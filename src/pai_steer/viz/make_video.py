"""動画 (§10): 保存した states を再描画し mp4 (H.264, 30fps) にする. agentview と topdown を左右に並べる.

1. --record で必要なロールアウトを実行し states を artifacts/rollouts/video_*.npy に保存
2. タイトル → ベースライン4画面 → P2 左右比較 ×2 → P1 α 段階 → 図 → まとめ
"""
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from pai_steer.common import CKPT, COLOR_JP, COLORS, FIGS, ROLLOUTS, VIDEOS, base_parser, ensure, load_config
from pai_steer.env.layout import Layout
from pai_steer.viz.render import StateRenderer, font, put_text

FPS, SIM_HZ, W, H = 30, 20, 1280, 720


def text_card(lines, bg=(22, 26, 34)):
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    y = H // 2 - sum(int(40 * s) + 16 for _, s in lines) // 2
    for s, scale in lines:
        f = font(int(40 * scale))
        d.text(((W - d.textlength(s, font=f)) / 2, y), s, font=f, fill=(240, 240, 240))
        y += int(40 * scale) + 16
    return np.array(im)


def image_card(path, caption):
    im = Image.open(path).convert("RGB")
    im.thumbnail((W - 40, H - 80))
    c = Image.new("RGB", (W, H), (255, 255, 255))
    c.paste(im, ((W - im.width) // 2, 60 + (H - 60 - im.height) // 2))
    d = ImageDraw.Draw(c)
    f = font(32)
    d.text(((W - d.textlength(caption, font=f)) / 2, 14), caption, font=f, fill=(0, 0, 0))
    return np.array(c)


def fit(fr):
    im = Image.fromarray(fr)
    im.thumbnail((W, H))
    c = Image.new("RGB", (W, H), (0, 0, 0))
    c.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    return np.array(c)


def pad(seqs):
    n = max(len(s) for s in seqs)
    return [s + [s[-1]] * (n - len(s)) for s in seqs]


def hcat(*seqs):
    return [np.concatenate(f, 1) for f in zip(*pad(list(seqs)))]


def vcat(*seqs):
    return [np.concatenate(f, 0) for f in zip(*pad(list(seqs)))]


def clip_frames(R, rec, speed, size_label, tl, steer_mark=True, cams=("agentview", "topdown")):
    """1ロールアウト → (agentview | topdown) のフレーム列 (speed 倍速, 30fps)."""
    states = list(rec["states"]) + [rec["final_state"]] * 10
    on = list(rec["steer_on"]) + [False] * 10
    n_out = int(len(states) / SIM_HZ / speed * FPS)
    frames = []
    for i in range(n_out):
        t = min(len(states) - 1, int(i * speed * SIM_HZ / FPS))
        fr = np.concatenate([R.render(states[t], rec["color"], c) for c in cams], 1)
        txt = tl + ("\n● 介入中" if steer_mark and on[t] else "")
        fr = put_text(fr, txt, "tl", sz=size_label)
        fr = put_text(fr, f"{speed:g}x", "br", sz=size_label, color=(255, 230, 0))
        frames.append(fr)
    return frames


def record(cfg, args):
    from pai_steer.steer.runner import run_jobs
    from pai_steer.steer.specs import p1_spec, p2_spec

    L = Layout(cfg["env"])
    s = args.seed
    jobs = [dict(name=f"base_{c}", color=c, seed=s, record_states=True) for c in COLORS]
    for ex in args.p2_examples.split(","):
        src, dst = ex.split(":")
        jobs.append(dict(name=f"p2_{src}_{dst}", color=src, seed=s, record_states=True, gate=args.p2_gate,
                         steer=p2_spec(src, dst, args.p2_beta, all_tokens=args.p2_all)))
    src, dst = args.p1_example.split(":")
    for a in map(float, args.p1_alphas.split(",")):
        jobs.append(dict(name=f"p1_{src}_{dst}_a{a:g}", color=src, seed=s, record_states=True,
                         steer=None if a == 0 else p1_spec(L, src, dst, a)))
    ensure(ROLLOUTS)
    for r in run_jobs(jobs, args.ckpt, cfg):
        j = r["job"]
        np.save(ROLLOUTS / f"video_{j['name']}.npy", dict(color=j["color"], states=r["states"], final_state=r["final_state"],
                                                          steer_on=r["steer_on"], placed_box=r["placed_box"]), allow_pickle=True)
        print(j["name"], "placed", r["placed_box"])


def load(name):
    return np.load(ROLLOUTS / f"video_{name}.npy", allow_pickle=True).item()


def build(cfg, args):
    ensure(VIDEOS)
    R = StateRenderer(cfg, size=320)
    out = []
    sec = lambda x: int(x * FPS)
    out += [text_card([("色キューブ仕分けタスクにおける", 1.0), ("ACT 方策への SAE 介入", 1.1), ("", .5),
                       ("Panda がキューブを同じ色の箱へ入れる方策を模倣学習 (robosuite)", .55),
                       ("学習済みの重みは固定し, 推論時の SAE 特徴への介入だけで行き先を変える", .55),
                       ("P1: 運動レベル (xy 移動特徴)    P2: 意図レベル (目標特徴)", .55)])] * sec(9)
    # 2. ベースライン (2x2, 各セル agentview|topdown, 2倍速)
    cells = []
    for c in COLORS:
        r = load(f"base_{c}")
        r["color"] = c
        cells.append(clip_frames(R, r, 2.0, 18, f"ベースライン: {COLOR_JP[c]}キューブ → {COLOR_JP[c]}の箱", steer_mark=False))
    out += [fit(f) for f in vcat(hcat(cells[0], cells[1]), hcat(cells[2], cells[3]))]
    # 3. P2: 左右比較 (上: 介入なし, 下: 介入あり)
    for ex in args.p2_examples.split(","):
        src, dst = ex.split(":")
        b, st = load(f"base_{src}"), load(f"p2_{src}_{dst}")
        b["color"] = st["color"] = src
        top = clip_frames(R, b, 1.5, 20, f"介入なし: {COLOR_JP[src]}キューブ (結果: {COLOR_JP.get(b['placed_box'], '箱外')}の箱)", False)
        bot = clip_frames(R, st, 1.5, 20, f"P2: {COLOR_JP[src]} → {COLOR_JP[dst]}の箱  β = {args.p2_beta:g} "
                                          f"(結果: {COLOR_JP.get(st['placed_box'], '箱外')}の箱)")
        out += [fit(f) for f in vcat(top, bot)]
    # 4. P1: α 段階 (topdown のみを横に並べる)
    src, dst = args.p1_example.split(":")
    tiles = []
    for a in map(float, args.p1_alphas.split(",")):
        r = load(f"p1_{src}_{dst}_a{a:g}")
        r["color"] = src
        tiles.append(clip_frames(R, r, 2.0, 16, f"P1: {COLOR_JP[src]} → {COLOR_JP[dst]}の箱\nα = {a:g} → {COLOR_JP.get(r['placed_box'], '箱外')}",
                                 cams=("topdown",)))
    out += [fit(f) for f in hcat(*tiles)]
    # 5. 図
    for pat, cap in args.figs:
        cands = sorted(FIGS.glob(f"{pat}.png"))
        if cands:
            out += [image_card(cands[0], cap)] * sec(6)
    # 6. まとめ
    out += [text_card([("まとめ", 1.0)] + [(l, .55) for l in args.summary.split("|")])] * sec(10)
    path = VIDEOS / "pai_final.mp4"
    imageio.mimwrite(path, out, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=16)
    print("saved", path, f"{len(out) / FPS:.1f}s")


def main():
    p = base_parser(__doc__)
    p.add_argument("--record", action="store_true")
    p.add_argument("--ckpt", default=str(CKPT / "act.pt"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p2-examples", default="red:blue,red:yellow")
    p.add_argument("--p2-beta", type=float, default=1.0)
    p.add_argument("--p2-all", action="store_true")
    p.add_argument("--p2-gate", default="lift")
    p.add_argument("--p1-example", default="red:yellow")
    p.add_argument("--p1-alphas", default="0,2,4,8")
    p.add_argument("--summary", default="P1 (運動レベル): 置き位置を連続的にずらす|P2 (意図レベル): 行き先の箱を切り替える")
    args = p.parse_args()
    args.figs = [("matrix_p2", "E1: P2 の 4×4 行列 (行き先の箱への到達率)"), ("matrix_p1", "E1: P1 の 4×4 行列"),
                 ("sweep_p2", "E2: P2 の強度スイープ"), ("sweep_p1", "E2: P1 の強度スイープ")]
    cfg = load_config(args.config)
    if args.record:
        record(cfg, args)
    build(cfg, args)


if __name__ == "__main__":
    main()
