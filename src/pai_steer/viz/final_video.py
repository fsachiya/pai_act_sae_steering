"""提出用の動画 (§10, 2分30秒以内, H.264 30fps, 1280x720). 保存済みのロールアウトの states を再描画する.

構成: タイトル → タスクとベースライン (4色) → 手法 → 色の介入 (介入なし / SAE / DiffMean) → E1 の結果
      → x/y 移動の介入 (従来 / 補正版, 届く方向と届かない方向) → 補正の効果 → 位置の比較・軌道の特徴 → まとめ
入力 artifacts/rollouts/grid_rollouts_seed0.npy (pattern_videos --grid), xy_corrected_rollouts_seed100.npy (xy_corrected_video)
出力 artifacts/videos/pai_final.mp4
"""
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from pai_steer.common import COLOR_JP, COLORS, FIGS, ROLLOUTS, VIDEOS, base_parser, ensure, load_config
from pai_steer.viz.pattern_videos import FPS, SIM_HZ, mark_steer
from pai_steer.viz.render import StateRenderer, font, put_text

W, H = 1280, 720
BG = (22, 26, 34)


def card(lines, sub=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    total = sum(int(40 * s) + 18 for _, s in lines)
    y = (H - total) // 2
    for s, scale in lines:
        f = font(int(40 * scale))
        d.text(((W - d.textlength(s, font=f)) / 2, y), s, font=f, fill=(240, 240, 240))
        y += int(40 * scale) + 18
    if sub:
        f = font(18)
        d.text(((W - d.textlength(sub, font=f)) / 2, H - 40), sub, font=f, fill=(150, 150, 150))
    return np.array(im)


def figure(path, caption, note=None):
    im = Image.open(path).convert("RGB")
    im.thumbnail((W - 40, H - (130 if note else 90)))
    c = Image.new("RGB", (W, H), (255, 255, 255))
    c.paste(im, ((W - im.width) // 2, 64 + (H - 64 - (50 if note else 0) - im.height) // 2))
    d = ImageDraw.Draw(c)
    f = font(30)
    d.text(((W - d.textlength(caption, font=f)) / 2, 14), caption, font=f, fill=(0, 0, 0))
    if note:
        f2 = font(20)
        d.text(((W - d.textlength(note, font=f2)) / 2, H - 42), note, font=f2, fill=(60, 60, 60))
    return np.array(c)


def clip(R, rec, color, label, speed, cam="agentview", hold=15):
    n_steps = len(rec["states"])
    states = list(rec["states"]) + [rec["final_state"]] * hold
    on_all = np.asarray(rec["steer_on"], bool)
    on = list(on_all) + [False] * hold
    n = int(len(states) / SIM_HZ / speed * FPS)
    out = []
    for i in range(n):
        t = min(len(states) - 1, int(i * speed * SIM_HZ / FPS))
        fr = R.render(states[t], color, cam)
        txt = label
        if t >= n_steps:
            pb = rec["placed_box"]
            txt += f"\n結果: {COLOR_JP[pb] + 'の箱' if isinstance(pb, str) and pb in COLOR_JP else '箱外'}"
        fr = put_text(fr, txt, "tl", sz=17)
        if on[t]:
            fr = put_text(fr, "● 介入中", "bl", sz=17, color=(230, 40, 40))
        out.append(mark_steer(fr, on[t], on_all, min(t, n_steps - 1), n_steps))
    return out


def row(clips):
    n = max(len(c) for c in clips)
    clips = [c + [c[-1]] * (n - len(c)) for c in clips]
    return [np.concatenate(fs, 1) for fs in zip(*clips)]


def fit(frames, title, speed):
    out = []
    for f in frames:
        im = Image.fromarray(f)
        im.thumbnail((W, H - 70))
        c = Image.new("RGB", (W, H), BG)
        c.paste(im, ((W - im.width) // 2, 60 + (H - 60 - im.height) // 2))
        d = ImageDraw.Draw(c)
        ft = font(26)
        d.text(((W - d.textlength(title, font=ft)) / 2, 16), title, font=ft, fill=(240, 240, 240))
        a = np.array(c)
        out.append(put_text(a, f"{speed:g}x", "br", sz=24, color=(255, 230, 0)))
    return out


def main():
    base_parser(__doc__).parse_args()
    cfg = load_config()
    ensure(VIDEOS)
    g0 = np.load(ROLLOUTS / "grid_rollouts_seed0.npy", allow_pickle=True).item()
    g1 = np.load(ROLLOUTS / "xy_corrected_rollouts_seed100.npy", allow_pickle=True).item()
    R = StateRenderer(cfg, size=420)
    sec = lambda s: int(s * FPS)
    out = []
    out += [card([("色キューブ仕分けタスクにおける", 1.0), ("ACT 方策への SAE 介入", 1.15), ("", .3),
                  ("学習済みの重みは固定し, 推論時の内部表現への介入だけで", .55),
                  ("キューブを別の色の箱へ入れられるか", .55)], "Physical AI 応用編1 最終課題")] * sec(6)
    out += [card([("タスク", .9), ("Panda がキューブ (1色) を掴み, 同じ色の箱へ入れる", .6),
                  ("robosuite 上の自作環境 / スクリプト方策のデモ 400 本で ACT を模倣学習", .5),
                  ("観測: 手先・キューブの位置姿勢 + キューブの色 (one-hot)", .5),
                  ("ベースライン成功率: 4色とも 100%", .55)])] * sec(6)
    # ベースライン 4色 (2x2)
    S = 2.0
    tiles = [clip(R, g0[("base", c, c)], c, f"{COLOR_JP[c]}キューブ → {COLOR_JP[c]}の箱", S) for c in COLORS]
    n = max(len(t) for t in tiles)
    tiles = [t + [t[-1]] * (n - len(t)) for t in tiles]
    grid = [np.concatenate([np.concatenate([tiles[0][i], tiles[1][i]], 1), np.concatenate([tiles[2][i], tiles[3][i]], 1)], 0)
            for i in range(n)]
    out += fit(grid, "ベースライン: 学習済み ACT (介入なし)", S)
    out += [card([("手法", .9), ("TopK SAE (8 か所) と DiffMean (平均活性の差) で介入", .55),
                  ("① 色 (意図レベル): encoder の color トークンの色の特徴を差し替え", .5),
                  ("② x/y 移動 (運動レベル): decoder 出力の移動方向の特徴を刺激", .5),
                  ("③ 軌道: decoder の色ごとの軌道パターン特徴を差し替え", .5),
                  ("介入は把持後 (持ち上げた後) だけ / 誤差を保存した介入", .45)])] * sec(8)
    # 色の介入
    S = 1.5
    t = [clip(R, g0[("base", "red", "red")], "red", "介入なし\n赤キューブ", S),
         clip(R, g0[("color_sae", "red", "blue")], "red", "色の介入 (SAE)\n赤 → 青の箱", S),
         clip(R, g0[("color_dm", "red", "yellow")], "red", "色の介入 (DiffMean)\n赤 → 黄の箱", S)]
    out += fit(row(t), "① 色の介入: 行き先の箱だけが切り替わる (把持は崩れない)", S)
    out += [figure(FIGS / "summary_matrix.png", "E1: 行き先の箱への到達率 (12 通り × 20 エピソード)",
                   "色: SAE 90% / DiffMean 100%   x/y 移動: 13% / 16%   軌道: 14% / 34%")] * sec(9)
    # x/y: 従来 vs 補正版
    S = 2.0
    t = [clip(R, g1[("xy_sae_old", "green", "red")], "green", "従来 (上位3特徴)\n緑 → 赤の箱 (+x)", S),
         clip(R, g1[("xy_corrected", "green", "red")], "green", "補正版\n緑 → 赤の箱 (+x)", S),
         clip(R, g1[("xy_corrected", "blue", "red")], "blue", "補正版\n青 → 赤の箱 (−y)", S)]
    out += fit(row(t), "② x/y 移動の介入: 補正で1軸の移動は届く / −y (目標より手前) は届かない", S)
    out += [figure(FIGS / "xy_corrected.png", "x/y 移動の介入の補正 (調整に使っていない seed で評価)",
                   "特徴の組み合わせ (直交成分を打ち消す) + 軸ごとの強さの較正で 16% → 58%。−y を含む方向は 0%")] * sec(9)
    out += [figure(FIGS / "layer_study.png", "SAE を適用する位置の比較 (8 か所)",
                   "色の介入は encoder の最初の層だけがよく効く")] * sec(6)
    out += [figure(FIGS / "traj_causal.png", "③ decoder 側の軌道の特徴 (因果的に選択)",
                   "約 50 特徴を状態に応じた値に差し替えれば 90% 切り替わるが, 決まった値では 2〜8%")] * sec(7)
    out += [card([("まとめ", .9),
                  ("行き先を自在に変えられるのは, 目的 (色) の表現への介入だけ", .55),
                  ("運動 (x/y) の介入は, 目的が決める止まる位置をずらす形でしか働かない", .5),
                  ("→ 補正すれば 1 軸の移動は 90〜100% 届くが, 目標より手前には置けない", .5),
                  ("decoder 側に「4 パターンの軌道」の記号はない (x と y の成分 × 状態依存)", .5),
                  ("限界: 色と行き先が常に 1 対 1 / 介入のタイミングにシミュレータの真の状態を使用", .42)])] * sec(12)
    path = VIDEOS / "pai_final.mp4"
    imageio.mimwrite(path, out, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=16)
    print("saved", path, f"{len(out) / FPS:.1f} 秒")


if __name__ == "__main__":
    main()
