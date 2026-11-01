"""提出用の動画 (§10, 2分30秒以内, H.264 30fps, 1280x720). 保存済みのロールアウトの states を再描画する.

構成: タイトル → 背景と目的 → タスクとベースライン (4色) → 手法 → 色・x/y 移動・軌道の介入
      → 網羅探索の結果 → 4つの要因と改善案 → 方向別 → x/y 移動の介入 (従来 / 改善案B) → 改善案B・A の効果 → まとめ
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
PAUSE = 3.0  # 動作の動画の終了後に止める秒数


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


def figure(path, caption, note=None, split=False):
    im = Image.open(path).convert("RGB")
    if split:  # 縦に2段の図は上下を左右に並べ直して大きく見せる
        h = im.height // 2
        top, bot = im.crop((0, 0, im.width, h)), im.crop((0, h, im.width, im.height))
        im = Image.new("RGB", (im.width * 2, h), (255, 255, 255))
        im.paste(top, (0, 0))
        im.paste(bot, (top.width, 0))
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


def clip(R, rec, color, label, speed, cam="agentview", hold=1, max_steps=None):
    states = list(rec["states"])
    on_all = np.asarray(rec["steer_on"], bool)
    cut = max_steps is not None and len(states) > max_steps  # 止まったまま長く続くエピソードは途中で打ち切る
    if cut:
        states, on_all = states[:max_steps], on_all[:max_steps]
    n_steps = len(states)
    states = states + [states[-1] if cut else rec["final_state"]] * hold
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
            if cut:
                txt += f"\n(全 {len(rec["states"])} ステップのうち {max_steps} ステップ以降は省略)"
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
    return out + [out[-1]] * int(PAUSE * FPS)  # 結果 (どの箱に入ったか) を確認できるよう最後の画面で止める


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
                  ("学習済みの重みは固定し, 推論時の内部の活性への介入だけで", .55),
                  ("キューブを別の色の箱へ入れられるか", .55)], "Physical AI 応用編1 最終課題")] * sec(6)
    out += [card([("背景と目的", .9),
                  ("学習済みの動作生成モデルを, 再学習せずに推論時の介入で調整したい", .5),
                  ("LLM では SAE / DiffMean による steering が提案されているが,", .5),
                  ("特徴を見つけられても (probing) 狙いどおり動かせるとは限らない (steering)", .5),
                  ("→ パラメータの少ない ACT で, probing と steering のギャップを検証する", .52),
                  ("想定: 生産ラインで, パーツ A を B の箱へ, B を A の箱へ運ぶように変更する", .45)])] * sec(8)
    out += [card([("タスク", .9), ("Panda がキューブ (4色のうち1色) を掴み, 同じ色の箱へ入れる", .55),
                  ("robosuite 上の自作環境 / スクリプト方策のデモ 400 本で ACT を模倣学習", .5),
                  ("観測: 手先・キューブの位置姿勢 + キューブの色 (one-hot)", .5),
                  ("x = 前後 (+x 奥), y = 左右 (+y 左) / ベースライン成功率: 4色とも 100%", .5)])] * sec(6)
    # ベースライン 4色 (2x2)
    S = 1.0
    tiles = [clip(R, g0[("base", c, c)], c, f"{COLOR_JP[c]}キューブ → {COLOR_JP[c]}の箱", S) for c in COLORS]
    n = max(len(t) for t in tiles)
    tiles = [t + [t[-1]] * (n - len(t)) for t in tiles]
    grid = [np.concatenate([np.concatenate([tiles[0][i], tiles[1][i]], 1), np.concatenate([tiles[2][i], tiles[3][i]], 1)], 0)
            for i in range(n)]
    out += fit(grid, "ベースライン: 学習済み ACT (介入なし)", S)
    out += [card([("手法", .9), ("TopK SAE (辞書 2048) と DiffMean (平均活性の差) で介入", .55),
                  ("① 色 (知覚レベル): encoder の color トークンの色の特徴を差し替え", .5),
                  ("② x/y 移動 (運動レベル): decoder 出力の移動方向の特徴を刺激", .5),
                  ("③ 軌道 (運動レベル): decoder 出力の行き先ごとの軌道の特徴を差し替え", .5),
                  ("介入は把持後だけ / 誤差を保存した介入 h' = h + W_dec (f' − f)", .45)])] * sec(8)
    # 色の介入
    S = 1.0
    t = [clip(R, g0[("base", "red", "red")], "red", "介入なし\n赤キューブ", S),
         clip(R, g0[("color_sae", "red", "blue")], "red", "色の介入 (SAE)\n赤 → 青の箱", S),
         clip(R, g0[("color_dm", "red", "yellow")], "red", "色の介入 (DiffMean)\n赤 → 黄の箱", S)]
    out += fit(row(t), "① 色の介入: 行き先の箱だけが切り替わる (把持は崩れない)", S)
    # x/y 移動・軌道の介入 (網羅探索で選んだ設定). 行き先へ届く例と, 元の箱へ戻る・別の箱や箱外へ外れる例
    S = 1.0
    t = [clip(R, g0[("xy_sae", "red", "blue")], "red", "x/y 移動 (SAE)\n赤 → 青の箱 (+y)", S),
         clip(R, g0[("xy_sae", "red", "green")], "red", "x/y 移動 (SAE)\n赤 → 緑の箱 (−x)", S),
         clip(R, g0[("xy_dm", "red", "green")], "red", "x/y 移動 (DiffMean)\n赤 → 緑の箱 (−x)", S)]
    out += fit(row(t), "② x/y 移動の介入: 届く方向もあるが, 別の箱や元の箱へ外れやすい", S)
    t = [clip(R, g0[("traj_dm", "red", "green")], "red", "軌道 (DiffMean)\n赤 → 緑の箱", S),
         clip(R, g0[("traj_sae", "red", "green")], "red", "軌道 (SAE)\n赤 → 緑の箱", S),
         clip(R, g0[("traj_dm", "red", "blue")], "red", "軌道 (DiffMean)\n赤 → 青の箱", S)]
    out += fit(row(t), "③ 軌道の介入: 切り替わる場合もあるが, 箱外や元の箱へ戻りやすい", S)
    out += [card([("網羅探索の結果 (166 条件, 9960 エピソード)", .8),
                  ("行き先の箱への最高到達率", .5), ("", .2),
                  ("① 色:  SAE 95%  /  DiffMean 100%", .62),
                  ("② x/y 移動:  SAE 18%  /  DiffMean 17%", .62),
                  ("③ 軌道:  SAE 22%  /  DiffMean 35%", .62), ("", .2),
                  ("位置・強度・特徴数をどう選んでも, 運動レベルの介入は効かない", .48)])] * sec(8)
    out += [card([("考察: 運動レベルの介入が効かない4つの要因", .75),
                  ("1. 色 (目的) の表現が元の箱へ引き戻す (平均 14% が元の箱へ戻る)", .47),
                  ("2. デモの動きの偏り: 運搬中 80% の時刻が +y (左), −y (右) は 3%", .47),
                  ("3. x/y の特徴が軸に沿っていない (例: +x の特徴が左へも動かす)", .47),
                  ("4. 「4 パターンの軌道」の特徴はなく, x と y の成分 × 状態依存で表現", .47), ("", .2),
                  ("改善案B: 特徴の組み合わせで軸からのずれを補正 (要因3)", .5),
                  ("改善案A: 色の特徴を抑えて引き戻しをなくす (要因1)", .5)])] * sec(12)
    out += [figure(FIGS / "direction_study.png", "要因2: 方向ごとの到達率",
                   "デモに多い +y (左) は届きやすく, デモにほぼない −y (右) はどの設定でも届きにくい")] * sec(7)
    # x/y: 従来 vs 補正版
    S = 1.0
    t = [clip(R, g1[("xy_sae_old", "green", "red")], "green", "従来 (上位3特徴)\n緑 → 赤の箱 (+x)", S),
         clip(R, g1[("xy_corrected", "green", "red")], "green", "改善案B\n緑 → 赤の箱 (+x)", S),
         clip(R, g1[("xy_corrected", "blue", "red")], "blue", "改善案B\n青 → 赤の箱 (−y)", S, max_steps=200)]
    out += fit(row(t), "改善案B の x/y 移動の介入: 補正で1軸の移動は届く / −y (右) は届かない", S)
    out += [figure(FIGS / "xy_corrected.png", "改善案B: 特徴の組み合わせ + 軸ごとの較正",
                   "到達率 16% → 58% (DiffMean 18%)。1軸は 98〜100%, −y を含む方向は 0%", split=True)] * sec(8)
    out += [figure(FIGS / "xy_corrected_suppress.png", "改善案A: 色の特徴の抑制を追加",
                   "+y を含む対角は 100% に上がるが, 1軸は大きく下がり全体は 46〜48% に低下", split=True)] * sec(8)
    out += [card([("まとめ", .9),
                  ("行き先を自在に変えられたのは, 目的 (色) の表現を差し替える介入だけ", .52),
                  ("運動 (x/y) の介入は, 目的が決める止まる位置をずらす形でしか働かない", .5),
                  ("→ 補正すれば 1 軸の移動は 98〜100% 届くが, −y (右) には置けない", .5),
                  ("SAE は特徴を組み合わせて DiffMean を上回れる (x/y: 58% vs 18%)", .5),
                  ("小さな動作生成モデルでも probing と steering のギャップがはっきり現れた", .5),
                  ("限界: 色と行き先が常に 1 対 1 / 介入のタイミングにシミュレータの真の状態を使用", .4)])] * sec(12)
    path = VIDEOS / "pai_final.mp4"
    imageio.mimwrite(path, out, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=16)
    print("saved", path, f"{len(out) / FPS:.1f} 秒")


if __name__ == "__main__":
    main()
