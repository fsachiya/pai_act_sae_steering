"""状態列 → 画像列 (保存した MuJoCo 状態を offscreen で再描画する)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pai_steer.env.cube_sort_env import make_env

FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"


def font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except OSError:
        return ImageFont.load_default()


class StateRenderer:
    def __init__(self, cfg, size=384, cameras=("agentview", "topdown")):
        self.env = make_env(cfg, noise=False, render=True, camera_names=cameras, camera_size=size)
        self.env.reset()
        self.size, self.cameras = size, cameras

    def render(self, state, color, cam):
        e = self.env
        e.apply_color(color)
        e.set_state(state)
        return e.sim.render(camera_name=cam, width=self.size, height=self.size)[::-1].copy()

    def frames(self, states, color, cam):
        return [self.render(s, color, cam) for s in states]


def put_text(img, text, pos="tl", sz=20, color=(255, 255, 255)):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = font(sz)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        w = d.textlength(line, font=f)
        x = 8 if pos in ("tl", "bl") else im.width - w - 8
        y = 6 + i * (sz + 4) if pos in ("tl", "tr") else im.height - (len(lines) - i) * (sz + 4) - 6
        d.text((x, y), line, font=f, fill=color, stroke_width=3, stroke_fill=(0, 0, 0))
    return np.array(im)
