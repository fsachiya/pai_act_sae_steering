"""箱・キューブの配置, 色の定義, 箱判定 box_of(xy)."""
import numpy as np

from pai_steer.common import COLORS

RGB = {
    "red": (0.85, 0.15, 0.15),
    "green": (0.15, 0.70, 0.20),
    "blue": (0.15, 0.30, 0.85),
    "yellow": (0.90, 0.80, 0.10),
}


class Layout:
    """config['env'] から作る配置情報. 座標は机の中心を原点とする (x 奥, y 左)."""

    def __init__(self, env_cfg):
        self.c = env_cfg
        self.table_top = env_cfg["table_offset"][2]
        self.box_xy = {k: np.array(v, dtype=float) for k, v in env_cfg["boxes"].items()}
        self.inner = env_cfg["box_inner"]
        self.wall_h = env_cfg["box_wall_height"]
        self.wall_t = env_cfg["box_wall_thickness"]
        self.cube_xy = np.array(env_cfg["cube_xy"], dtype=float)
        self.cube_size = env_cfg["cube_size"]

    @property
    def wall_top(self):
        """箱の壁の上端の z (絶対座標)."""
        return self.table_top + self.wall_h

    def box_of(self, xy):
        """xy がどの箱の内寸の中にあるか. 無ければ None."""
        for c in COLORS:
            d = np.abs(np.asarray(xy[:2]) - self.box_xy[c])
            if (d < self.inner / 2).all():
                return c
        return None

    def to_dict(self):
        return dict(boxes={k: v.tolist() for k, v in self.box_xy.items()}, inner=self.inner, wall_height=self.wall_h,
                    wall_thickness=self.wall_t, cube_xy=self.cube_xy.tolist(), cube_size=self.cube_size,
                    table_top=self.table_top)
