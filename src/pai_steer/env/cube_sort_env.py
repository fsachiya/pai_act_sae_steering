"""色キューブ仕分け環境 (§3). robosuite 1.5 ManipulationEnv のサブクラス (Lift を雛形).

机の上に色の違う4つの箱 (固定) と, 1色のキューブが1つ. Panda がキューブを同じ色の箱へ入れる.
"""
import logging

import numpy as np
import robosuite
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.objects.composite.bin import Bin
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat

from pai_steer.common import COLORS
from pai_steer.env.layout import RGB, Layout

ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)   # controller 設定の警告がリセットごとに出るのを抑制


class FixedBin(Bin):
    """自由関節を持たない(机に固定された)上の開いた箱."""

    def _get_geom_attrs(self):
        a = super()._get_geom_attrs()
        a["joints"] = None
        return a


def yaw_quat_wxyz(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class CubeSortEnv(ManipulationEnv):
    def __init__(self, cfg, cube_color=None, noise=True, robots="Panda", has_offscreen_renderer=False,
                 use_camera_obs=False, camera_names=("agentview", "topdown"), camera_size=None,
                 horizon=None, **kwargs):
        self.cfg = cfg
        e = cfg["env"]
        self.layout = Layout(e)
        self.cube_color = cube_color
        self.noise = noise
        self.color = cube_color or "red"
        self.rng = np.random.default_rng(0)
        self.table_full_size = tuple(e["table_size"])
        self.table_offset = np.array(e["table_offset"])
        cs = camera_size or e["camera_size"]
        ctrl = load_composite_controller_config(controller="BASIC", robot="Panda")
        init_noise = {"magnitude": e["noise"]["robot_qpos"], "type": "uniform"} if noise else None
        super().__init__(
            robots=robots,
            controller_configs=ctrl,
            gripper_types="default",
            initialization_noise=init_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=False,
            has_offscreen_renderer=has_offscreen_renderer or use_camera_obs,
            render_camera="agentview",
            control_freq=e["control_freq"],
            horizon=horizon or e["horizon"],
            ignore_done=True,
            hard_reset=False,
            camera_names=list(camera_names),
            camera_heights=cs,
            camera_widths=cs,
            **kwargs,
        )

    # ---------- 乱数 ----------
    def set_seed(self, s):
        """初期状態ノイズ用の乱数 (robosuite の関節ノイズは np.random を使うので併せて設定)."""
        self.rng = np.random.default_rng(s)
        np.random.seed(s)

    # ---------- モデル ----------
    def _load_model(self):
        super()._load_model()
        L = self.layout
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=(1, 0.005, 0.0001),
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        # 真上から見下ろすカメラ: 画像の上 = +x (奥), 左 = +y
        pts = np.array(list(L.box_xy.values()) + [L.cube_xy])
        cx, cy = (pts.min(0) + pts.max(0)) / 2
        arena.set_camera(camera_name="topdown", pos=[float(cx), float(cy), L.table_top + 0.95],
                         quat=[0.7071068, 0.0, 0.0, -0.7071068])
        outer = L.inner + 2 * L.wall_t
        self.boxes = {}
        for c in COLORS:
            b = FixedBin(name=f"box_{c}", bin_size=(outer, outer, L.wall_h), wall_thickness=L.wall_t,
                         transparent_walls=False, use_texture=False, rgba=(*RGB[c], 1.0), friction=(1.0, 0.005, 0.0001))
            xy = L.box_xy[c]
            b.get_obj().set("pos", array_to_string([xy[0], xy[1], L.table_top + L.wall_h / 2]))
            self.boxes[c] = b
        h = L.cube_size / 2
        self.cube = BoxObject(name="cube", size=[h, h, h], rgba=(*RGB[self.color], 1.0),
                              friction=(self.cfg["env"]["cube_friction"], 0.005, 0.0001))
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=[self.cube] + list(self.boxes.values()))

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.cube_geom_ids = [self.sim.model.geom_name2id(g) for g in self.cube.visual_geoms + self.cube.contact_geoms]
        self.eef_site_id = self.robots[0].eef_site_id["right"]

    def _setup_observables(self):
        observables = super()._setup_observables()
        m = "object"

        @sensor(modality=m)
        def cube_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.cube_body_id])

        @sensor(modality=m)
        def cube_quat(obs_cache):
            return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

        @sensor(modality=m)
        def cube_to_eef_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.cube_body_id]) - np.array(self.sim.data.site_xpos[self.eef_site_id])

        @sensor(modality=m)
        def cube_color_onehot(obs_cache):
            return np.eye(4)[COLORS.index(self.color)]

        for s in [cube_pos, cube_quat, cube_to_eef_pos, cube_color_onehot]:
            observables[s.__name__] = Observable(name=s.__name__, sensor=s, sampling_rate=self.control_freq)
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        L, nz = self.layout, self.cfg["env"]["noise"]
        self.apply_color(self.cube_color or COLORS[self.rng.integers(4)])
        xy = L.cube_xy.copy()
        yaw = 0.0
        if self.noise:
            xy += self.rng.uniform(-nz["cube_xy"], nz["cube_xy"], 2)
            yaw = np.deg2rad(self.rng.uniform(-nz["cube_yaw_deg"], nz["cube_yaw_deg"]))
        pos = [xy[0], xy[1], L.table_top + L.cube_size / 2]
        self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([pos, yaw_quat_wxyz(yaw)]))
        self.sim.forward()

    def apply_color(self, color):
        """キューブの色を変える (モデルを作り直さずに geom の rgba を書き換える)."""
        self.color = color
        for g in self.cube_geom_ids:
            if self.sim.model.geom_rgba[g][3] > 0:
                self.sim.model.geom_rgba[g][:3] = RGB[color]

    def get_state(self):
        return self.sim.get_state().flatten().copy()

    def set_state(self, state):
        self.sim.set_state_from_flattened(state)
        self.sim.forward()

    # ---------- 判定 ----------
    @property
    def cube_pos(self):
        return np.array(self.sim.data.body_xpos[self.cube_body_id])

    @property
    def eef_pos(self):
        return np.array(self.sim.data.site_xpos[self.eef_site_id])

    def gripper_opening(self):
        """指 2本の qpos 合計 (全開 ≈ 0.08)."""
        q = self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes["right"]]
        return float(np.abs(q).sum())

    def gripper_open(self):
        return self.gripper_opening() > self.layout.cube_size + 0.015

    def placed_box(self):
        p = self.cube_pos
        if p[2] > self.layout.wall_top + 0.01:
            return None
        return self.layout.box_of(p[:2])

    def _check_success(self):
        p = self.cube_pos
        return bool(self.layout.box_of(p[:2]) == self.color and p[2] < self.layout.wall_top and self.gripper_open())

    def reward(self, action=None):
        return float(self._check_success())


def make_env(cfg, cube_color=None, noise=True, render=False, camera_names=("agentview", "topdown"), camera_size=None):
    return CubeSortEnv(cfg, cube_color=cube_color, noise=noise, has_offscreen_renderer=render,
                       camera_names=camera_names, camera_size=camera_size)
