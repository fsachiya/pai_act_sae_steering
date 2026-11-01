"""共通ユーティリティ: config 読込, device 選択, seed, パス."""
import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
DATA_HDF5 = DATA / "cube_sort.hdf5"
SAMPLES = ART / "samples"
STATS = ART / "stats.npz"
CKPT = ART / "ckpt"
LOGS = ART / "logs"
ACTS = ART / "acts"
SAE_DIR = ART / "sae"
FEATURES = ART / "features"
RESULTS = ART / "results"
FIGS = ART / "figs"
ROLLOUTS = ART / "rollouts"
VIDEOS = ART / "videos"
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

COLORS = ["red", "green", "blue", "yellow"]
# SAE を学習する位置. encoder_layer3 = encoder_out, decoder_out = decoder 最終層
ENC_SITES = [f"encoder_layer{i}" for i in range(4)]
DEC_SITES = [f"decoder_layer{i}" for i in range(3)] + ["decoder_out"]
SITES = ENC_SITES + DEC_SITES


def site_module(site):
    """SAE の位置名 → ACT のフック用モジュール名."""
    return site if site == "decoder_out" else f"{site}_out"


def is_decoder(site):
    return site.startswith("decoder")
COLOR_JP = {"red": "赤", "green": "緑", "blue": "青", "yellow": "黄"}


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def base_parser(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--quick", action="store_true", help="動作確認用の短縮版")
    return p


def get_device():
    import torch

    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def setup_mpl():
    """matplotlib を Agg + 日本語フォント(ヒラギノ角ゴ)で設定する."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    for p in ["/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    # レポートに幅いっぱいで載せても読める文字の大きさ
    matplotlib.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12, "xtick.labelsize": 11,
                                "ytick.labelsize": 11, "legend.fontsize": 10.5, "figure.titlesize": 14})
