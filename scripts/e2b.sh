#!/bin/bash
# E2 第2弾: 色の SAE (color トークン専用) と 軌道の SAE (P3, decoder) を分けて検証
set -e
cd "$(dirname "$0")/.."
S="uv run python -m pai_steer.steer.sweep"
$S --pattern p2 --color-sae 2>&1 | grep -v "it/s" > artifacts/logs/sweep_p2_colorsae.log
$S --pattern p3 2>&1 | grep -v "it/s" > artifacts/logs/sweep_p3.log
$S --pattern p3 --per-k 2>&1 | grep -v "it/s" > artifacts/logs/sweep_p3_perk.log
$S --pattern p3_dm 2>&1 | grep -v "it/s" > artifacts/logs/sweep_p3_dm.log
$S --pattern p3 --per-k --site decoder_layer2 2>&1 | grep -v "it/s" > artifacts/logs/sweep_p3_l2_perk.log
echo E2B_DONE
