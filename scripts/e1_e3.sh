#!/bin/bash
# E1 (4×4 行列, 各セル seed 0..19) → E3 (軌跡) → 図. 強度は E2 の結果から決めた値.
set -e
cd "$(dirname "$0")/.."
M="uv run python -m pai_steer.steer.run_matrix"
$M --pattern p2 --strength 2 2>&1 | grep -v "it/s" > artifacts/logs/matrix_p2.log
$M --pattern p2 --strength 2 --gate all 2>&1 | grep -v "it/s" > artifacts/logs/matrix_p2_start.log
$M --pattern p2 --color-sae --strength 3 2>&1 | grep -v "it/s" > artifacts/logs/matrix_p2_colorsae.log
$M --pattern p2_dm --strength 1.5 2>&1 | grep -v "it/s" > artifacts/logs/matrix_p2_dm.log
$M --pattern p1 --strength 0.5 2>&1 | grep -v "it/s" > artifacts/logs/matrix_p1.log
$M --pattern p1_dm --strength 1 2>&1 | grep -v "it/s" > artifacts/logs/matrix_p1_dm.log
$M --pattern p3 --per-k --strength 8 2>&1 | grep -v "it/s" > artifacts/logs/matrix_p3.log
$M --pattern p3_dm --strength 1 2>&1 | grep -v "it/s" > artifacts/logs/matrix_p3_dm.log
uv run python -m pai_steer.viz.plots --e3 --p1-alpha 0.5 --p2-beta 2 --e3-srcs red,green > artifacts/logs/plots.log 2>&1
echo E1_E3_DONE
