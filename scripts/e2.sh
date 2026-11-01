#!/bin/bash
# E2 強度スイープ (主な介入位置: P2 = encoder_layer0/color, P1 = decoder_out)
set -e
cd "$(dirname "$0")/.."
S="uv run python -m pai_steer.steer.sweep"
$S --pattern p2 2>&1 | grep -v "it/s" > artifacts/logs/sweep_p2.log
$S --pattern p2_dm 2>&1 | grep -v "it/s" > artifacts/logs/sweep_p2_dm.log
$S --pattern p1 2>&1 | grep -v "it/s" > artifacts/logs/sweep_p1.log
$S --pattern p1_dm 2>&1 | grep -v "it/s" > artifacts/logs/sweep_p1_dm.log
$S --pattern p2 --site encoder_layer3 --all-tokens --strengths 0,0.5,1,2,3 2>&1 | grep -v "it/s" > artifacts/logs/sweep_p2_l3all.log
$S --pattern p2_dm --site encoder_layer3 --all-tokens --strengths 0,0.5,1,1.5,2 2>&1 | grep -v "it/s" > artifacts/logs/sweep_p2_dm_l3all.log
echo E2_DONE
