#!/bin/bash
# 全パイプライン. P1/P2 の強度は E2 の結果を見て与える (P1_ALPHA, P2_BETA)
set -e
cd "$(dirname "$0")/.."
R="uv run python -m pai_steer"
$R.data.generate --samples                 # §4.6 サンプル (確認ゲート)
$R.data.generate                           # 400 本
$R.policy.train_act
$R.policy.eval_baseline                    # E0
$R.sae.collect_acts
for s in decoder_out encoder_layer0 encoder_layer1 encoder_layer2 encoder_layer3; do $R.sae.train_sae --site $s; done
for s in decoder_out encoder_layer3; do $R.policy.eval_baseline --sae-recon $s; done
$R.analysis.find_xy_features
$R.analysis.find_goal_features
$R.analysis.probe_baseline
$R.steer.sweep --pattern p1               # E2
$R.steer.sweep --pattern p2
$R.steer.run_matrix --pattern p1 --strength ${P1_ALPHA:-4}     # E1
$R.steer.run_matrix --pattern p2 --strength ${P2_BETA:-1}
$R.steer.run_matrix --pattern p2 --strength ${P2_BETA:-1} --gate all
$R.viz.plots --e3 --p1-alpha ${P1_ALPHA:-4} --p2-beta ${P2_BETA:-1}
$R.viz.make_video --record --p2-beta ${P2_BETA:-1}
