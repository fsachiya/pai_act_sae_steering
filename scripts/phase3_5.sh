#!/bin/bash
# ACT 学習完了待ち → E0 → 活性収集 → SAE 学習 (全8位置) → 特徴同定 → DiffMean → 層比較
set -e
cd "$(dirname "$0")/.."
R="uv run python -m pai_steer"
while pgrep -f "pai_steer.policy.train_act" > /dev/null; do sleep 15; done
$R.policy.eval_baseline 2>&1 | grep -v "it/s" > artifacts/logs/e0.log
$R.sae.collect_acts > artifacts/logs/collect_acts.log 2>&1
for s in decoder_out decoder_layer0 decoder_layer1 decoder_layer2 encoder_layer0 encoder_layer1 encoder_layer2 encoder_layer3; do
  $R.sae.train_sae --site $s 2>&1 | grep -v "it/s" > artifacts/logs/sae_$s.log
done
$R.analysis.find_goal_features > artifacts/logs/find_goal.log 2>&1
$R.analysis.find_xy_features --site all > artifacts/logs/find_xy.log 2>&1
$R.analysis.diffmean > artifacts/logs/diffmean.log 2>&1
$R.analysis.layer_study 2>&1 | grep -v "it/s" > artifacts/logs/layer_study.log
echo PHASE3_5_DONE
