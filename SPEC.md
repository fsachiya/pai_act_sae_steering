# SPEC: 色キューブ仕分けタスクにおける ACT 方策への SAE 介入

Physical AI 応用編1 最終課題（提出締切 **2026-11-02(月) 10:00**）用プロジェクトの実装仕様書。
Claude Code はこのファイルを唯一の正として実装すること。不明点は §15「未確定事項」に従って判断し、判断内容を `NOTES.md` に記録する。

> **⚠ 確認ゲート（必ず守ること）**
> Phase 1 で **4パターンのサンプルデータ**（§4.6）を作ったら、**そこで作業を止めてユーザーの確認を待つ**。
> ユーザーが「OK」と言うまで、データの大量生成（Phase 2）以降に進まない。

---

## 0. 概要

### 0.1 タスク
- 机の上に**色の違う4つの箱**（赤・緑・青・黄）が固定位置に置かれている。
- 4色のうち**1色のキューブが1つだけ**、決まった位置に置かれている。
- ロボット（Panda）はキューブを掴み、**キューブと同じ色の箱**に入れる（赤→赤の箱、…）。

### 0.2 目的
1. スクリプトで生成したデモで ACT を模倣学習させる。
2. 学習済みの重みは固定したまま、推論時に **SAE 特徴への介入だけ**で「赤いキューブを緑・青・黄の箱に入れる」ことを目指す。

### 0.3 検証する2つの介入パターン
| ID | 名称 | 介入する層 | 狙い |
|---|---|---|---|
| P1 | xy 移動特徴の刺激 | ACT decoder 出力（action head の直前） | 手先の x/y 方向の移動を直接ずらし、置く位置を連続的に変える |
| P2 | 目標（色→箱）特徴の刺激 | ACT encoder 出力（color / cube トークン） | 「どの箱へ運ぶか」の表現を差し替え、軌道全体を別の箱行きに切り替える |

主な問い：**運動レベル（P1）と意図レベル（P2）の介入で、到達できる範囲と副作用（把持の崩れなど）はどう違うか。**

---

## 1. 実行環境

| 項目 | 値 |
|---|---|
| マシン | MacBook Air (Apple M5, arm64, macOS)。**CUDA なし**。学習は MPS、使えなければ CPU |
| パッケージ管理 | `uv`（プロジェクト内 `.venv`） |
| シミュレータ | **robosuite 1.5.x**（DeepMind の `mujoco` Python バインディング版）＋ Panda |
| Python | 3.12 を第一候補とする。robosuite 1.5 が入らなければ 3.11 → 3.10 の順に下げる |
| 配置 | `~/work/pai_act_sae_steering/` |
| 長時間ジョブ | `nohup` と `caffeinate -i` で実行し、ログは `artifacts/logs/` に出す。MacBook Air はファンレスなので熱による速度低下を前提にする |

### 1.1 依存関係
`robosuite==1.5.*`, `mujoco`, `torch`（MPS 対応版）, `numpy`, `h5py`, `matplotlib`, `pyyaml`, `tqdm`, `imageio`, `imageio-ffmpeg`, `pandas`

- LeRobot、robomimic、MimicGen は使わない。
- ACT は自前で最小実装する。フックを掛けるモジュール名を自分で管理するため。
- SAE は `~/work/sae_so101/src/sae_so101/sae/` の TopK 実装を**コピーして移植**する。パス依存にはしない。

### 1.2 レンダリング
- offscreen 描画（`has_offscreen_renderer=True`）が macOS で動くかを Phase 0 で確認する。`MUJOCO_GL` は未設定、`cgl`、`glfw` の順に試す。
- 動かなければ、生成とロールアウトのときは状態（qpos/qvel）だけを保存し、別プロセスで `mujoco.Renderer` を使って再生・描画する。

---

## 2. ディレクトリ構成

```
pai_act_sae_steering/
  SPEC.md  NOTES.md  README.md  pyproject.toml
  configs/default.yaml        # 全パラメータ（環境配置・生成・学習・SAE・介入）
  src/pai_steer/
    common.py                 # config 読込, device 選択(MPS→CPU), seed, パス
    env/
      cube_sort_env.py        # §3 のカスタム環境（robosuite ManipulationEnv のサブクラス）
      layout.py               # 箱・キューブの配置, 色の定義, 箱判定 box_of(xy)
    data/
      scripted_expert.py      # §4 のスクリプト方策（ウェイポイント＋P制御）
      generate.py             # デモ生成（--samples で §4.6 のサンプル4本）
      dataset.py              # 学習用 Dataset（チャンク切り出し, 正規化）
    policy/
      act.py  train_act.py  rollout.py
    sae/
      topk_sae.py  collect_acts.py  train_sae.py
    analysis/
      find_xy_features.py  find_goal_features.py  probe_baseline.py
    steer/
      hooks.py  run_matrix.py  sweep.py
    viz/
      render.py               # 状態列 → 画像列（offscreen 不可時の再生描画も担当）
      plots.py  make_video.py
  scripts/run_all.sh
  data/        # .gitignore
  artifacts/   # .gitignore（samples, ckpt, 活性, 図, 動画, ログ）
```

---

## 3. 環境（`env/cube_sort_env.py`, `env/layout.py`）

robosuite 1.5 の `ManipulationEnv` を継承した `CubeSortEnv` を作る（`Lift` 環境の実装を雛形にするとよい）。

### 3.1 シーン
- アリーナ: `TableArena`（天板サイズ 0.8×0.8m、`table_offset=(0, 0, 0.8)`）
- ロボット: Panda 1台、グリッパーは標準の `PandaGripper`
- controller: `BASIC` の composite controller（アームは OSC_POSE の差分指令）。**行動は7次元**（dpos 3, drot 3, gripper 1）、`control_freq=20`
- **配置はすべて固定**（毎エピソード同じ）。ばらつきはノイズとしてだけ入れる（§3.4）

### 3.2 箱（4個、固定物）
- 上が開いた箱を、底板1枚と壁4枚の `BoxObject` で組む。robosuite に箱型の composite object があればそれを使ってもよい。
- 内寸 0.10×0.10m、壁の高さ 0.05m、壁の厚さ 0.005m。
- 色: 赤 (0.85,0.15,0.15)、緑 (0.15,0.70,0.20)、青 (0.15,0.30,0.85)、黄 (0.90,0.80,0.10)。
- 自由関節は持たせず、机に固定する。
- **2×2 の格子状に配置する**。座標は机の中心を原点とし、x はロボットから遠ざかる向き、y は左向き。

| 箱 | 中心 (x, y) |
|---|---|
| 赤 | (0.10, +0.12) |
| 緑 | (0.10, −0.12) |
| 青 | (0.24, +0.12) |
| 黄 | (0.24, −0.12) |

- この配置にすると、赤→緑と赤→青は「隣」（y だけ、または x だけのずれ）、赤→黄は「対角」（x と y の両方）になる。P1 と P2 の比較に使う。
- 座標は config で変更できるようにする。**Panda が無理なく届くことを Phase 0 で確認し、届かなければ x を手前にずらす**（ずらした値を `NOTES.md` に記録）。

### 3.3 キューブ
- 一辺 0.04m の `BoxObject`。自由関節あり。密度は既定、摩擦は把持しやすいよう大きめ（例: 1.5）。
- 色は §3.2 の4色のどれか。環境の引数 `cube_color ∈ {red, green, blue, yellow}` で指定する。`None` ならリセットごとに一様ランダム。
- 初期位置の公称値: (x, y) = (−0.10, 0.00)、机の上に置く（中心の z は天板＋0.02）、yaw = 0。

### 3.4 初期状態のノイズ（config で設定。サンプル生成時は §4.6 に従う）
| 対象 | ノイズ |
|---|---|
| キューブの xy | 一様 ±0.01m |
| キューブの yaw | 一様 ±10° |
| ロボットの初期関節角 | robosuite 既定の初期姿勢＋一様 ±0.02rad |

### 3.5 観測（低次元、`_setup_observables` で定義）
| キー | 次元 | 内容 |
|---|---|---|
| `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos` | 3, 4, 2 | 標準 |
| `cube_pos`, `cube_quat` | 3, 4 | キューブの位置と姿勢 |
| `cube_to_eef_pos` | 3 | 手先から見たキューブの相対位置 |
| `cube_color_onehot` | 4 | キューブの色（赤・緑・青・黄の順） |

- 画像は `agentview`, `birdview`（真上から）, `robot0_eye_in_hand` を 128×128 で取得できるようにする。学習には使わない。動画と確認用。
- 真上から見下ろすカメラ `topdown` を追加してよい。箱4つとキューブが全部入る画角にする。

### 3.6 成功判定と補助関数
- `box_of(xy) -> {red, green, blue, yellow, None}`: xy がどの箱の内寸の中にあるか。
- `success`: 次の3つをすべて満たす。
  1. キューブの xy が、自分と同じ色の箱の内寸の中にある
  2. キューブの z が箱の壁の上端より低い
  3. グリッパーが開いている
- `placed_box`: エピソード終了時にキューブが入っている箱（介入実験では成否ではなくこれを記録する）。

---

## 4. データ生成（`data/scripted_expert.py`, `data/generate.py`）

### 4.1 スクリプト方策
ウェイポイントの列を順に追う。各ウェイポイントで手先位置の P 制御を行う。

$$
a_{\mathrm{pos}} = \mathrm{clip}\left(k_p \,(p_{\mathrm{wp}} - p_{\mathrm{eef}}),\ -1,\ 1\right)
$$

- 姿勢は真下向きを保つよう、回転誤差に比例した `drot` を出す（キューブの yaw に合わせて手首を回す）。
- 手先とウェイポイントの距離が 0.01m 未満になったら、次のウェイポイントへ進む。グリッパーの開閉は決めたステップ数だけ待つ。
- **フェーズ**（ウェイポイントの番号）を真値として毎ステップ記録する。SAE の解析にそのまま使う。

| # | フェーズ名 | 目標 |
|---|---|---|
| 0 | `approach` | キューブの真上 +0.10m、グリッパーは開 |
| 1 | `descend` | キューブの中心の高さ（指先がキューブの中段に来る高さ） |
| 2 | `grasp` | その場でグリッパーを閉じ、15 ステップ待つ |
| 3 | `lift` | 机から +0.15m |
| 4 | `transport` | 目標の箱の中心の真上、+0.15m |
| 5 | `lower` | 箱の壁の上端 +0.03m |
| 6 | `release` | グリッパーを開き、10 ステップ待つ |
| 7 | `retreat` | 真上へ +0.10m |

### 4.2 デモの多様性
- 各ウェイポイントの xy に一様 ±0.01m のノイズを加える。
- $k_p$ をエピソードごとに [0.8, 1.2] 倍の範囲で変える。
- 行動に小さなガウスノイズ（σ=0.05、グリッパー以外）を加える。
- いずれも config で 0 にできるようにする。

### 4.3 生成量と採用条件
- 色ごとに 100 本、合計 400 本を目標にする（config で変更可）。
- `success` が真で、長さが最大ステップ数（400）以内のものだけを採用する。
- 色ごとの採用率を記録する。採用率 < 80% ならスクリプト方策を直す。

### 4.4 保存形式（`data/cube_sort.hdf5`、robomimic に似た構成）
```
data/
  attrs: env_config(json), layout(json), generator_config(json), created_at
  demo_0/
    actions        (T, 7)
    obs/<key>      (T, d)   §3.5 の低次元観測すべて
    states         (T, n)   MuJoCo の状態（動画の再描画用）
    phase          (T,)     int, §4.1 のフェーズ番号
    attrs: cube_color, success, placed_box, length, seed
```
画像はデモ本体には保存しない。必要なときは `states` から再描画する。

### 4.5 生成速度
描画なしで回す。1本あたりの所要時間をログに出す。

### 4.6 サンプル4パターン（確認ゲート）
`uv run python -m pai_steer.data.generate --samples` で、**4色それぞれ1本ずつ**生成する。
- 初期状態のノイズとデモのノイズはどちらも 0 にする（公称の配置そのものを見せるため）。
- 出力先は `artifacts/samples/`。

| ファイル | 内容 |
|---|---|
| `init_grid.png` | 4パターンの初期状態を 2×2 に並べた画像（`agentview` と `topdown`）。色名のラベルつき |
| `sample_{color}.mp4` | 各パターンの動画。`agentview` と `topdown` を左右に並べ、左上に「色」と「現在のフェーズ名」、右上にステップ数を重ねる。20fps、等倍 |
| `trajectories_topdown.png` | 4パターンの手先の xy 軌跡を1枚に重ねた図。箱の枠を実際の色で描き、キューブの初期位置も示す |
| `actions_{color}.png` | 行動7次元とフェーズの時系列 |
| `summary.json` | 各パターンの success、placed_box、長さ、キューブの最終位置 |
| `samples.hdf5` | §4.4 と同じ形式のサンプルデータ |

**ここで作業を止め、ユーザーに次の確認項目を提示して返事を待つ。**
- [ ] 机の上のキューブは1つだけで、箱は4つとも見えているか
- [ ] 4パターンとも、キューブの初期位置が同じか
- [ ] キューブの色と、入れた先の箱の色が一致しているか
- [ ] 把持が安定していて、運搬中にキューブが落ちたりずれたりしないか
- [ ] 箱の配置（2×2）と、隣・対角の関係が意図どおりか
- [ ] 動き（速さ、持ち上げる高さ、置き方）が自然か

ユーザーから修正の指示があれば、修正してサンプルを作り直し、再び確認を待つ。

---

## 5. 観測・行動の表現と ACT（`policy/act.py`）

### 5.1 入力トークン
| トークン | 中身 | 次元 |
|---|---|---|
| robot | `eef_pos` + `eef_quat` + `gripper_qpos` | 9 |
| cube | `cube_pos` + `cube_quat` + `cube_to_eef_pos` | 10 |
| color | `cube_color_onehot`。**「a」を表す入力で、P2 で操作する「a→A」の表現はここから作られる** | 4 |

- 各トークンは、それぞれ専用の Linear で d_model 次元に写す。
- 正規化: 観測と行動は学習データで平均0・分散1にする（one-hot は除く）。統計は `artifacts/stats.npz` に保存する。

### 5.2 ACT の構成（Zhao et al. 2023 に準拠した最小実装）
| 項目 | 値（初期値、config で変更可） |
|---|---|
| チャンク長 K | 20 |
| 1回の推論で実行するステップ数 | 5（temporal ensembling は**既定 OFF**。介入効果がなまるため） |
| d_model / heads / FFN | 256 / 8 / 1024 |
| encoder / decoder 層数 | 4 / 4 |
| CVAE 潜在次元 | 32（推論時は z=0） |
| KL 重み | 10 |
| 損失 | L1（行動）＋ KL |
| 最適化 | AdamW, lr 1e-4, wd 1e-4, batch 64, 30k step（MPS で1時間以内を目標） |

- encoder の入力: `[latent, robot, cube, color]`（＋学習可能な位置埋め込み）
- decoder: K 個の学習可能クエリ → `action_head`（Linear d_model→7）
- **フック用のモジュール名を固定する**（それぞれ Identity モジュールを挟む）:
  - `encoder_layer{i}_out`（i=0..3）と `encoder_out`: 形は `[B, 4, d]`
  - `decoder_out`: action head の直前、形は `[B, K, d]`

### 5.3 合格基準（Phase 3 の DoD）
- 色ごとに 50 エピソード（初期状態のノイズあり、seed 0–49）を評価し、**4色すべてで成功率 ≥ 90%**。
- 届かない場合の調整順: step を 2 倍 → デモを色ごとに 200 本へ増やす → K=30。

---

## 6. ロールアウト（`policy/rollout.py`）
- 引数: 方策、`cube_color`、seed、`steer_fn`（フック）、`gate_fn(info) -> bool`（今介入するか）。
- gate の判定には、シミュレータの真の状態から作るフェーズ推定を使ってよい（例: グリッパーが閉じていて、キューブが机から 2cm 以上浮いている＝運搬中）。実験条件の制御として扱い、レポートに明記する。
- 毎ステップの記録: 手先の xyz、キューブの xyz、推定フェーズ、介入の有無、行動、`states`。
- 終了時の記録: `placed_box`、`success`、把持に成功したか。

---

## 7. SAE

### 7.1 活性の収集（`collect_acts.py`）
- 学習データの全時刻について、方策の forward（z=0）を回して収集する。
  - `decoder_out`: `[T, K, d]` を `[T·K, d]` に平らにする（チャンク内の位置 k も保存）
  - `encoder_out` と各 `encoder_layer{i}_out`: `[T·3, d]`（robot / cube / color の3トークン。トークン種別も保存）
- ラベル（cube_color、phase、行動、手先の将来の変位 dx, dy）を対応づけて保存する。

### 7.2 SAE の構成（`topk_sae.py`）
sae_so101 の TopK 実装を移植する（入力の正規化、デコーダの単位ノルム化、任意で OrtSAE の直交化損失）。

$$
f = \mathrm{TopK}\left(W_{\mathrm{enc}}(\tilde{h} - b_{\mathrm{dec}}) + b_{\mathrm{enc}}\right), \qquad \hat{\tilde{h}} = W_{\mathrm{dec}} f + b_{\mathrm{dec}}
$$

（$\tilde{h}$ は正規化済みの活性）

| 項目 | 値 |
|---|---|
| 辞書サイズ | 8×d = 2048 |
| k | 32（decoder_out）, 16（encoder 側） |
| 学習 | Adam lr 1e-3, batch 4096, 20 epoch 程度 |

### 7.3 合格基準
- 説明分散 R² ≥ 0.9、死んだ特徴の割合 < 10%
- SAE の再構成で置き換えたときの成功率の低下が 10pt 以内

---

## 8. 特徴の同定と介入

### 8.1 誤差を保存した介入（`steer/hooks.py`）
フック位置の活性 $h$ を次のように置き換える。

$$
h' = h + W_{\mathrm{dec}}(f' - f)
$$

（正規化空間で計算して元のスケールに戻す。SAE が再構成できない残差は壊さない）

介入モード:
- `add`: 指定した特徴に α を足す
- `clamp`: 指定した特徴を固定値にする
- `swap`: 元の目標の特徴群を 0 にし、行き先の目標の特徴群を「その目標での平均活性 × β」にする

### 8.2 P1: xy 移動特徴（`find_xy_features.py`、フック `decoder_out`）
1. 各特徴のデコーダ方向 $d_i$ が行動に与える寄与を、action head を通して求める。

$$
\Delta a_i = W_{\mathrm{head}} \, d_i
$$

2. $\Delta a_i$ の x・y 成分の大きさと、運搬中（phase 4–5）の活性と dx, dy との相関から、**+x, −x, +y, −y の各方向で上位3特徴**を選ぶ。
3. 介入: 元の箱の中心から行き先の箱の中心までのずれ $(\Delta x, \Delta y)$ の符号に応じて、x と y の特徴に $\alpha_x, \alpha_y$ を足す。
4. gate: **把持後（lift 以降）だけ**介入する。
5. 出力: `artifacts/features/xy_features.json` と寄与ベクトルの図。

### 8.3 P2: 目標特徴（`find_goal_features.py`、フック `encoder_out` または `encoder_layer{i}_out`）
1. lift〜lower の区間の活性を、cube_color ごとに平均する。
2. 選択性スコアで、各色 c に固有の特徴集合 $S_c$（上位5特徴）を選ぶ。

$$
s_{i,c} = \mathbb{E}[f_i \mid c] - \max_{c' \neq c} \mathbb{E}[f_i \mid c']
$$

3. 層とトークン（color / cube / robot）のうち、選択性が最も高いところを介入位置にする。
4. 介入（`swap`）: 色 a のキューブの運搬中に、$S_a$ を 0 にし、$S_b$ を「色 b の平均活性 × β」にする。
5. 比較条件: 介入を「把持後だけ」に入れる場合と「エピソードの最初から」入れる場合を比べる。最初から入れると、掴みに行く動作は同じで（キューブの位置は1つなので）、行き先だけが変わるはず。もし把持まで崩れれば、それ自体を考察する。
6. **限界**: 色（one-hot）と箱は常にセットで現れるので、$S_c$ が「色の識別」なのか「箱 c 行き」なのかはデータからは区別できない。レポートの考察に書く。
7. 出力: `artifacts/features/goal_features.json` と、選択性のヒートマップ（層×トークン）。

### 8.4 任意の比較（`probe_baseline.py`）
同じフック位置で、平均差ベクトル $\mu_b - \mu_a$ を足す操作と、P2 の成功率を比べる。

---

## 9. 実験と評価

| 実験 | 内容 | 出力 |
|---|---|---|
| E0 ベースライン | 色ごとに 50 エピソード。SAE で再構成した活性に置き換えた条件も | 成功率の表 |
| E1 4×4 行列 | 元の色 a × 行き先の箱 b を、P1 と P2 で実施。各セル 20 エピソード（seed 共通） | `matrix_{p1,p2}.csv`、到達率のヒートマップ（対角は介入なし） |
| E2 強度スイープ | P1: α ∈ {0, 0.5, 1, 2, 4, 8}（活性の典型値の倍数）、P2: β ∈ {0, 0.25, 0.5, 1, 2} | 到達率・置いた位置のずれ・把持失敗率を、隣の箱と対角の箱に分けて描く |
| E3 定性分析 | 同じ初期状態で、介入なしと行き先3通りの手先 xy 軌跡を、真上から見た図に重ねる | 軌跡図、動画用のロールアウト |

- 到達率には 95% 信頼区間（Wilson）をつける。
- 介入なしと介入ありは同じ seed（同じ初期状態）で比べる。

---

## 10. 動画（`viz/make_video.py`）
- 保存した `states` を再描画し、mp4（H.264、30fps）にする。`agentview` と `topdown` を左右に並べる。
- 左上に条件（例: 「P2: 赤 → 青の箱」「β = 1」）、右下に倍速表記を重ねる。
- 構成案（2分30秒以内、2倍速まで）:
  1. タイトルと課題の説明
  2. ベースライン: 4色をそれぞれの箱へ（4画面を並べる）
  3. P2: 同じ初期状態で、介入なしと介入ありを左右に並べる（2例）
  4. P1: α を上げていくと置き場所がずれていく様子
  5. 4×4 行列と強度スイープの図
  6. まとめ
- 動画・コード・レポートに氏名、所属、メールアドレスを入れない。

---

## 11. 提出物

| 提出物 | 仕様 |
|---|---|
| 動画 | MP4, 1〜3分。YouTube のタイトルは `PAI最終課題_<omnicampusアカウント名>`。限定公開以上（全体公開だと加点） |
| レポート | `説明資料_PAI最終課題_<アカウント名>.pdf`。概要、目的、手法、結果、考察、工夫点、失敗と反省 |
| コード | GitHub の public リポジトリ。README に再現手順と引用（robosuite / ACT） |

- git のコミット作者名は個人名にしない。リポジトリ単位で `git config user.name <アカウント名>` と noreply のメールアドレスを設定する。
- `data/`、`artifacts/`、`*.hdf5`、`*.pt` は `.gitignore` に入れる。図と動画のサンプルだけ `docs/` にコピーしてコミットしてよい。

### 11.1 レポートの構成（`docs/report.md` を書き、PDF に変換する）
1. 課題の概要と目的
2. 環境とデータ（自作の色キューブ仕分け環境、スクリプトによるデモ生成）
3. 手法: ACT、SAE、2つの介入パターン、誤差を保存した介入
4. 結果: E0〜E3
5. 考察: P1 と P2 の違い、色の識別と目標が切り分けられない問題、gate に真の状態を使った限界
6. 工夫した点、うまくいかなかった点と反省

---

## 12. 拡張（MVP のあとで時間が余れば）
- 画像入力の ACT（色を one-hot ではなく画像から読む）。このとき P2 は「視覚的な色の特徴」への介入になり、より強い主張になる。
- 箱の配置をエピソードごとに入れ替え、色と位置の対応を観測から読ませる版（色の識別と目標の切り分けができるようになる）。
- Transcoder による回路解析。

---

## 13. フェーズとタイムライン（日時は JST）

| Phase | 内容 | DoD | 目安 |
|---|---|---|---|
| 0 | uv 環境、robosuite 1.5 の動作確認、offscreen 描画の可否、Panda の可動範囲の確認 | `NOTES.md` に結果が記録されている | 10/31 14:30 |
| 1 | `cube_sort_env`、`scripted_expert`、`generate --samples` | §4.6 の成果物がそろう → **確認ゲートで停止** | 10/31 17:00 |
| 2 | （ユーザーの OK のあと）400 本を生成 | 4色とも採用率 ≥ 80% | 10/31 18:30 |
| 3 | `dataset`、`act`、学習（夜間）、E0 | §5.3 の基準を満たす | 11/01 07:00 |
| 4 | 活性の収集、SAE の学習 | §7.3 の基準を満たす | 11/01 10:00 |
| 5 | P2 の特徴同定と介入。**1例（例: 赤→青の箱）の成功を最優先** | 介入あり/なしの動画が1本できる | 11/01 13:00 |
| 6 | P1、E1〜E3 | csv と図がそろう | 11/01 19:00 |
| 7 | 動画、レポート PDF、README、GitHub 公開 | 提出物3点がそろう | 11/01 24:00 |
| 8 | YouTube にアップロードし、omnicampus で提出 | 提出完了 | **11/02 09:00 まで** |

**打ち切りの判断**
- 10/31 15:30 の時点で robosuite 1.5 が Mac で動かない: 素の `mujoco` ＋ mujoco_menagerie の Panda で同じ環境を作る（OSC は自前実装、把持は接触ベースを維持）。
- 11/01 13:00 の時点で P2 が1例も成功しない: E1 を縮小し、特徴の分析・失敗の考察・P1 の結果で提出物をまとめる。

---

## 14. コーディング規約
- すべてのスクリプトは `uv run python -m pai_steer.<module>` で起動でき、`--config` と `--quick`（動作確認用の短縮版）を受け付ける。
- 乱数 seed は config で一括管理する。成果物のパスは `common.py` にまとめる。
- 各 Phase の終わりに、`NOTES.md` に「何をしたか、数値、次にやること」を3行で追記する。

## 15. 未確定事項（実装中に決め、`NOTES.md` に記録）
- robosuite 1.5 の対応 Python、controller 設定の書き方、offscreen 描画の可否（Phase 0）
- 箱とキューブの座標（Panda の可動範囲に合わせて調整、Phase 0–1）
- 把持の高さ、P ゲイン、待ちステップ数（Phase 1 で、サンプルを見ながら調整）
- 学習 step 数、K（Phase 3 の結果を見て）
- 特徴の選択数、α と β の範囲（Phase 5–6 の結果を見て）
