# LLM オンボーディングサマリー

> 本書は、本リポジトリで作業を引き継ぐ LLM エージェント（worker / auditor / team-lead など）が
> 最初に読むためのオンボーディング要約です。事実は `docs/` 配下の各設計ドキュメント・`README.md`・
> `skills/`・`pixi.toml` から確認・展開しています。ブランチは `focal-cu127`（`main` は素のインポートを温存）。
> 最終更新: 2026-06-09

---

## 1. プロジェクト概要 / 目的

| 項目 | 内容 |
|---|---|
| 名称 | focal-cu127: pixi + ROS2(Humble via RoboStack) + MuJoCo + LeRobot + SmolVLA Ubuntu sandbox |
| ドメイン | VLA（Vision-Language-Action）ロボットマニピュレーションのシミュレーション |
| 最終成果物 | SmolVLA 推論（将来は fine-tune も）を、ROS2 / MuJoCo 上のシミュレーテッドアームで、ハードウェア無しに本マシン上で動かす単一リポジトリ |
| ビジネス価値 | 実機なしで VLA（SmolVLA）を高速に試行・反復し、SO-101 へ繋げる |
| 構成方針 | MuJoCo ノード（`vla_mujoco`）と方策ノード（`vla_policy`）を ROS2 トピックで疎結合に分離。`backend:=mock` で疎通、`backend:=smolvla` で VLA 接続を切替 |

### 進捗（Status）

| マイルストーン / トラック | 状態 |
|---|---|
| M0（環境構築 + mock 閉ループ） | ✅ 完了・独立監査済み（`focal-cu127` の commit `ac23bf8`） |
| SmolVLA 推論（standalone） | ✅ PASS |
| 実レンダリング（EGL） | ✅ PASS |
| toy_arm 指令追従（モデル修正） | ✅ 完了 |
| **SmolVLA 駆動の「見えて連続的に動く」E2E** | 🔄 **進行中**（`action_scale` 調整 ＋ クリーンな単一インスタンス録画を反復中）← 現在のボトルネック |
| skills/ 作成・最終コミット + docs | ⏳ 仕上げ中 |

---

## 2. クリティカルな制約（壊してはいけないもの / Do-Not-Break）

- **`main` ブランチは upstream の素のインポートで pristine。作業のために絶対に変更・checkout しない。全作業は `focal-cu127` で行う。**
- **依存ピン（`pixi.toml` / lerobot 連鎖で確定）**:
  - Python **3.12** に固定（`lerobot[smolvla]` が Python>=3.12 を要求、RoboStack humble は cp39/cp311/cp312 のみで両立点が 3.12）。
  - setuptools **>=68,<80**（upstream の `<=58` は 3.12 で distutils 削除により破綻。`>=80` は `setup.py develop`=`--symlink-install` を削除）。
  - numpy **>=2.0,<2.3**（lerobot 0.5.x 要件）。
  - torch **>=2.7,<2.8（2.7.1+cu126）/ torchvision 0.22.1+cu126**。**cu128 nightly（torch 2.10.0+cu128）は地雷**（`torch._dynamo.utils.NP_SUPPORTED_MODULES` 削除で torchvision import が壊れる）ので使わない。
  - packaging **>=24.2,<26.0**（conda-forge 既定 26.x との PyPI ソルバ競合回避）、opencv は lerobot 0.5.1 要件で <4.14（conda 4.13 で OK）。
- **対象ホスト**: Ubuntu 20.04 Focal / NVIDIA RTX A5000 24GB（Ampere, CC 8.6）/ Driver 565 / CUDA 12.7。RoboStack により Focal でも ROS2 Humble が成立し、**システム ROS（`/opt/ros`）は不要**。
- **録画ディシプリン（最重要）**: 録画の前に必ず残存 sim/policy プロセスを全 kill し、**Publisher count == 1** を検証してから単一クリーンインスタンスで録画する（多重 publisher のトピックフリッカが偽の「動き」を作る）。
- **画像の真偽検証**: 画像が合成プレースホルダでなく実描画であることを確認（合成 = R 列方向ランプ / G 行方向ランプ / B 全画素一様。実描画はこの 3 条件をすべて破る）。
- **動きの真偽検証**: same-parity（gap-2）フレーム差分で判定（連続フレーム差分だけ見るとフリッカに騙される）。
- **現スコープ = 推論 + 可視化のみ。training / fine-tune は含めない。** `smolvla_base` は本 toy アーム上では未学習のため、その挙動はタスク的に無意味（=想定どおり）。

---

## 3. 合意済み参考ドキュメント一覧

| ファイル | 役割 |
|---|---|
| `docs/PROPOSAL.md` | 要求 / 提案書（単一リポジトリ構成の結論と採用理由） |
| `README.md` | 要件・概要・クイックスタート・主要コマンド一覧 |
| `docs/ARCHITECTURE.md` | 設計（2 ノード構成・データフロー・将来移行先） |
| `docs/TOPIC_CONTRACT.md` | トピック契約（観測 / 行動トピック・型・publisher/subscriber） |
| `docs/SMOLVLA_ADAPTER.md` | SmolVLA アダプタ設計（feature key / 状態・行動次元・YAML 駆動） |
| `docs/ROADMAP.md` | ロードマップ（M0〜M5） |
| `docs/IMPLEMENTATION_PLAN.md` | 実装計画（Phase 0〜5） |
| `docs/UBUNTU_NOTES.md` | Ubuntu 前提の注意（Humble / GPU / MuJoCo レンダリング） |
| `docs/FOCAL_CU127.md` | Focal 注記（Ubuntu 20.04 + CUDA 12.7 対応・検証マシン実測値） |
| `docs/WORKLOG.md` | 作業記録 WORKLOG（findings / struggles / tips / timeline） |
| `docs/REFERENCE_REPO_NOTES.md` | 参考リポジトリ・記事から取り込む要点（RoboStack / pixi 例 / ABEJA / mujoco_ros2_control） |
| `skills/pixi-env-doctor/SKILL.md` | ヘルパー skill: pixi + RoboStack + CUDA 環境診断 |
| `skills/smolvla-infer-smoke/SKILL.md` | ヘルパー skill: ROS なし SmolVLA 推論スモーク |
| `skills/clean-sim-run/SKILL.md` | ヘルパー skill: 単一クリーン sim+policy 起動・Publisher==1 検証 |
| `skills/record-and-verify-motion/SKILL.md` | ヘルパー skill: カメラ録画 + 実運動 / フリッカ判定 |

---

## 4. タスク境界（委譲する / しない）

### 委譲する（Delegate）

- **実装 / ビルド / 実行 / デバッグ** → worker（Sonnet）。
- **独立検証** → auditor（Opus、独立に証拠を再取得）。
- **オーケストレーション / ゲート判断** → team-lead。

### やらない（Do Not）

- `main` を変更しない。
- ユーザー承認なしに DoD（完了定義）を変更しない。
- auditor が**独立に再取得した証拠で PASS する前に**マイルストーンを "done" と宣言しない。
- fine-tune / train を行わない（現スコープ外）。
- 汚染された / 多重インスタンス環境の録画を信用しない。

---

## 5. 対話・報告ポリシー

- 簡潔・証拠ベース。**実測値**で報告する。
- 独立監査なしに成功を主張しない。
- 失敗 / 不確実性は正直に表面化する（TBD を断定しない）。

---

## 6. 試用オンボーディングタスク（2〜3 個）

1. **環境診断**: `pixi run doctor`（または `just doctor`）を実行し、`torch ... cuda_available=True, device=NVIDIA RTX A5000` と `python 3.12` を確認する。
   - 参照 skill: `skills/pixi-env-doctor/SKILL.md`。
2. **SmolVLA 推論スモーク**: `pixi run python scripts/smolvla_standalone_test.py` を実行し、`select_action` が `cuda` 上で `(1, 6)` の Tensor を返し `Standalone inference: PASS` で終わることを確認する。
   - 参照 skill: `skills/smolvla-infer-smoke/SKILL.md`。
   - 初回は HF から `lerobot/smolvla_base`（~865MB）を取得するためネットワークが要る。
3. **クリーン録画 + 運動判定**: `clean-sim-run` と `record-and-verify-motion` skill を用いて、単一インスタンスでクリーン録画し、same-parity（gap-2）フレーム差分を計算して、実運動かフリッカかを判定する。

---

## 7. 運用ルール（Ops）

- 作業中は `docs/WORKLOG.md` を随時更新する（findings / struggles / tips）。
- 不明点は **TBD** と明示する。
- マイルストーンのクローズは **auditor PASS ゲート**が必須。
- DoD の変更は**ユーザー経由のみ**。
- 記入済みの docs は `focal-cu127` でバージョン管理下に保つ。

---

## 付録（Appendix）

### 主なディレクトリ

| ディレクトリ | 役割 |
|---|---|
| `ros2_ws/` | ROS2 ワークスペース（`src/vla_mujoco`、`src/vla_policy`） |
| `docs/` | 設計・運用ドキュメント |
| `scripts/` | 診断・補助スクリプト（`doctor.py`、`smolvla_standalone_test.py`、`record_cameras.py`、`render_offscreen.py` ほか） |
| `skills/` | ヘルパー skill（上記 4 種の `SKILL.md`） |
| `temp/` | 生成物（MP4 / PNG など）。**gitignore 対象**（`/tmp` は揮発するため成果物はここに保存） |

### 代表的なコマンド

- `just <recipe>` / `pixi run <task>`
- 例: `pixi run doctor`、`pixi run build`、`pixi run sim`、`pixi run policy-mock`、`pixi run policy-smolvla`、`pixi run all-mock`、`pixi run list-topics`、`pixi run echo-joints`。
- ROS2 CLI / daemon は pixi 環境内に存在するため `pixi run -- ros2 ...` で呼ぶ。
- PATH は非永続。各シェル冒頭で `export PATH="$HOME/.pixi/bin:$PATH"`。ネットワーク / GPU / pixi コマンドは Bash の `dangerouslyDisableSandbox=true` が必要。

### 主要依存

- ROS2 Humble（RoboStack 経由）、mujoco 3.9（pixi では `>=3.3.0,<4`）、lerobot 0.5.1、torch 2.7.1+cu126。

### オーナー / 連絡先

- プロジェクトオーナー（ユーザー: yoshikawa@inaho.co）。
