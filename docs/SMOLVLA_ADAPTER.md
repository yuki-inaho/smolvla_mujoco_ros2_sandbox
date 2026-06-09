# SmolVLAアダプタ設計

## 基本方針

SmolVLAは、複数カメラ画像、ロボット状態、自然言語指示を入力し、連続行動を出力するVLAです。ただし、実際に期待されるfeature key、画像サイズ、状態次元、行動次元は、事前学習またはfine-tuneに使ったデータセットに依存します。

そのため、本リポジトリの `SmolVLABackend` は以下をYAMLから指定します。

```yaml
camera_names: [top, side, wrist]
camera_topics:
  - /top_camera/image_raw
  - /side_camera/image_raw
  - /wrist_camera/image_raw
camera_keys:
  - observation.images.top
  - observation.images.side
  - observation.images.wrist
state_key: observation.state
task_key: task
action_mode: delta_joint
```

## 実装上の注意

- `lerobot/smolvla_base` はベースモデルなので、対象環境に最適化するには自前データでfine-tuneする前提です。
- 初期疎通では `backend:=mock` を使い、ROS 2トピック、関節順序、画像publishを先に固めます。
- `backend:=smolvla` に切り替える前に、対象チェックポイントのdataset featuresを確認し、`camera_keys` と `state_key` を合わせます。
- SmolVLAの出力がaction chunkの場合、このスケルトンでは先頭actionだけを取り出します。必要に応じてchunk bufferを実装してください。
- 関節角直接制御ではなくエンドエフェクタdeltaを出す方策の場合、IKまたはコントローラ変換層を `vla_policy` と `vla_mujoco` の間に追加します。

## 推奨する開発順序

1. `pixi run sim` と `pixi run policy-mock` で動作確認する。
2. 実SO-101または対象MuJoCoモデルの関節名に `joint_names` を合わせる。
3. 画像トピックとカメラ視点を固定する。
4. rosbagまたはLeRobot形式でデモを収集する。
5. `lerobot-train` でSmolVLAをfine-tuneする。
6. `pixi run policy-smolvla` でfine-tuned checkpointを指定して閉ループ評価する。
