# pixi + ROS 2 + MuJoCo + LeRobot + SmolVLA Ubuntu Sandbox

Ubuntu 上で、`pixi` を依存関係の入口にし、ROS 2 Humble、MuJoCo、LeRobot、SmolVLA を単一リポジトリで扱うための設計・実装スケルトンです。

このリポジトリは、以下の2段階を明確に分けます。

1. **ラピッド検証**: MuJoCoノードとモック方策を接続し、ROS 2トピック、画像、関節状態、JointTrajectoryの疎通を確認する。
2. **SmolVLA接続**: LeRobot/SmolVLAアダプタを有効化し、観測キー、状態次元、行動次元、タスク文を対象チェックポイントに合わせる。

## 前提環境

- Ubuntu 22.04 LTS または Ubuntu 24.04 LTS
- Ubuntu 20.04 LTS (Focal) も `focal-cu127` ブランチで対応（ROS 2 は RoboStack 経由で pixi 環境内に閉じ込めるため、system ROS には非依存）。CUDA 12.7 環境向けの詳細は [`docs/FOCAL_CU127.md`](docs/FOCAL_CU127.md) を参照。
- `linux-64` 環境
- pixi
- NVIDIA GPU は推奨ですが、モック方策とMuJoCoのヘッドレス疎通はCPUでも確認可能です。

## クイックスタート

```bash
# 1. pixi のインストール済み環境で実行
pixi install

# 2. 環境診断
pixi run doctor

# 3. ROS 2 ワークスペースのビルド
pixi run build

# 4. ターミナルA: MuJoCoシミュレーター起動
pixi run sim

# 5. ターミナルB: まずモック方策で制御トピックを流す
pixi run policy-mock

# 6. トピック確認
pixi run list-topics
pixi run echo-joints
```

SmolVLA を使う場合は、モデルの初回ダウンロードとGPUメモリを考慮してください。

```bash
pixi run policy-smolvla
```

## 主要コマンド

| コマンド | 内容 |
|---|---|
| `pixi run doctor` | Python、ROS 2、MuJoCo、LeRobot、Torch/CUDAの確認 |
| `pixi run build` | `ros2_ws` を `colcon build --symlink-install` でビルド |
| `pixi run sim` | MuJoCoシミュレーターROSノードを起動 |
| `pixi run policy-mock` | SmolVLAを使わず、モック方策でJointTrajectoryをpublish |
| `pixi run policy-smolvla` | LeRobot/SmolVLAバックエンドを使う方策ノードを起動 |
| `pixi run all-mock` | sim + mock policy を同一端末から起動 |
| `pixi run record-bag` | 主要トピックをrosbag2で記録 |
| `pixi run train-smolvla` | 環境変数で指定したLeRobotデータセットを使いSmolVLAをfine-tune |

## 推奨アーキテクチャ

```text
+-------------------------+        ROS 2 topics        +--------------------------+
| vla_mujoco              |  /joint_states             | vla_policy               |
| - MuJoCo step loop      |  /top_camera/image_raw     | - ObservationBuffer      |
| - camera publisher      |  /side_camera/image_raw    | - MockPolicyBackend      |
| - trajectory subscriber |  /wrist_camera/image_raw   | - SmolVLABackend         |
+------------+------------+                            +------------+-------------+
             ^                                                      |
             | /arm_controller/joint_trajectory                     |
             | /gripper_controller/joint_trajectory                 |
             +------------------------------------------------------+
```

## 重要な設計判断

### 1. Ubuntuでは `mjpython` 起動を必須にしない

Mac向けの参考実装ではMuJoCo GUI起動の都合で `mjpython` ラッパーが必要になり得ますが、Ubuntu前提ではまず通常のPythonプロセスとしてMuJoCoをヘッドレス起動する設計にします。GUI/レンダリングが必要になった段階で EGL/OSMesa/Xvfb を切り替えます。

### 2. 最初は `mujoco_ros2_control` に直行しない

`mujoco_ros2_control` は将来の本命候補ですが、ラピッド検証ではROS 2標準メッセージだけで閉じる独自ラッパーの方が速く試せます。本リポジトリでは `/joint_states` と `/trajectory_msgs/JointTrajectory` を契約として固定し、後から ros2_control system interface へ差し替えられるようにします。

### 3. SmolVLAアダプタは設定駆動にする

SmolVLAは、学習済みチェックポイントが前提にする観測キー、状態次元、行動次元に強く依存します。そのため、画像トピック名、LeRobot側のfeature key、行動の解釈をYAMLで切り替えます。

詳細は `docs/ARCHITECTURE.md`、`docs/TOPIC_CONTRACT.md`、`docs/SMOLVLA_ADAPTER.md` を参照してください。
