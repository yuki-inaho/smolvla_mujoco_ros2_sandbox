# 実装計画

## Phase 0: リポジトリ初期化

- `pixi.toml` を置く。
- `ros2_ws/src` に `vla_mujoco` と `vla_policy` を置く。
- `pixi run build` でROS 2 Python packageがビルドできることを確認する。

## Phase 1: トピック疎通

- `vla_mujoco` が `/joint_states` と3カメラ画像をpublishする。
- `vla_policy` がそれらをsubscribeする。
- `backend:=mock` がJointTrajectoryをpublishする。
- `vla_mujoco` がJointTrajectoryを受け、MuJoCoまたはモック状態を更新する。

## Phase 2: SO-101/MJCFへの差し替え

- `assets/toy_arm.xml` をSO-101のMJCF/URDF変換済みモデルへ置き換える。
- `joint_names`、`arm_joint_names`、`gripper_joint_names` を対象モデルに合わせる。
- カメラ名とカメラトピックを固定する。

## Phase 3: LeRobotデータ化

- rosbag2または直接LeRobotDataset writerを使い、画像・状態・行動・taskを保存する。
- SmolVLA fine-tune前に、状態次元と行動次元をdataset featuresとして確定する。

## Phase 4: SmolVLA推論

- `backend:=smolvla` を有効化する。
- checkpointのfeature keyに合わせ、`policy.yaml` の `camera_keys`、`state_key`、`task_key` を修正する。
- action chunkの扱い、delta/absolute、IKの有無を確定する。

## Phase 5: ros2_control移行

- `mujoco_ros2_control` を評価する。
- Controller Manager、JointTrajectoryController、JointStateBroadcasterへ接続する。
- `vla_policy` は原則そのまま維持する。
