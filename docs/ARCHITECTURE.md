# アーキテクチャ設計

## 方針

ラピッド開発では、最初から `mujoco_ros2_control` や複雑なコントローラ構成へ寄せすぎない方が良いです。初期版では、以下の責務に分けます。

- `vla_mujoco`: MuJoCoを進め、関節状態と画像をROS 2標準メッセージでpublishする。JointTrajectoryをsubscribeし、MuJoCo側の制御入力へ変換する。
- `vla_policy`: ROS 2観測をLeRobot形式の観測辞書に変換し、モック方策またはSmolVLAで行動を生成する。
- `pixi`: ROS 2、Python、MuJoCo、LeRobot、補助ツールの依存関係と起動タスクを一元管理する。

## なぜ2ノード構成にするか

MuJoCo側とVLA推論側は、速度、依存関係、失敗モードが異なります。MuJoCoは100〜500Hz程度で軽く回したい一方、SmolVLAは数Hz〜十数Hzの推論周期になります。単一プロセスにすると、モデルロード失敗やGPUメモリ不足でシミュレーターごと落ちます。そのためROS 2ノードを分け、トピック境界で疎結合にします。

## 初期版のデータフロー

```text
vla_mujoco
  publish:
    /joint_states
    /top_camera/image_raw
    /side_camera/image_raw
    /wrist_camera/image_raw
    /top_camera/camera_info
    /side_camera/camera_info
    /wrist_camera/camera_info
  subscribe:
    /arm_controller/joint_trajectory
    /gripper_controller/joint_trajectory

vla_policy
  subscribe:
    /joint_states
    /top_camera/image_raw
    /side_camera/image_raw
    /wrist_camera/image_raw
  publish:
    /arm_controller/joint_trajectory
    /gripper_controller/joint_trajectory
```

## 将来の移行先

### ros2_control化

`mujoco_ros2_control` が十分安定し、対象ロボットのURDF/MJCF変換とcontroller設定が整ったら、`vla_mujoco` のJointTrajectory subscriberをros2_control側へ置き換えます。このとき `vla_policy` は基本的に変更しません。

### SO-101本体への接続

SO-101実機や既存LeRobotロボットクラスへ接続する場合も、`vla_policy` は観測と行動の変換層として残せます。シミュレーター側だけを実機ブリッジに差し替える設計が望ましいです。
