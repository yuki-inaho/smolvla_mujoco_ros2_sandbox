# 提案書: pixi + ROS 2 + MuJoCo + LeRobot + SmolVLA 単一リポジトリ構成

## 結論

Ubuntu前提であれば、単一リポジトリは以下の構造が扱いやすいです。

```text
.
├── pixi.toml
├── scripts/
├── docs/
└── ros2_ws/
    └── src/
        ├── vla_mujoco/
        └── vla_policy/
```

初期段階では、MuJoCoとSmolVLAを同一プロセスにせず、ROS 2トピックで分離します。これにより、MuJoCoレンダリング、SmolVLAモデルロード、CUDA、LeRobot API変更の問題を個別に切り分けられます。

## 実装方針

1. `pixi.toml` は `linux-64` 固定、ROS 2 Humble、MuJoCo、LeRobot/SmolVLAを管理します。
2. `vla_mujoco` は `/joint_states` と3カメラ画像をpublishし、JointTrajectoryをsubscribeします。
3. `vla_policy` は観測をLeRobot形式へ変換し、モック方策またはSmolVLAで行動を生成します。
4. 初期疎通は `backend:=mock`、VLA接続は `backend:=smolvla` で切り替えます。
5. 将来 `mujoco_ros2_control` に移行する場合でも、`vla_policy` の契約は維持します。

## 採用理由

- ラピッドに失敗箇所を切り分けやすい。
- ROS 2標準メッセージだけで初期版を構成できる。
- LeRobot/SmolVLAのfeature key変更をYAMLで吸収できる。
- UbuntuではMac向けの `mjpython` 回避策が不要で、構成を単純にできる。
- 将来のros2_control化、実機接続、LeRobotDataset化へ拡張しやすい。

## 最初の実行順序

```bash
pixi install
pixi run doctor
pixi run build
pixi run sim
pixi run policy-mock
```

SmolVLAを使う段階では、対象checkpointに合わせて `ros2_ws/src/vla_policy/config/policy.yaml` を修正します。
