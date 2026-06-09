# ロードマップ

## M0: Smoke test

- `pixi install`
- `pixi run doctor`
- `pixi run build`
- `pixi run all-mock`
- `/joint_states` が変化することを確認

## M1: SO-101モデル化

- SO-101 MJCFまたはURDF変換モデルを追加
- 関節名と可動範囲をYAMLで固定
- 3カメラ画像の視点を固定

## M2: データ収集

- モックまたは手動制御でrosbag2を記録
- LeRobotDatasetへの変換スクリプトを追加
- task instructionをepisode単位で保存

## M3: SmolVLA fine-tune

- 50 episode程度から開始
- バリエーションごとのデモ数を確保
- state/action featureを固定して学習

## M4: 閉ループ評価

- fine-tuned checkpointを指定して `policy-smolvla` 起動
- action chunk bufferと安全制限を追加
- 失敗時の停止条件を追加

## M5: 本格制御

- ros2_control / mujoco_ros2_control評価
- JointTrajectoryControllerへ移行
- 実機SO-101へのブリッジ設計
