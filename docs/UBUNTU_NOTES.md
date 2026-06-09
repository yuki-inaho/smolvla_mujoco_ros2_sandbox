# Ubuntu前提の注意点

## Ubuntu 22.04 / ROS 2 Humble

本リポジトリはROS 2 Humbleを前提にし、Python 3.12へ固定しています（focal-cu127 ブランチ）。RoboStack humble は cp39/cp311/cp312 のみ提供しており、lerobot[smolvla] も Python >=3.12 を必須とするため 3.12 を採用しています。pixi + RoboStackを使うため、システムの `/opt/ros/humble` を必須にしません。

## GPU

SmolVLA推論とfine-tuneにはNVIDIA GPUが望ましいです。`pixi run doctor` で `torch.cuda.is_available()` を確認してください。CUDA付きPyTorch wheelの解決は環境差が出やすいため、必要に応じてLeRobot/Torchの導入方針を固定してください。

## MuJoCoレンダリング

初期設定では `use_renderer: false` で、合成RGB画像をpublishします。これはトピック疎通確認用です。実際の視覚方策評価では `use_renderer:=true` を指定し、`MUJOCO_GL=egl`、NVIDIAドライバ、EGL実行環境を整えてください。

```bash
MUJOCO_GL=egl pixi run sim --ros-args -p use_renderer:=true
```

レンダリングが失敗する場合は、まずヘッドレスのまま `backend:=mock` で制御ループを確認してください。

## Mac向け回避策を入れない理由

参考実装ではMac上でMuJoCoノードを `mjpython` 経由で起動する工夫があります。今回はUbuntu前提なので、まず通常のPythonエントリポイントで起動します。これによりlaunch fileとpixi taskが単純になります。
