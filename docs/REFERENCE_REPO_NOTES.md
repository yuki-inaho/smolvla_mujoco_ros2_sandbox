# 参考リポジトリ・記事から取り込むべき要点

## RoboStack / ros-humble

ROS 2 Humbleをconda/pixi環境内に閉じ込める方針を採用します。`robostack-humble` と `conda-forge` をpixi channelとして使う構成にします。

## RoboStack / vinca

vincaはROSパッケージをconda recipe化する側の道具です。本リポジトリの利用者がすぐ触るものではありません。ただし、将来 `mujoco_ros2_control` などを社内conda channelへ固める場合は候補になります。

## pixi_ros2_gazebo_example

`pixi run colcon build --symlink-install`、`[activation] scripts = ["install/setup.sh"]`、`pixi run ros2 launch ...` という運用パターンを踏襲します。ただし、本リポジトリでは `ros2_ws/install/setup.sh` が存在しない初期状態でもpixi activationが失敗しないよう、`scripts/activate_ros_overlay.sh` を挟みます。

## ros2-so101-mujoco / ABEJA記事

`lerobot_mujoco` と `lerobot_vla` を分ける構成、MuJoCo側がjoint stateと複数カメラ画像をpublishし、VLA側がそれをLeRobot形式へ変換してJointTrajectoryを返す構成を踏襲します。一方で、今回はUbuntu前提なのでMac向けの `mjpython` 起動回避策は標準化しません。

## mujoco_ros2_control

将来的には有力です。ただし初期開発では、ros2_control設定、URDF/MJCF変換、controller設定の調整がボトルネックになり得ます。本リポジトリでは、まず標準ROSメッセージでVLA閉ループを作り、後から差し替える方針にします。
