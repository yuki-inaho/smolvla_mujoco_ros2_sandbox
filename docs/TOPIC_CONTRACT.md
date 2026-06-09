# ROS 2 トピック契約

## 観測

| Topic | Type | Publisher | Subscriber | 内容 |
|---|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | `vla_mujoco` | `vla_policy` | 関節名、角度、速度、effort |
| `/top_camera/image_raw` | `sensor_msgs/Image` | `vla_mujoco` | `vla_policy` | 上面カメラRGB画像 |
| `/side_camera/image_raw` | `sensor_msgs/Image` | `vla_mujoco` | `vla_policy` | 側面カメラRGB画像 |
| `/wrist_camera/image_raw` | `sensor_msgs/Image` | `vla_mujoco` | `vla_policy` | 手先カメラRGB画像 |

## 行動

| Topic | Type | Publisher | Subscriber | 内容 |
|---|---|---|---|---|
| `/arm_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | `vla_policy` | `vla_mujoco` | アーム関節の目標軌道 |
| `/gripper_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | `vla_policy` | `vla_mujoco` | グリッパの目標軌道 |

## 設計上の注意

1. 関節順序は `joint_names` で固定します。SmolVLAの出力次元と一致しない場合、必ずYAML側で明示します。
2. 画像は初期版では `rgb8` とします。実機カメラやOpenCVドライバで `bgr8` が来る場合、`vla_policy` 側でRGBへ変換します。
3. 同期は初期版ではlatest-value方式です。厳密同期が必要になったら `message_filters.ApproximateTimeSynchronizer` へ置き換えます。
4. actionは `delta_joint` と `absolute_joint` を切り替えられるようにします。SmolVLAチェックポイントがどちらを前提にしているかを確認してください。
