from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from trajectory_msgs.msg import JointTrajectory


class MujocoSimNode(Node):
    """Small MuJoCo-to-ROS2 bridge for rapid VLA policy iteration.

    The node intentionally publishes only standard ROS messages. This keeps the
    first iteration lightweight and leaves a later migration to ros2_control open.
    """

    def __init__(self) -> None:
        super().__init__("mujoco_sim_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("sim_hz", 200.0)
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("use_renderer", False)
        self.declare_parameter("image_width", 320)
        self.declare_parameter("image_height", 240)
        self.declare_parameter("joint_names", ["joint1", "joint2", "gripper"])
        self.declare_parameter("camera_names", ["top", "side", "wrist"])
        self.declare_parameter("camera_topics", ["/top_camera/image_raw", "/side_camera/image_raw", "/wrist_camera/image_raw"])
        self.declare_parameter("camera_info_topics", ["/top_camera/camera_info", "/side_camera/camera_info", "/wrist_camera/camera_info"])
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("arm_command_topic", "/arm_controller/joint_trajectory")
        self.declare_parameter("gripper_command_topic", "/gripper_controller/joint_trajectory")
        # Optional initial joint positions (radians). Empty = use MuJoCo model defaults.
        self.declare_parameter("initial_qpos", [0.0, 0.0, 0.0])

        self.model_path = str(self.get_parameter("model_path").value)
        self.sim_hz = float(self.get_parameter("sim_hz").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.use_renderer = bool(self.get_parameter("use_renderer").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.joint_names = list(self.get_parameter("joint_names").value)
        self.camera_names = list(self.get_parameter("camera_names").value)
        self.camera_topics = list(self.get_parameter("camera_topics").value)
        self.camera_info_topics = list(self.get_parameter("camera_info_topics").value)

        if len(self.camera_names) != len(self.camera_topics):
            raise ValueError("camera_names and camera_topics must have the same length")
        if len(self.camera_names) != len(self.camera_info_topics):
            raise ValueError("camera_names and camera_info_topics must have the same length")

        self.model = None
        self.data = None
        self.mujoco = None
        self.renderer = None
        self._initial_qpos = list(self.get_parameter("initial_qpos").value)
        self._load_mujoco()

        self.current_positions = np.zeros(len(self.joint_names), dtype=np.float64)
        self.target_positions = np.zeros(len(self.joint_names), dtype=np.float64)
        if self.model is not None and self.data is not None:
            # Apply user-specified initial joint positions before reading qpos.
            if self._initial_qpos:
                count_init = min(len(self._initial_qpos), self.model.nq)
                self.data.qpos[:count_init] = self._initial_qpos[:count_init]
                self.mujoco.mj_forward(self.model, self.data)
            count = min(len(self.joint_names), self.model.nq)
            self.current_positions[:count] = self.data.qpos[:count]
            self.target_positions[:count] = self.current_positions[:count]

        joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.joint_pub = self.create_publisher(JointState, joint_state_topic, 10)
        self.image_pubs = {
            name: self.create_publisher(Image, topic, 10)
            for name, topic in zip(self.camera_names, self.camera_topics)
        }
        self.camera_info_pubs = {
            name: self.create_publisher(CameraInfo, topic, 10)
            for name, topic in zip(self.camera_names, self.camera_info_topics)
        }

        arm_topic = str(self.get_parameter("arm_command_topic").value)
        gripper_topic = str(self.get_parameter("gripper_command_topic").value)
        self.create_subscription(JointTrajectory, arm_topic, self._on_trajectory, 10)
        self.create_subscription(JointTrajectory, gripper_topic, self._on_trajectory, 10)

        self._last_publish_time = 0.0
        self._start_time = time.monotonic()
        self.create_timer(1.0 / self.sim_hz, self._tick)
        self.get_logger().info(
            f"Started MuJoCo ROS wrapper: joints={self.joint_names}, model={self.model_path or 'mock'}"
        )

    def _load_mujoco(self) -> None:
        try:
            import mujoco

            self.mujoco = mujoco
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"MuJoCo import failed; running mock simulation only: {exc}")
            return

        if not self.model_path:
            self.get_logger().warning("model_path is empty; running mock simulation only")
            return

        path = Path(self.model_path).expanduser()
        if not path.exists():
            self.get_logger().warning(f"MuJoCo model not found: {path}; running mock simulation only")
            return

        self.model = self.mujoco.MjModel.from_xml_path(str(path))
        self.data = self.mujoco.MjData(self.model)

        if self.use_renderer:
            try:
                self.renderer = self.mujoco.Renderer(self.model, height=self.image_height, width=self.image_width)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"Renderer initialization failed; synthetic images will be used: {exc}")
                self.renderer = None

    def _on_trajectory(self, msg: JointTrajectory) -> None:
        if not msg.points:
            return
        positions = list(msg.points[-1].positions)
        if not positions:
            return

        if msg.joint_names:
            name_to_value = dict(zip(msg.joint_names, positions))
            for index, name in enumerate(self.joint_names):
                if name in name_to_value:
                    self.target_positions[index] = float(name_to_value[name])
        else:
            count = min(len(positions), len(self.target_positions))
            self.target_positions[:count] = np.asarray(positions[:count], dtype=np.float64)

    def _tick(self) -> None:
        self._advance_simulation()
        now = time.monotonic()
        if now - self._last_publish_time >= 1.0 / self.publish_hz:
            self._last_publish_time = now
            self._publish_joint_state()
            self._publish_images()

    def _advance_simulation(self) -> None:
        if self.model is not None and self.data is not None:
            count_ctrl = min(self.model.nu, len(self.target_positions))
            if count_ctrl > 0:
                self.data.ctrl[:count_ctrl] = self.target_positions[:count_ctrl]
            self.mujoco.mj_step(self.model, self.data)
            count_q = min(self.model.nq, len(self.current_positions))
            self.current_positions[:count_q] = self.data.qpos[:count_q]
        else:
            # Mock fallback: first-order convergence to the latest target.
            alpha = min(1.0, 8.0 / self.sim_hz)
            self.current_positions += alpha * (self.target_positions - self.current_positions)

    def _publish_joint_state(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_positions.astype(float).tolist()
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)
        self.joint_pub.publish(msg)

    def _publish_images(self) -> None:
        for camera_name in self.camera_names:
            image = self._render_or_synthesize(camera_name)
            stamp = self.get_clock().now().to_msg()
            image_msg = self._image_to_msg(image, frame_id=f"{camera_name}_camera", stamp=stamp)
            info_msg = self._camera_info_msg(frame_id=f"{camera_name}_camera", stamp=stamp)
            self.image_pubs[camera_name].publish(image_msg)
            self.camera_info_pubs[camera_name].publish(info_msg)

    def _render_or_synthesize(self, camera_name: str) -> np.ndarray:
        if self.renderer is not None and self.model is not None and self.data is not None:
            try:
                self.renderer.update_scene(self.data, camera=camera_name)
                return np.asarray(self.renderer.render(), dtype=np.uint8)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().debug(f"Renderer failed for camera={camera_name}: {exc}")

        elapsed = time.monotonic() - self._start_time
        h, w = self.image_height, self.image_width
        yy, xx = np.mgrid[0:h, 0:w]
        joint_signal = float(np.sum(self.current_positions)) if len(self.current_positions) else 0.0
        camera_offset = (sum(ord(c) for c in camera_name) % 37) / 37.0
        image = np.empty((h, w, 3), dtype=np.uint8)
        image[..., 0] = ((xx / max(w - 1, 1)) * 255).astype(np.uint8)
        image[..., 1] = ((yy / max(h - 1, 1)) * 255).astype(np.uint8)
        wave = 0.5 + 0.5 * math.sin(elapsed + joint_signal + camera_offset)
        image[..., 2] = int(255 * wave)
        return image

    def _image_to_msg(self, image: np.ndarray, frame_id: str, stamp) -> Image:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected HxWx3 uint8 RGB image")
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(image.shape[1] * 3)
        msg.data = image.tobytes()
        return msg

    def _camera_info_msg(self, frame_id: str, stamp) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = self.image_height
        msg.width = self.image_width
        fx = fy = float(self.image_width)
        cx = float(self.image_width) / 2.0
        cy = float(self.image_height) / 2.0
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
