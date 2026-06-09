from __future__ import annotations

from typing import List, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from vla_policy.backends import create_backend
from vla_policy.observation_buffer import ObservationBuffer


class PolicyNode(Node):
    """ROS 2 node that converts ROS observations into policy actions."""

    def __init__(self) -> None:
        super().__init__("policy_node")

        self.declare_parameter("backend", "mock")
        self.declare_parameter("policy_path", "lerobot/smolvla_base")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("task_instruction", "Move the object to the target area.")
        self.declare_parameter("control_hz", 5.0)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("arm_command_topic", "/arm_controller/joint_trajectory")
        self.declare_parameter("gripper_command_topic", "/gripper_controller/joint_trajectory")
        self.declare_parameter("joint_names", ["joint1", "joint2", "gripper"])
        self.declare_parameter("arm_joint_names", ["joint1", "joint2"])
        self.declare_parameter("gripper_joint_names", ["gripper"])
        self.declare_parameter("camera_names", ["top", "side", "wrist"])
        self.declare_parameter("camera_topics", ["/top_camera/image_raw", "/side_camera/image_raw", "/wrist_camera/image_raw"])
        self.declare_parameter("camera_keys", ["observation.images.top", "observation.images.side", "observation.images.wrist"])
        self.declare_parameter("state_key", "observation.state")
        self.declare_parameter("task_key", "task")
        self.declare_parameter("action_mode", "delta_joint")
        self.declare_parameter("action_scale", 0.05)
        self.declare_parameter("max_delta", 0.05)
        self.declare_parameter("command_duration_sec", 0.2)

        backend_name = str(self.get_parameter("backend").value)
        policy_path = str(self.get_parameter("policy_path").value)
        device = str(self.get_parameter("device").value)
        self.task_instruction = str(self.get_parameter("task_instruction").value)
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.joint_names = list(self.get_parameter("joint_names").value)
        self.arm_joint_names = list(self.get_parameter("arm_joint_names").value)
        self.gripper_joint_names = list(self.get_parameter("gripper_joint_names").value)
        self.camera_names = list(self.get_parameter("camera_names").value)
        self.camera_topics = list(self.get_parameter("camera_topics").value)
        self.camera_keys = list(self.get_parameter("camera_keys").value)
        self.state_key = str(self.get_parameter("state_key").value)
        self.task_key = str(self.get_parameter("task_key").value)
        self.action_mode = str(self.get_parameter("action_mode").value)
        self.action_scale = float(self.get_parameter("action_scale").value)
        self.max_delta = float(self.get_parameter("max_delta").value)
        self.command_duration_sec = float(self.get_parameter("command_duration_sec").value)

        if len(self.camera_names) != len(self.camera_topics):
            raise ValueError("camera_names and camera_topics must have the same length")
        if len(self.camera_names) != len(self.camera_keys):
            raise ValueError("camera_names and camera_keys must have the same length")

        self.buffer = ObservationBuffer(
            camera_names=self.camera_names,
            joint_names=self.joint_names,
            task_instruction=self.task_instruction,
        )
        self.latest_joint_positions = np.zeros(len(self.joint_names), dtype=np.float32)

        self.backend = create_backend(
            backend=backend_name,
            action_dim=len(self.joint_names),
            policy_path=policy_path,
            device=device,
            image_keys=self.camera_keys,
            state_key=self.state_key,
            task_key=self.task_key,
        )

        self.arm_pub = self.create_publisher(JointTrajectory, str(self.get_parameter("arm_command_topic").value), 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, str(self.get_parameter("gripper_command_topic").value), 10)
        self.create_subscription(JointState, str(self.get_parameter("joint_state_topic").value), self._on_joint_state, 10)

        for camera_name, topic in zip(self.camera_names, self.camera_topics):
            self.create_subscription(Image, topic, lambda msg, name=camera_name: self._on_image(name, msg), 10)

        self.create_timer(1.0 / self.control_hz, self._tick)
        self.get_logger().info(f"Started policy node backend={backend_name}, task='{self.task_instruction}'")

    def _on_joint_state(self, msg: JointState) -> None:
        self.buffer.update_joint_state(msg)
        name_to_position = dict(zip(msg.name, msg.position))
        self.latest_joint_positions = np.asarray([name_to_position.get(name, 0.0) for name in self.joint_names], dtype=np.float32)

    def _on_image(self, camera_name: str, msg: Image) -> None:
        try:
            self.buffer.update_image(camera_name, msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))

    def _tick(self) -> None:
        if not self.buffer.ready():
            self.get_logger().debug("Waiting for joint states and camera images")
            return

        frame = self.buffer.make_frame()
        try:
            raw_action = self.backend.predict(frame)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Policy inference failed: {type(exc).__name__}: {exc}")
            return

        target = self._postprocess_action(raw_action)
        self._publish_commands(target)

    def _postprocess_action(self, raw_action: np.ndarray) -> np.ndarray:
        action = np.asarray(raw_action, dtype=np.float32)
        if action.size < len(self.joint_names):
            action = np.pad(action, (0, len(self.joint_names) - action.size))
        action = action[: len(self.joint_names)]

        if self.action_mode == "absolute_joint":
            return action
        if self.action_mode == "delta_joint":
            delta = np.clip(action * self.action_scale, -self.max_delta, self.max_delta)
            return self.latest_joint_positions + delta
        raise ValueError(f"Unsupported action_mode: {self.action_mode}")

    def _publish_commands(self, target: np.ndarray) -> None:
        name_to_target = dict(zip(self.joint_names, target.astype(float).tolist()))
        stamp = self.get_clock().now().to_msg()
        duration = Duration(sec=int(self.command_duration_sec), nanosec=int((self.command_duration_sec % 1.0) * 1e9))

        if self.arm_joint_names:
            arm_msg = self._trajectory_msg(self.arm_joint_names, name_to_target, stamp, duration)
            self.arm_pub.publish(arm_msg)
        if self.gripper_joint_names:
            gripper_msg = self._trajectory_msg(self.gripper_joint_names, name_to_target, stamp, duration)
            self.gripper_pub.publish(gripper_msg)

    @staticmethod
    def _trajectory_msg(joint_names: List[str], name_to_target: dict[str, float], stamp, duration: Duration) -> JointTrajectory:
        msg = JointTrajectory()
        msg.header.stamp = stamp
        msg.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = [name_to_target.get(name, 0.0) for name in joint_names]
        point.time_from_start = duration
        msg.points.append(point)
        return msg


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
