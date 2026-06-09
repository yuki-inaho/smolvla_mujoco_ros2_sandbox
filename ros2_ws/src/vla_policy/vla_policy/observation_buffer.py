from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sensor_msgs.msg import Image, JointState

from vla_policy.messages import ObservationFrame


def ros_image_to_numpy(msg: Image) -> np.ndarray:
    """Convert a ROS Image message into an HxWxC uint8 numpy array.

    The bridge publishes rgb8 images. bgr8 is accepted for convenience because
    camera drivers often use that encoding.
    """

    if msg.encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        array = array[..., ::-1]
    return array.copy()


class ObservationBuffer:
    """Stores latest state and image messages.

    This intentionally uses latest-value semantics for rapid iteration. For
    stricter temporal alignment, replace this class with message_filters based
    ApproximateTimeSynchronizer.
    """

    def __init__(self, camera_names: list[str], joint_names: list[str], task_instruction: str) -> None:
        self.camera_names = camera_names
        self.joint_names = joint_names
        self.task_instruction = task_instruction
        self._latest_joint_state: Optional[JointState] = None
        self._latest_images: Dict[str, np.ndarray] = {}

    def update_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def update_image(self, camera_name: str, msg: Image) -> None:
        self._latest_images[camera_name] = ros_image_to_numpy(msg)

    def ready(self) -> bool:
        return self._latest_joint_state is not None and all(name in self._latest_images for name in self.camera_names)

    def make_frame(self) -> ObservationFrame:
        if not self.ready():
            missing = [name for name in self.camera_names if name not in self._latest_images]
            raise RuntimeError(f"ObservationBuffer is not ready. Missing cameras={missing}")

        assert self._latest_joint_state is not None
        name_to_position = dict(zip(self._latest_joint_state.name, self._latest_joint_state.position))
        positions = np.asarray([name_to_position.get(name, 0.0) for name in self.joint_names], dtype=np.float32)
        return ObservationFrame(
            joint_names=self.joint_names,
            joint_positions=positions,
            images={name: self._latest_images[name] for name in self.camera_names},
            task_instruction=self.task_instruction,
        )
