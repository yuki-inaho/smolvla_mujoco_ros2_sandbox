from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from vla_policy.messages import ObservationFrame


class PolicyBackend(ABC):
    """Policy backend interface used by the ROS 2 policy node."""

    @abstractmethod
    def predict(self, frame: ObservationFrame) -> np.ndarray:
        """Return an action vector matching the configured joint order."""


class MockPolicyBackend(PolicyBackend):
    """Deterministic small-motion policy for topic and controller smoke tests."""

    def __init__(self, action_dim: int, amplitude: float = 0.03) -> None:
        self.action_dim = action_dim
        self.amplitude = amplitude
        self.start_time = time.monotonic()

    def predict(self, frame: ObservationFrame) -> np.ndarray:
        elapsed = time.monotonic() - self.start_time
        action = np.zeros(self.action_dim, dtype=np.float32)
        if self.action_dim > 0:
            action[0] = self.amplitude * math.sin(elapsed)
        if self.action_dim > 1:
            action[1] = self.amplitude * math.cos(elapsed * 0.7)
        return action


class SmolVLABackend(PolicyBackend):
    """Thin LeRobot/SmolVLA adapter.

    Important: SmolVLA checkpoints are tied to feature names and dimensions.
    Configure image_keys, state_key, task_key, joint ordering, and action
    post-processing to match the dataset/checkpoint that you fine-tuned.
    """

    def __init__(
        self,
        policy_path: str,
        device: str,
        action_dim: int,
        image_keys: List[str],
        state_key: str,
        task_key: str,
    ) -> None:
        self.policy_path = policy_path
        self.device_name = device
        self.action_dim = action_dim
        self.image_keys = image_keys
        self.state_key = state_key
        self.task_key = task_key

        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import LeRobot SmolVLA dependencies. "
                "Run `pixi run doctor` and verify lerobot[smolvla]/torch installation."
            ) from exc

        self.torch = torch
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.policy = SmolVLAPolicy.from_pretrained(policy_path).to(self.device).eval()

        self.preprocess = None
        self.postprocess = None
        try:
            self.preprocess, self.postprocess = make_pre_post_processors(
                self.policy.config,
                policy_path,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
        except Exception:
            # Older/newer LeRobot versions may alter processor APIs. The backend
            # still attempts direct policy.select_action below.
            self.preprocess = None
            self.postprocess = None

    def predict(self, frame: ObservationFrame) -> np.ndarray:
        candidate = self._make_lerobot_frame(frame)
        with self.torch.inference_mode():
            model_input: Dict[str, Any] = self.preprocess(candidate) if self.preprocess is not None else candidate
            try:
                output = self.policy.select_action(model_input)
            except Exception:
                # Some examples call select_action on the unprocessed frame.
                output = self.policy.select_action(candidate)
            if self.postprocess is not None:
                try:
                    output = self.postprocess(output)
                except Exception:
                    pass

        return self._to_action_vector(output)

    def _make_lerobot_frame(self, frame: ObservationFrame) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            self.state_key: self.torch.as_tensor(frame.joint_positions, dtype=self.torch.float32, device=self.device),
            self.task_key: frame.task_instruction,
        }

        for camera_name, image_key in zip(frame.images.keys(), self.image_keys):
            image = frame.images[camera_name]
            tensor = self.torch.as_tensor(image, dtype=self.torch.float32, device=self.device).permute(2, 0, 1) / 255.0
            result[image_key] = tensor
        return result

    def _to_action_vector(self, output: Any) -> np.ndarray:
        if isinstance(output, dict):
            for key in ("action", "actions", "pred_action"):
                if key in output:
                    output = output[key]
                    break
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        array = np.asarray(output, dtype=np.float32)
        while array.ndim > 1:
            array = array[0]
        if array.size < self.action_dim:
            array = np.pad(array, (0, self.action_dim - array.size))
        return array[: self.action_dim]


def create_backend(
    backend: str,
    action_dim: int,
    policy_path: str,
    device: str,
    image_keys: List[str],
    state_key: str,
    task_key: str,
) -> PolicyBackend:
    if backend == "mock":
        return MockPolicyBackend(action_dim=action_dim)
    if backend == "smolvla":
        return SmolVLABackend(
            policy_path=policy_path,
            device=device,
            action_dim=action_dim,
            image_keys=image_keys,
            state_key=state_key,
            task_key=task_key,
        )
    raise ValueError(f"Unknown policy backend: {backend}")
