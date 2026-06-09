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
    """Thin LeRobot/SmolVLA adapter for lerobot 0.5.1.

    API notes (verified against lerobot 0.5.1 source):

    - SmolVLAPolicy.from_pretrained(path) loads the model.
    - make_pre_post_processors(config, pretrained_path) returns
      (preprocessor, postprocessor) pipelines.
    - preprocessor takes a raw obs dict with un-batched tensors and a "task" string;
      it adds the batch dimension, tokenises the task string, normalises, and moves
      to device.
    - policy.select_action(batch) returns a Tensor of shape (1, action_dim) on GPU.
    - postprocessor takes that Tensor cast to PolicyAction (a Tensor subclass via
      .as_subclass()) and returns an unnormalised Tensor on CPU.

    smolvla_base input contract (from config.json):
      - observation.state  shape (6,) float32 on CPU before preprocessing
      - observation.images.camera1/2/3  shape (3, H, W) float32 [0,1] on CPU
      - "task"  str (task instruction, preprocessor appends \\n if missing)
      action output: shape (1, 6) float32 → trimmed to action_dim for ROS
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
        self.action_dim = action_dim
        self.image_keys = image_keys
        self.state_key = state_key
        self.task_key = task_key

        try:
            import torch
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.processor import PolicyAction
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import LeRobot SmolVLA dependencies. "
                "Run `pixi run doctor` and verify lerobot[smolvla]/torch installation."
            ) from exc

        self.torch = torch
        self.PolicyAction = PolicyAction
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

        self.policy = SmolVLAPolicy.from_pretrained(policy_path)
        self.policy = self.policy.to(self.device).eval()
        self.policy.reset()

        # Build pre/post processors from pretrained checkpoint.
        # make_pre_post_processors(config, pretrained_path) is the correct
        # lerobot 0.5.1 signature — no extra keyword overrides needed.
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_path,
        )

    def predict(self, frame: ObservationFrame) -> np.ndarray:
        obs = self._make_obs_dict(frame)

        # Debug observation stats on first call only (cleared after logging).
        if not getattr(self, "_obs_logged", False):
            self._obs_logged = True
            import sys
            state_t = obs[self.state_key]
            print(
                f"[SmolVLABackend] OBS CHECK: state shape={state_t.shape} "
                f"min={state_t.min():.4f} max={state_t.max():.4f}",
                file=sys.stderr, flush=True,
            )
            for k, v in obs.items():
                if hasattr(v, "shape"):
                    print(
                        f"[SmolVLABackend] OBS CHECK: {k} shape={v.shape} "
                        f"mean={v.float().mean():.4f} std={v.float().std():.4f}",
                        file=sys.stderr, flush=True,
                    )

        with self.torch.inference_mode():
            # Preprocessor: adds batch dim, tokenises task, normalises, moves to device
            batch = self.preprocess(obs)
            # select_action: returns Tensor (1, action_dim) on GPU
            action_tensor = self.policy.select_action(batch)
            # Postprocessor: unnormalise, move to CPU
            # PolicyAction is a torch.Tensor subclass; cast via as_subclass()
            policy_action = action_tensor.as_subclass(self.PolicyAction)
            unnorm_action = self.postprocess(policy_action)

        action_np = unnorm_action.cpu().numpy().reshape(-1)
        trimmed = self._trim_to_action_dim(action_np)

        # Log action on first call.
        if not getattr(self, "_action_logged", False):
            self._action_logged = True
            import sys
            print(
                f"[SmolVLABackend] ACTION: raw={action_np[:6].tolist()} "
                f"trimmed={trimmed.tolist()}",
                file=sys.stderr, flush=True,
            )

        return trimmed

    def _make_obs_dict(self, frame: ObservationFrame) -> Dict[str, Any]:
        """Build un-batched obs dict that the preprocessor expects.

        The preprocessor's AddBatchDimensionProcessorStep will add the batch dim,
        so tensors here are (C, H, W) for images and (state_dim,) for state.
        """
        obs: Dict[str, Any] = {
            self.state_key: self.torch.as_tensor(
                frame.joint_positions, dtype=self.torch.float32
            ),
            self.task_key: frame.task_instruction,
        }
        for camera_name, image_key in zip(frame.images.keys(), self.image_keys):
            image = frame.images[camera_name]  # HWC uint8 numpy
            # Convert HWC uint8 → CHW float32 [0, 1]
            tensor = (
                self.torch.as_tensor(image, dtype=self.torch.float32)
                .permute(2, 0, 1)
                .div(255.0)
            )
            obs[image_key] = tensor
        return obs

    def _trim_to_action_dim(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.size < self.action_dim:
            action = np.pad(action, (0, self.action_dim - action.size))
        return action[: self.action_dim]


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
