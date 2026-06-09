#!/usr/bin/env python3
"""Standalone SmolVLA inference test — no ROS, no LeRobot training framework.

Usage:
    pixi run python scripts/smolvla_standalone_test.py

Confirms:
  - SmolVLAPolicy.from_pretrained("lerobot/smolvla_base") loads
  - make_pre_post_processors works with lerobot 0.5.1 API
  - select_action returns a Tensor of the expected shape
  - Prints action shape and first few values
"""
from __future__ import annotations

import torch
import numpy as np

POLICY_PATH = "lerobot/smolvla_base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASK = "Move the object to the target area.\n"

# Image keys and state key as defined in smolvla_base config.json
IMAGE_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]
STATE_KEY = "observation.state"
STATE_DIM = 6       # from config.json: observation.state shape=[6]
ACTION_DIM = 6      # from config.json: action shape=[6]
IMG_H, IMG_W = 256, 256  # from config.json image shapes


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Loading SmolVLAPolicy from '{POLICY_PATH}' ...")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
    policy = policy.to(DEVICE).eval()
    print(f"Model loaded. config.action_feature.shape = {policy.config.action_feature.shape}")
    print(f"  max_state_dim={policy.config.max_state_dim}, max_action_dim={policy.config.max_action_dim}")
    print(f"  chunk_size={policy.config.chunk_size}, n_action_steps={policy.config.n_action_steps}")
    print(f"  image_features keys: {list(policy.config.image_features.keys())}")

    # Build pre/post processors from pretrained path (loads preprocessor.json / postprocessor.json)
    print("\nBuilding pre/post processors ...")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=POLICY_PATH,
    )
    print("Processors built OK.")

    # Build a synthetic observation dict matching the smolvla_base feature contract
    # Images: float32 [0,1], shape (C, H, W)
    # State: float32, shape (STATE_DIM,)
    # Task: string
    obs = {
        STATE_KEY: torch.zeros(STATE_DIM, dtype=torch.float32),
        "task": TASK,
    }
    for key in IMAGE_KEYS:
        obs[key] = torch.zeros(3, IMG_H, IMG_W, dtype=torch.float32)

    print(f"\nRaw obs keys: {list(obs.keys())}")

    # Run preprocessor (adds batch dim, tokenises task, normalises, moves to device)
    print("Running preprocessor ...")
    batch = preprocess(obs)
    print(f"Preprocessed batch keys: {list(batch.keys())}")

    # Run select_action
    print("Running select_action ...")
    policy.reset()
    with torch.inference_mode():
        action_tensor = policy.select_action(batch)
    print(f"\nselect_action output type: {type(action_tensor)}")
    print(f"select_action output shape: {action_tensor.shape}")
    print(f"select_action output dtype: {action_tensor.dtype}")
    print(f"select_action output device: {action_tensor.device}")

    # Run postprocessor (unnormalise, move to CPU)
    # PolicyAction is a torch.Tensor subclass — cast via as_subclass()
    from lerobot.processor import PolicyAction
    policy_action = action_tensor.as_subclass(PolicyAction)
    post_action = postprocess(policy_action)
    action_np = post_action.cpu().numpy()
    print(f"\nPost-processed action shape: {action_np.shape}")
    print(f"Post-processed action values (normalized→unnormalized): {action_np}")

    # Trim to actual action dim (smolvla_base: action_dim=6)
    action_out = action_np.reshape(-1)[:ACTION_DIM]
    print(f"\nFinal action (first {ACTION_DIM} dims): {action_out}")
    print("\nStandalone inference: PASS")


if __name__ == "__main__":
    main()
