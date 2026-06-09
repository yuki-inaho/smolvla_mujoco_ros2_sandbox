---
name: smolvla-infer-smoke
description: Smoke-test SmolVLA inference (load + select_action) without ROS; use to debug the lerobot 0.5.1 API fast.
---

## When to use

Use this when you need to quickly verify that SmolVLA inference works end to end
(model load -> preprocess -> `select_action` -> postprocess) WITHOUT bringing up
the ROS 2 stack. This is the fastest iteration loop when debugging the lerobot
0.5.1 API or the pre/post-processor pipeline used by `SmolVLABackend`.

If this standalone test passes but the ROS node fails, the bug is in the ROS glue
(`ros2_ws/src/vla_policy/vla_policy/backends.py`), not in lerobot itself.

## Steps

Every shell must start by putting pixi on PATH (PATH is NOT persistent):

```bash
export PATH="$HOME/.pixi/bin:$PATH"
```

Run the standalone test (run with the Bash tool's `dangerouslyDisableSandbox=true`
because the first run needs network/GPU):

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi run python scripts/smolvla_standalone_test.py
```

- First run fetches `lerobot/smolvla_base` (~865MB) from the Hugging Face Hub, so it
  needs network. Subsequent runs are HF-cached and run offline.
- Stay on the `focal-cu127` branch. Never touch the `main` branch.

Expected output (PASS):

- Model loads: prints `config.action_feature.shape`, `max_state_dim`, `max_action_dim`,
  `chunk_size`, `n_action_steps`, and the `image_features` keys.
- Preprocessor and postprocessor build OK from the pretrained path.
- `select_action` returns a `torch.Tensor` of shape `(1, 6)`, dtype `float32`, on
  `cuda:0` (or `cpu` if no GPU).
- Post-processed action is a non-trivial 6-vector; the script ends with
  `Standalone inference: PASS`.

## Why / gotchas

The original skeleton was written against an unknown lerobot version: it used
non-existent kwargs and silent `try/except` fallbacks that masked real failures.
Always validate against the INSTALLED lerobot 0.5.1 source. Ground truth, confirmed
against both `scripts/smolvla_standalone_test.py` and `SmolVLABackend` in
`ros2_ws/src/vla_policy/vla_policy/backends.py`:

- Load with `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")`
  (import from `lerobot.policies.smolvla.modeling_smolvla`).
- Build processors with
  `make_pre_post_processors(policy.config, pretrained_path=<path>)`
  (import from `lerobot.policies.factory`). There is NO `preprocessor_overrides`
  kwarg — passing one fails.
- `preprocess` takes an UN-batched dict: `observation.state` (1D tensor),
  `observation.images.camera1/2/3` (CHW float32 in `[0,1]`), and a `"task"`
  string. The preprocessor adds the batch dim, tokenises the task (appends `\n`
  if missing), normalises, and moves to device.
- `select_action` returns a `Tensor (1, 6)`. Before `postprocess`, cast it with
  `action.as_subclass(PolicyAction)` (`PolicyAction` from `lerobot.processor` is a
  `torch.Tensor` subclass) — do not wrap or reconstruct it.
- `smolvla_base` is 6-DoF (SO-100). `max_state_dim` is large/flexible (32), so a
  shorter state vector works without manual padding; the standalone script uses a
  6-dim state to match the model's native dims.
- The checkpoint is UNTRAINED on our toy arm, so the returned actions are NOT
  meaningful — fine-tuning is required for real behaviour. A non-trivial 6-vector
  only proves the pipeline runs, not that the policy is useful.
