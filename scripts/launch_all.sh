#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-mock}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}/ros2_ws"

cleanup() {
  jobs -p | xargs -r kill || true
}
trap cleanup EXIT INT TERM

ros2 launch vla_mujoco sim.launch.py &
sleep 3
ros2 launch vla_policy policy.launch.py backend:="${BACKEND}" &
wait
