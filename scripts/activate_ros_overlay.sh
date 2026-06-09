#!/usr/bin/env bash
# Pixi activation hook. This file must succeed even before `pixi run build` creates ros2_ws/install/setup.sh.
set +u

export PIXI_PROJECT_ROOT="${PIXI_PROJECT_ROOT:-$(pwd)}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

if [ -f "${PIXI_PROJECT_ROOT}/ros2_ws/install/setup.sh" ]; then
  # shellcheck disable=SC1091
  source "${PIXI_PROJECT_ROOT}/ros2_ws/install/setup.sh"
fi
