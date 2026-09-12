"""Launch the harvest simulation, RGB-D broadcaster, and web viewer."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SCENE_ID = "7a7e56ff__camera_l__2026-08-04_08-19-43-646"


def _default_scene_path() -> str:
    candidates = [
        Path(os.environ.get("PIXI_PROJECT_ROOT", "/home/kasm-user/Desktop/smolvla_mujoco_ros2_sandbox"))
        / "ros2_ws"
        / "src"
        / "tomato_harvest_sim"
        / "assets"
        / "scenes"
        / SCENE_ID
        / "scene.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def _default_model_path() -> str:
    candidates = [
        Path(os.environ.get("PIXI_PROJECT_ROOT", "/home/kasm-user/Desktop/smolvla_mujoco_ros2_sandbox"))
        / "ros2_ws"
        / "src"
        / "tomato_harvest_sim"
        / "assets"
        / "tomato_scara.xml",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def _nodes(context, *args, **kwargs):
    scene_path = LaunchConfiguration("scene_path").perform(context) or _default_scene_path()
    model_path = LaunchConfiguration("model_path").perform(context) or _default_model_path()
    port = LaunchConfiguration("port").perform(context)
    if not scene_path:
        raise RuntimeError("scene.json not found; run `pixi run build-scene` or pass scene_path:=")
    if not model_path:
        raise RuntimeError("tomato_scara.xml not found; pass model_path:=")

    return [
        Node(
            package="tomato_harvest_sim",
            executable="harvest_sim_node",
            name="harvest_sim_node",
            output="screen",
            parameters=[{"scene_path": scene_path, "model_path": model_path, "time_scale": 0.5, "loop": True}],
        ),
        Node(
            package="tomato_harvest_sim",
            executable="rgbd_broadcaster_node",
            name="rgbd_broadcaster_node",
            output="screen",
            parameters=[{"scene_path": scene_path}],
        ),
        Node(
            package="tomato_harvest_sim",
            executable="viewer_node",
            name="viewer_node",
            output="screen",
            parameters=[{"scene_path": scene_path, "port": int(port)}],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("scene_path", default_value=""),
            DeclareLaunchArgument("model_path", default_value=""),
            DeclareLaunchArgument("port", default_value="8765"),
            OpaqueFunction(function=_nodes),
        ]
    )
