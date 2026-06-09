from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare("vla_policy"),
        "config",
        "policy.yaml",
    ])

    config_arg = DeclareLaunchArgument("config", default_value=default_config)
    backend_arg = DeclareLaunchArgument("backend", default_value="mock")
    task_arg = DeclareLaunchArgument("task_instruction", default_value="Move the object to the target area.")
    device_arg = DeclareLaunchArgument("device", default_value="cuda")

    node = Node(
        package="vla_policy",
        executable="policy_node",
        name="policy_node",
        output="screen",
        parameters=[
            LaunchConfiguration("config"),
            {
                "backend": LaunchConfiguration("backend"),
                "task_instruction": LaunchConfiguration("task_instruction"),
                "device": LaunchConfiguration("device"),
            },
        ],
    )

    return LaunchDescription([config_arg, backend_arg, task_arg, device_arg, node])
