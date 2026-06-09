from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare("vla_mujoco"),
        "config",
        "mujoco_sim.yaml",
    ])
    default_model = PathJoinSubstitution([
        FindPackageShare("vla_mujoco"),
        "assets",
        "toy_arm.xml",
    ])

    config_arg = DeclareLaunchArgument("config", default_value=default_config)
    model_arg = DeclareLaunchArgument("model_path", default_value=default_model)
    renderer_arg = DeclareLaunchArgument("use_renderer", default_value="false")

    node = Node(
        package="vla_mujoco",
        executable="mujoco_sim_node",
        name="mujoco_sim_node",
        output="screen",
        parameters=[
            LaunchConfiguration("config"),
            {
                "model_path": LaunchConfiguration("model_path"),
                "use_renderer": LaunchConfiguration("use_renderer"),
            },
        ],
    )

    return LaunchDescription([config_arg, model_arg, renderer_arg, node])
