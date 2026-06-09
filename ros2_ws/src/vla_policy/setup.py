from glob import glob
from setuptools import setup

package_name = "vla_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Team",
    maintainer_email="robotics@example.com",
    description="ROS 2 policy bridge for MockPolicy and LeRobot/SmolVLA.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_node = vla_policy.policy_node:main",
        ],
    },
)
