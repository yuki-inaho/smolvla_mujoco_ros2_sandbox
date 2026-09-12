import os
from glob import glob

from setuptools import setup

package_name = "tomato_harvest_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "assets"), glob("assets/*.xml")),
        (os.path.join("share", package_name, "viewer"), glob("viewer/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Team",
    maintainer_email="robotics@example.com",
    description="RGB-D point-cloud driven tomato harvest simulation on MuJoCo.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "scene_builder = tomato_harvest_sim.scene_builder:main",
            "harvest_sim_node = tomato_harvest_sim.harvest_sim_node:main",
            "rgbd_broadcaster_node = tomato_harvest_sim.rgbd_broadcaster_node:main",
            "viewer_node = tomato_harvest_sim.viewer_node:main",
        ],
    },
)
