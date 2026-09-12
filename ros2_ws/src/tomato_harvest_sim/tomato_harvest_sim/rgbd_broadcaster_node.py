"""Publish recorded RGB-D frames (from scene.json) as ROS 2 camera topics.

Topics:
  /camera/color/image_raw    sensor_msgs/Image   rgb8   800x600
  /camera/depth/image_raw    sensor_msgs/Image   16UC1  640x480
  /camera/color/camera_info  sensor_msgs/CameraInfo
  /camera/depth/camera_info  sensor_msgs/CameraInfo
  /camera/depth/points       sensor_msgs/PointCloud2 (xyz32, depth frame)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

RGB_FRAME_ID = "camera_rgb_optical_frame"
DEPTH_FRAME_ID = "camera_depth_optical_frame"
DEFAULT_DEPTH_SCALE = 0.001


def load_scene(path: str) -> dict:
    scene_path = Path(path).expanduser()
    if not scene_path.exists():
        raise FileNotFoundError(f"scene.json not found: {scene_path}")
    return json.loads(scene_path.read_text())


def load_rgb(path: str) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_depth(path: str, scale: float = 0.001) -> np.ndarray:
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"failed to read depth image: {path}")
    if raw.dtype != np.uint16:
        raise ValueError(f"depth image must be uint16, got {raw.dtype} for {path}")
    return raw.astype(np.float32) * float(scale)


def depth_to_points(depth: np.ndarray, k, stride: int = 4) -> np.ndarray:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    fx, _, cx = float(k[0]), float(k[1]), float(k[2])
    _, fy, cy = float(k[3]), float(k[4]), float(k[5])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("camera intrinsics fx/fy must be non-zero")
    height, width = depth.shape
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[0:height:stride, 0:width:stride]
    valid = z > 0.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x[valid], y[valid], z[valid]], axis=1)
    return np.ascontiguousarray(points, dtype=np.float32)


def make_camera_info(frame_id: str, width: int, height: int, k, p=None) -> CameraInfo:
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width = int(width)
    info.height = int(height)
    info.distortion_model = ""
    info.d = []
    k_values = [float(value) for value in k]
    info.k = k_values
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    if p is not None and len(p) == 12:
        info.p = [float(value) for value in p]
    else:
        info.p = [
            k_values[0],
            0.0,
            k_values[2],
            0.0,
            0.0,
            k_values[4],
            k_values[5],
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
    return info


def make_image(stamp, frame_id: str, array: np.ndarray, encoding: str) -> Image:
    message = Image()
    message.header = Header(stamp=stamp, frame_id=frame_id)
    message.height = int(array.shape[0])
    message.width = int(array.shape[1])
    message.encoding = encoding
    message.is_bigendian = 0
    if encoding == "rgb8":
        message.step = int(array.shape[1] * 3)
        message.data = np.ascontiguousarray(array, dtype=np.uint8).tobytes()
    elif encoding == "16UC1":
        message.step = int(array.shape[1] * 2)
        message.data = np.ascontiguousarray(array, dtype="<u2").tobytes()
    else:
        raise ValueError(f"unsupported encoding: {encoding}")
    return message


def make_pointcloud2(stamp, frame_id: str, points: np.ndarray) -> PointCloud2:
    cloud = PointCloud2()
    cloud.header = Header(stamp=stamp, frame_id=frame_id)
    cloud.height = 1
    cloud.width = int(points.shape[0])
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 12
    cloud.row_step = 12 * int(points.shape[0])
    cloud.is_dense = True
    cloud.data = np.ascontiguousarray(points, dtype=np.float32).tobytes()
    return cloud


class RgbdBroadcaster(Node):
    def __init__(self) -> None:
        super().__init__("rgbd_broadcaster_node")
        self.declare_parameter("scene_path", "")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("stride", 4)
        scene_path = self.get_parameter("scene_path").get_parameter_value().string_value
        rate_hz = self.get_parameter("rate_hz").get_parameter_value().double_value
        self.stride = self.get_parameter("stride").get_parameter_value().integer_value
        if not scene_path:
            raise RuntimeError("scene_path parameter is required")
        self.scene = load_scene(scene_path)
        self.frames = self.scene["frames"]
        if not self.frames:
            raise RuntimeError("scene.json contains no frames")
        self.color_info = self.scene["camera_info"]["color"]
        self.depth_info = self.scene["camera_info"]["depth"]
        self.depth_scale = float(
            self.scene["camera_info"]["depth"].get("depth_scale", DEFAULT_DEPTH_SCALE)
        )
        self.index = 0

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.color_pub = self.create_publisher(Image, "/camera/color/image_raw", qos)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", qos)
        self.color_info_pub = self.create_publisher(CameraInfo, "/camera/color/camera_info", qos)
        self.depth_info_pub = self.create_publisher(CameraInfo, "/camera/depth/camera_info", qos)
        self.cloud_pub = self.create_publisher(PointCloud2, "/camera/depth/points", qos)
        self.create_timer(1.0 / max(rate_hz, 0.1), self._tick)
        self.get_logger().info(
            f"rgbd_broadcaster ready: {len(self.frames)} frame(s), stride={self.stride}"
        )

    def _tick(self) -> None:
        frame = self.frames[self.index % len(self.frames)]
        stamp = self.get_clock().now().to_msg()

        rgb = load_rgb(frame["rgb"])
        depth = load_depth(frame["depth"], self.depth_scale)
        points = depth_to_points(depth, self.depth_info["K"], self.stride)

        self.color_pub.publish(make_image(stamp, self.color_info["frame_id"], rgb, "rgb8"))
        depth_raw = np.rint(depth / self.depth_scale).astype(np.uint16)
        self.depth_pub.publish(make_image(stamp, self.depth_info["frame_id"], depth_raw, "16UC1"))

        color_info = make_camera_info(
            self.color_info["frame_id"],
            self.color_info["width"],
            self.color_info["height"],
            self.color_info["K"],
        )
        color_info.header.stamp = stamp
        depth_info = make_camera_info(
            self.depth_info["frame_id"],
            self.depth_info["width"],
            self.depth_info["height"],
            self.depth_info["K"],
        )
        depth_info.header.stamp = stamp
        self.color_info_pub.publish(color_info)
        self.depth_info_pub.publish(depth_info)

        self.cloud_pub.publish(make_pointcloud2(stamp, self.depth_info["frame_id"], points))
        self.index += 1


def main(argv=None) -> int:
    rclpy.init(args=argv)
    try:
        node = RgbdBroadcaster()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
