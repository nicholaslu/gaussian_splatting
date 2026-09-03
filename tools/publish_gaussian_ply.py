#!/usr/bin/env python3

import argparse
import math
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA

from gaussian_splatting_msgs.msg import Gaussian, GaussianSplats


SH_C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def quat_to_rotmat(q):
    # GraphDECO stores rot as w, x, y, z
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / n, x / n, y / n, z / n

    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float32)


def compute_covariance(scale, rot):
    r = quat_to_rotmat(rot)
    s = np.diag(scale)
    m = r @ s
    cov = m @ m.T
    return cov


def read_3dgs_ply(path, max_splats=None, stride=1, scale_mult=1.0):
    from gsply import plyread

    data = plyread(path)

    # Convert PLY log-scales/logit-opacities to linear values,
    # and SH DC coefficients to RGB.
    data.denormalize()
    data.to_rgb()

    means = data.means
    scales = data.scales
    quats = data.quats
    opacities = data.opacities
    colors = data.sh0

    print("scales min/max", scales.min(), scales.max())
    print("opacities min/max", opacities.min(), opacities.max())

    count = means.shape[0]
    selected = np.arange(0, count, stride)
    if max_splats is not None:
        selected = selected[:max_splats]

    splats = []
    for i in selected:
        xyz = means[i]
        scale = scales[i] * scale_mult
        rot = quats[i]
        cov = compute_covariance(scale, rot)

        rgb = np.clip(colors[i], 0.0, 1.0)
        alpha = float(np.clip(opacities[i], 0.0, 1.0))

        g = Gaussian()
        g.position = Point32(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        g.color = ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=alpha)

        g.cov_xx = float(cov[0, 0])
        g.cov_yy = float(cov[1, 1])
        g.cov_zz = float(cov[2, 2])
        g.cov_xy = float(cov[0, 1])
        g.cov_xz = float(cov[0, 2])
        g.cov_yz = float(cov[1, 2])
        splats.append(g)

    return splats


class GaussianPlyPublisher(Node):
    def __init__(self, args):
        super().__init__("gaussian_ply_publisher")
        self.publisher = self.create_publisher(GaussianSplats, args.topic, 1)
        self.frame_id = args.frame_id
        self.splats = read_3dgs_ply(
            args.ply,
            max_splats=args.max_splats,
            stride=args.stride,
            scale_mult=args.scale_mult,
        )

        self.get_logger().info(f"loaded {len(self.splats)} splats from {args.ply}")
        self.timer = self.create_timer(1.0 / args.rate, self.publish_once)

    def publish_once(self):
        msg = GaussianSplats()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.splats = self.splats
        self.publisher.publish(msg)
        self.get_logger().info(f"published {len(msg.splats)} splats", throttle_duration_sec=2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("--topic", default="/gaussian_splats")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--max-splats", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--scale-mult", type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = GaussianPlyPublisher(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
