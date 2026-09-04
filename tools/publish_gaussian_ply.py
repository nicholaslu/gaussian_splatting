#!/usr/bin/env python3
"""Publish a 3DGS .ply file as a gaussian_splatting_msgs/GaussianSplats message.

The message carries the definitional form of the parameters (linear scales,
linear opacity, normalised quaternions in ROS order), so this converts out of
the PLY's optimiser parameterisation on the way.

The data is published unrotated in --frame-id. COLMAP-derived reconstructions
are not in a REP-103 frame, but rotating the points here would also require
rotating the spherical harmonics (Wigner D matrices for degree >= 1), so the
axis change belongs in TF instead:

    ros2 run tf2_ros static_transform_publisher \\
        --frame-id map --child-frame-id splats --roll -1.5708
"""

import argparse

import numpy as np
import rclpy
from rclpy.node import Node

from gaussian_splatting_msgs.msg import GaussianSplats


def load_3dgs_ply(path, max_splats=None, stride=1, scale_mult=1.0):
    from gsply import plyread

    data = plyread(path)

    # PLY stores log-scales and logit-opacities; denormalize() applies exp()
    # and sigmoid(). to_rgb() is deliberately NOT called: the message carries
    # raw SH coefficients, not RGB.
    data.denormalize()

    means = np.asarray(data.means, dtype=np.float32)
    scales = np.asarray(data.scales, dtype=np.float32)
    quats = np.asarray(data.quats, dtype=np.float32)
    opacities = np.asarray(data.opacities, dtype=np.float32).reshape(-1)
    sh0 = np.asarray(data.sh0, dtype=np.float32)
    shN = np.asarray(data.shN, dtype=np.float32)
    sh_degree = int(data.get_sh_degree())

    if stride > 1:
        sel = slice(None, None, stride)
        means, scales, quats = means[sel], scales[sel], quats[sel]
        opacities, sh0, shN = opacities[sel], sh0[sel], shN[sel]
    if max_splats:
        n = min(len(means), max_splats)
        means, scales, quats = means[:n], scales[:n], quats[:n]
        opacities, sh0, shN = opacities[:n], sh0[:n], shN[:n]

    # gsply returns normalised quaternions in PLY order (w, x, y, z); the
    # message uses the ROS component order (x, y, z, w).
    quats = np.roll(quats, -1, axis=1)

    msg = GaussianSplats()
    msg.type = GaussianSplats.TYPE_3DGS
    msg.rasterize_mode = GaussianSplats.RASTERIZE_MODE_CLASSIC
    msg.eps2d = 0.3
    msg.sh_degree = sh_degree

    msg.means = (means * scale_mult).reshape(-1).tolist()
    msg.scales = (scales * scale_mult).reshape(-1).tolist()
    msg.quats = quats.reshape(-1).tolist()
    msg.opacities = opacities.tolist()
    msg.sh_dc = sh0.reshape(-1).tolist()
    # gsply already returns shN as (N, K-1, 3), which is the coefficient-major
    # layout the message wants, so no transpose is needed here.
    msg.sh_rest = shN.reshape(-1).tolist()
    return msg


class GaussianPlyPublisher(Node):
    def __init__(self, args):
        super().__init__("gaussian_ply_publisher")
        self.publisher = self.create_publisher(GaussianSplats, args.topic, 1)
        self.msg = load_3dgs_ply(
            args.ply,
            max_splats=args.max_splats,
            stride=args.stride,
            scale_mult=args.scale_mult,
        )
        self.msg.header.frame_id = args.frame_id

        count = len(self.msg.means) // 3
        self.get_logger().info(
            f"loaded {count} splats from {args.ply} (sh_degree={self.msg.sh_degree})")
        self.timer = self.create_timer(1.0 / args.rate, self.publish_once)

    def publish_once(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.msg)
        self.get_logger().info(
            f"published {len(self.msg.means) // 3} splats", throttle_duration_sec=2.0)


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
