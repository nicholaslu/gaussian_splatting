#!/usr/bin/env python3
"""Publish a 3DGS .ply file as a gaussian_splatting_msgs/GaussianSplats message.

The message carries the definitional form of the parameters (linear scales,
linear opacity, normalised quaternions in ROS order), so this converts out of
the PLY's optimiser parameterisation on the way.

The splats are published unrotated in --frame-id. Reconstructions are rarely
in a REP-103 frame -- 3DGS scenes are typically Y-up, and COLMAP leaves the
world orientation to the SfM gauge, so there is usually a residual tilt too.

Rotating the points here would be wrong on its own: the spherical harmonics
are expressed in the same frame and would have to be rotated with them
(Wigner D matrices for degree >= 1). Expressing the correction as a transform
avoids that entirely, because a renderer evaluates SH along the view direction
taken into the node's own frame. So --parent-frame publishes a static
transform instead of touching the data:

    # measure the ground plane and lay it on the parent's XY plane at z = 0
    publish_gaussian_ply.py scene.ply --frame-id splats --parent-frame map \\
        --align-ground

    # or state the pose yourself: rotation in radians, translation in metres
    publish_gaussian_ply.py scene.ply --frame-id splats --parent-frame map \\
        --rpy 1.5708 0 0 --xyz 0 0 -1.2
"""

import argparse

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_ros import StaticTransformBroadcaster

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
    return msg, means, opacities


def estimate_ground_transform(
        means, opacities, up_hint=None, max_tilt_deg=45.0,
        samples=200000, tolerance=None, seed=0):
    """Rotation and height laying the scene's ground plane on the parent's XY plane.

    3DGS scenes usually rest on one large surface, so the plane with the most
    inliers is the ground. RANSAC rather than PCA because floaters and the
    scene's own vertical extent skew a covariance fit.

    The largest plane is not always the ground, though, and geometry alone
    cannot say which way is up: indoors the ceiling is as flat as the floor and
    often larger, and a room with both has all its mass on one side of either.
    So up_hint, in the PLY's own frame, breaks the tie when the caller knows
    roughly which way is up: it fixes the sign and rejects candidates tilted
    more than max_tilt_deg away from it, which is also what keeps a large wall
    from winning. It is off by default because the tilt gate needs a hint that
    is actually right -- a wrong one costs more than no hint at all. Without it
    the largest plane of any orientation wins, oriented so the bulk of the
    scene is above it.

    Returns (rotation, offset_z, tilt_deg, inlier_fraction), where offset_z is
    the translation to put on the static transform so the plane lands at z = 0,
    and tilt_deg is the angle between the fitted normal and up_hint.
    """
    rng = np.random.default_rng(seed)
    points = means[opacities > 0.3]
    if len(points) < 3:
        points = means
    if len(points) > samples:
        points = points[rng.choice(len(points), samples, replace=False)]

    if tolerance is None:
        # Relative to the scene, or the same absolute slack is meaninglessly
        # tight outdoors and uselessly loose on a tabletop. Percentiles rather
        # than the bounding box because a few floaters can be kilometres out.
        span = np.percentile(points, 99, axis=0) - np.percentile(points, 1, axis=0)
        tolerance = 0.01 * float(np.linalg.norm(span))

    hint = None
    if up_hint is not None:
        hint = np.asarray(up_hint, dtype=np.float64)
        norm = np.linalg.norm(hint)
        hint = hint / norm if norm > 1e-9 else None
    min_cos = np.cos(np.radians(max_tilt_deg))
    centroid = points.mean(0)

    # Two passes: prefer a candidate that agrees with the hint, but keep the
    # best unconstrained fit too, so an unexpected gauge degrades to the old
    # behaviour instead of to nothing. The caller sees the tilt either way.
    best_inliers, best = 0, None
    best_any_inliers, best_any = 0, None
    for _ in range(3000):
        a, b, c = points[rng.integers(0, len(points), 3)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal / length
        inliers = int((np.abs((points - a) @ normal) < tolerance).sum())
        if inliers > best_any_inliers:
            best_any_inliers, best_any = inliers, (normal, a)
        if hint is not None:
            if abs(normal @ hint) < min_cos:
                continue
            if normal @ hint < 0:
                normal = -normal
        elif normal @ (centroid - a) < 0:
            normal = -normal  # face the bulk of the scene, so "up" is up
        if inliers > best_inliers:
            best_inliers, best = inliers, (normal, a)

    if best is None:
        best_inliers, best = best_any_inliers, best_any
        if best is None:
            return np.eye(3, dtype=np.float64), 0.0, 0.0, 0.0
        normal, a = best
        if normal @ (centroid - a) < 0:
            best = (-normal, a)

    normal, point = best
    tilt = float(np.degrees(np.arccos(np.clip(normal @ hint, -1.0, 1.0)))) \
        if hint is not None else 0.0

    axis = np.cross(normal, [0.0, 0.0, 1.0])
    sin = np.linalg.norm(axis)
    cos = float(normal @ [0.0, 0.0, 1.0])
    if sin < 1e-9:
        rotation = np.eye(3) if cos > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + skew + skew @ skew * ((1 - cos) / (sin * sin))

    # Rotating alone leaves the ground at whatever height the reconstruction
    # put it, so measure that height too. The RANSAC plane is pinned to three
    # random samples, so take it from every inlier rather than from those
    # three; median because the tolerance band is symmetric but its contents
    # are not, a floor having objects on one side only.
    heights = points @ rotation[2]
    inliers = heights[np.abs(heights - rotation[2] @ point) < tolerance]
    height = float(np.median(inliers)) if len(inliers) else float(rotation[2] @ point)
    return rotation, -height, tilt, best_inliers / len(points)


class GaussianPlyPublisher(Node):
    def __init__(self, args):
        super().__init__("gaussian_ply_publisher")
        self.publisher = self.create_publisher(GaussianSplats, args.topic, 1)
        self.msg, self.means, self.opacities = load_3dgs_ply(
            args.ply,
            max_splats=args.max_splats,
            stride=args.stride,
            scale_mult=args.scale_mult,
        )
        self.msg.header.frame_id = args.frame_id

        count = len(self.msg.means) // 3
        self.get_logger().info(
            f"loaded {count} splats from {args.ply} (sh_degree={self.msg.sh_degree})")

        if args.parent_frame:
            self.publish_static_transform(args)
        elif args.align_ground or any(args.rpy) or any(args.xyz):
            self.get_logger().warning(
                "--align-ground/--rpy do nothing without --parent-frame: there is no "
                "transform to put the rotation on. Add e.g. --frame-id splats "
                "--parent-frame map.")
        self.timer = self.create_timer(1.0 / args.rate, self.publish_once)

    def publish_static_transform(self, args):
        if args.align_ground:
            rotation, offset_z, tilt, inlier_fraction = estimate_ground_transform(
                self.means, self.opacities,
                up_hint=args.ground_up if any(args.ground_up) else None)
            quat = Rotation.from_matrix(rotation).as_quat()
            rpy = np.degrees(Rotation.from_matrix(rotation).as_euler("xyz"))
            translation = [0.0, 0.0, offset_z]
            self.get_logger().info(
                f"ground plane fit: {100 * inlier_fraction:.0f}% inliers, "
                f"rpy {rpy.round(1)} deg, z {offset_z:+.3f}, "
                f"{tilt:.0f} deg off the up hint")
            if inlier_fraction < 0.10:
                self.get_logger().warning(
                    "few inliers: the scene may have no dominant plane. Check the "
                    "result and fall back to --rpy if it looks wrong.")
            if tilt > 45.0:
                self.get_logger().warning(
                    "no plane matched --ground-up, so the largest plane of any "
                    "orientation was used. If the scene comes out on its side or "
                    "upside down, give the reconstruction's real up axis.")
        else:
            quat = Rotation.from_euler("xyz", args.rpy).as_quat()
            translation = args.xyz

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = args.parent_frame
        transform.child_frame_id = args.frame_id
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])
        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.static_broadcaster.sendTransform(transform)
        self.get_logger().info(
            f"published static transform {args.parent_frame} -> {args.frame_id}")

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
    parser.add_argument(
        "--max-splats", type=int, default=0,
        help="Keep at most this many splats. 0 (the default) keeps the whole file; "
             "a non-zero value truncates, which is useful for building a scaling series.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--parent-frame", default=None,
        help="Publish a static transform from this frame to --frame-id. Without it "
             "no transform is published and the splats land unrotated.")
    parser.add_argument(
        "--rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("R", "P", "Y"),
        help="Rotation for that static transform, in radians. 3DGS scenes are "
             "usually Y-up, which is --rpy 1.5708 0 0.")
    parser.add_argument(
        "--xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
        help="Translation for that static transform, in metres, applied after "
             "--rpy. --align-ground measures it instead.")
    parser.add_argument(
        "--ground-up", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
        help="Rough up direction in the PLY's own frame, to help --align-ground tell "
             "the floor from the ceiling and reject walls. Indoor scenes need it: a "
             "ceiling is as flat as a floor and often larger, and no amount of "
             "geometry says which is which. COLMAP-derived scenes are usually Y-down, "
             "so --ground-up 0 -1 0. Off by default, since a wrong hint fits a worse "
             "plane than no hint.")
    parser.add_argument(
        "--align-ground", action="store_true",
        help="Fit the scene's dominant plane and use the rotation and height that "
             "lay it flat at z = 0, instead of --rpy and --xyz. Also corrects the "
             "reconstruction's tilt.")
    parser.add_argument("--scale-mult", type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = GaussianPlyPublisher(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
