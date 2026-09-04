#!/usr/bin/env python3

import argparse
import struct


M_MESH = 0x3000
M_SUBMESH = 0x4000
M_SUBMESH_OPERATION = 0x4010
M_GEOMETRY = 0x5000
M_GEOMETRY_VERTEX_DECLARATION = 0x5100
M_GEOMETRY_VERTEX_ELEMENT = 0x5110
M_GEOMETRY_VERTEX_BUFFER = 0x5200
M_GEOMETRY_VERTEX_BUFFER_DATA = 0x5210

OT_POINT_LIST = 1

VET_UBYTE4_NORM = 30
VET_HALF3 = 38

VES_POSITION = 1
VES_COLOUR = 5
VES_TEXTURE_COORDINATES = 7


class MeshReader:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()

    def u16(self, offset):
        return struct.unpack_from("<H", self.data, offset)[0]

    def u32(self, offset):
        return struct.unpack_from("<I", self.data, offset)[0]

    def chunk(self, offset):
        chunk_id = self.u16(offset)
        length = self.u32(offset + 2)
        body = offset + 6
        end = offset + length
        return chunk_id, length, body, end

    def string(self, offset):
        end = self.data.index(b"\n", offset)
        return self.data[offset:end].decode("utf-8"), end + 1


def read_half3(data, offset):
    return struct.unpack_from("<3e", data, offset)


def read_colour(data, offset):
    r, g, b, a = struct.unpack_from("<4B", data, offset)
    return r / 255.0, g / 255.0, b / 255.0, a / 255.0


def parse_geometry(reader, body, end):
    vertex_count = reader.u32(body)
    elements = []
    buffers = {}
    offset = body + 4

    while offset < end:
        chunk_id, _, chunk_body, chunk_end = reader.chunk(offset)

        if chunk_id == M_GEOMETRY_VERTEX_DECLARATION:
            elem_offset = chunk_body
            while elem_offset < chunk_end:
                elem_id, _, elem_body, elem_end = reader.chunk(elem_offset)
                if elem_id == M_GEOMETRY_VERTEX_ELEMENT:
                    source, elem_type, semantic, elem_byte_offset, index = struct.unpack_from(
                        "<5H", reader.data, elem_body)
                    elements.append({
                        "source": source,
                        "type": elem_type,
                        "semantic": semantic,
                        "offset": elem_byte_offset,
                        "index": index,
                    })
                elem_offset = elem_end

        elif chunk_id == M_GEOMETRY_VERTEX_BUFFER:
            bind_index, vertex_size = struct.unpack_from("<2H", reader.data, chunk_body)
            data_id, _, data_body, data_end = reader.chunk(chunk_body + 4)
            if data_id != M_GEOMETRY_VERTEX_BUFFER_DATA:
                raise RuntimeError("missing vertex buffer data chunk")
            buffers[bind_index] = {
                "vertex_size": vertex_size,
                "body": data_body,
                "end": data_end,
            }

        offset = chunk_end

    return vertex_count, elements, buffers


def find_element(elements, semantic, index=0):
    for element in elements:
        if element["semantic"] == semantic and element["index"] == index:
            return element
    raise RuntimeError(f"missing vertex element semantic={semantic} index={index}")


def element_offset(buffers, element, vertex_index):
    buffer = buffers[element["source"]]
    return buffer["body"] + vertex_index * buffer["vertex_size"] + element["offset"]


SH_C0 = 0.28209479177387814


def covariances_to_scales_quats(cov):
    """Recover linear scales and (x, y, z, w) quaternions from 3x3 covariances.

    The Ogre .mesh carries Sigma directly, but the message stores the
    definitional scale/rotation pair, so this inverts Sigma = R diag(s)^2 R^T.
    The decomposition is unique only up to axis permutation and sign, which
    describe the same ellipsoid.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    scales = np.sqrt(np.clip(eigenvalues, 0.0, None)).astype(np.float32)

    # eigh may return a left-handed basis; mirror one axis so it is a rotation.
    flip = np.linalg.det(eigenvectors) < 0.0
    eigenvectors[flip, :, 0] *= -1.0

    quats = Rotation.from_matrix(eigenvectors).as_quat().astype(np.float32)
    return scales, quats  # as_quat() is already (x, y, z, w)


def parse_ogre_gaussian_mesh(path, max_splats=None):
    reader = MeshReader(path)

    header_id = reader.u16(0)
    if header_id != 0x1000:
        raise RuntimeError("not an Ogre mesh file")

    version, offset = reader.string(2)
    if not version.startswith("[MeshSerializer"):
        raise RuntimeError(f"unsupported Ogre mesh header: {version}")

    mesh_id, _, mesh_body, mesh_end = reader.chunk(offset)
    if mesh_id != M_MESH:
        raise RuntimeError("mesh chunk not found")

    offset = mesh_body + 1  # skeletallyAnimated bool
    vertex_count = None
    elements = None
    buffers = None
    operation_type = None

    while offset < mesh_end:
        chunk_id, _, chunk_body, chunk_end = reader.chunk(offset)

        if chunk_id == M_SUBMESH:
            _, sub_offset = reader.string(chunk_body)
            use_shared_vertices = bool(reader.data[sub_offset])
            sub_offset += 1
            index_count = reader.u32(sub_offset)
            sub_offset += 4
            indexes_32_bit = bool(reader.data[sub_offset])
            sub_offset += 1
            sub_offset += index_count * (4 if indexes_32_bit else 2)

            while sub_offset < chunk_end:
                sub_id, _, sub_body, sub_end = reader.chunk(sub_offset)
                if sub_id == M_GEOMETRY and not use_shared_vertices:
                    vertex_count, elements, buffers = parse_geometry(reader, sub_body, sub_end)
                elif sub_id == M_SUBMESH_OPERATION:
                    operation_type = reader.u16(sub_body)
                sub_offset = sub_end

        elif chunk_id == M_GEOMETRY:
            vertex_count, elements, buffers = parse_geometry(reader, chunk_body, chunk_end)

        offset = chunk_end

    if operation_type != OT_POINT_LIST:
        raise RuntimeError(f"expected point_list mesh, got operation type {operation_type}")
    if vertex_count is None or elements is None or buffers is None:
        raise RuntimeError("no geometry found")

    position_elem = find_element(elements, VES_POSITION, 0)
    colour_elem = find_element(elements, VES_COLOUR, 0)
    cov_diag_elem = find_element(elements, VES_TEXTURE_COORDINATES, 0)
    cov_upper_elem = find_element(elements, VES_TEXTURE_COORDINATES, 1)

    expected = [
        (position_elem, VET_HALF3, "position"),
        (colour_elem, VET_UBYTE4_NORM, "colour"),
        (cov_diag_elem, VET_HALF3, "cov_diag"),
        (cov_upper_elem, VET_HALF3, "cov_upper"),
    ]
    for element, elem_type, name in expected:
        if element["type"] != elem_type:
            raise RuntimeError(f"unexpected {name} element type {element['type']}")

    import numpy as np

    count = min(vertex_count, max_splats) if max_splats else vertex_count

    means = np.empty((count, 3), np.float32)
    rgba = np.empty((count, 4), np.float32)
    cov = np.empty((count, 3, 3), np.float64)
    for i in range(count):
        means[i] = read_half3(reader.data, element_offset(buffers, position_elem, i))
        rgba[i] = read_colour(reader.data, element_offset(buffers, colour_elem, i))
        xx, yy, zz = read_half3(reader.data, element_offset(buffers, cov_diag_elem, i))
        xy, xz, yz = read_half3(reader.data, element_offset(buffers, cov_upper_elem, i))
        cov[i, 0, 0], cov[i, 1, 1], cov[i, 2, 2] = xx, yy, zz
        cov[i, 0, 1] = cov[i, 1, 0] = xy
        cov[i, 0, 2] = cov[i, 2, 0] = xz
        cov[i, 1, 2] = cov[i, 2, 1] = yz

    scales, quats = covariances_to_scales_quats(cov)
    return means, rgba, scales, quats, vertex_count


def build_message(means, rgba, scales, quats):
    from gaussian_splatting_msgs.msg import GaussianSplats

    msg = GaussianSplats()
    msg.type = GaussianSplats.TYPE_3DGS
    msg.rasterize_mode = GaussianSplats.RASTERIZE_MODE_CLASSIC
    msg.eps2d = 0.3
    msg.sh_degree = 0

    msg.means = means.reshape(-1).tolist()
    msg.scales = scales.reshape(-1).tolist()
    msg.quats = quats.reshape(-1).tolist()
    msg.opacities = rgba[:, 3].tolist()
    # The mesh stores RGB, so invert the degree 0 SH mapping rgb = sh*C0 + 0.5.
    msg.sh_dc = ((rgba[:, :3] - 0.5) / SH_C0).reshape(-1).tolist()
    msg.sh_rest = []
    return msg


def create_publisher_node_class():
    import rclpy
    from rclpy.node import Node
    from gaussian_splatting_msgs.msg import GaussianSplats

    class OgreGaussianMeshPublisher(Node):
        def __init__(self, args):
            super().__init__("ogre_gaussian_mesh_publisher")
            self.publisher = self.create_publisher(GaussianSplats, args.topic, 1)
            means, rgba, scales, quats, source_count = parse_ogre_gaussian_mesh(
                args.mesh, args.max_splats)
            self.msg = build_message(means, rgba, scales, quats)
            self.msg.header.frame_id = args.frame_id
            self.count = len(means)
            self.get_logger().info(
                f"loaded {self.count} / {source_count} splats from {args.mesh}")
            self.timer = self.create_timer(1.0 / args.rate, self.publish_once)

        def publish_once(self):
            self.msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(self.msg)
            self.get_logger().info(
                f"published {self.count} splats",
                throttle_duration_sec=2.0)

    return rclpy, OgreGaussianMeshPublisher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("--topic", default="/gaussian_splats")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--max-splats", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        means, rgba, scales, quats, source_count = parse_ogre_gaussian_mesh(
            args.mesh, args.max_splats)
        print(f"loaded {len(means)} / {source_count} splats from {args.mesh}")
        print("first mean  :", means[0])
        print("first rgba  :", rgba[0])
        print("first scale :", scales[0], "(linear sigma, metres)")
        print("first quat  :", quats[0], "(x, y, z, w)")
        return

    rclpy, OgreGaussianMeshPublisher = create_publisher_node_class()
    rclpy.init()
    node = OgreGaussianMeshPublisher(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
