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


def parse_ogre_gaussian_mesh(path, max_splats=None):
    try:
        from geometry_msgs.msg import Point32
        from std_msgs.msg import ColorRGBA
        from gaussian_splatting_msgs.msg import Gaussian
    except ModuleNotFoundError:
        class Point32:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        class ColorRGBA:
            def __init__(self, r=0.0, g=0.0, b=0.0, a=0.0):
                self.r = r
                self.g = g
                self.b = b
                self.a = a

        class Gaussian:
            pass

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

    count = min(vertex_count, max_splats) if max_splats else vertex_count
    splats = []
    for i in range(count):
        px, py, pz = read_half3(reader.data, element_offset(buffers, position_elem, i))
        r, g, b, a = read_colour(reader.data, element_offset(buffers, colour_elem, i))
        cov_xx, cov_yy, cov_zz = read_half3(
            reader.data, element_offset(buffers, cov_diag_elem, i))
        cov_xy, cov_xz, cov_yz = read_half3(
            reader.data, element_offset(buffers, cov_upper_elem, i))

        splat = Gaussian()
        splat.position = Point32(x=float(px), y=float(py), z=float(pz))
        splat.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))
        splat.cov_xx = float(cov_xx)
        splat.cov_yy = float(cov_yy)
        splat.cov_zz = float(cov_zz)
        splat.cov_xy = float(cov_xy)
        splat.cov_xz = float(cov_xz)
        splat.cov_yz = float(cov_yz)
        splats.append(splat)

    return splats, vertex_count


def create_publisher_node_class():
    import rclpy
    from rclpy.node import Node
    from gaussian_splatting_msgs.msg import GaussianSplats

    class OgreGaussianMeshPublisher(Node):
        def __init__(self, args):
            super().__init__("ogre_gaussian_mesh_publisher")
            self.publisher = self.create_publisher(GaussianSplats, args.topic, 1)
            self.frame_id = args.frame_id
            self.splats, source_count = parse_ogre_gaussian_mesh(args.mesh, args.max_splats)
            self.get_logger().info(
                f"loaded {len(self.splats)} / {source_count} splats from {args.mesh}")
            self.timer = self.create_timer(1.0 / args.rate, self.publish_once)

        def publish_once(self):
            msg = GaussianSplats()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.splats = self.splats
            self.publisher.publish(msg)
            self.get_logger().info(
                f"published {len(msg.splats)} splats",
                throttle_duration_sec=2.0)

    return rclpy, OgreGaussianMeshPublisher


class _Unused:
    def __init__(self, args):
        raise RuntimeError(args)


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
        splats, source_count = parse_ogre_gaussian_mesh(args.mesh, args.max_splats)
        print(f"loaded {len(splats)} / {source_count} splats from {args.mesh}")
        first = splats[0]
        print(
            "first splat:",
            first.position.x, first.position.y, first.position.z,
            first.color.r, first.color.g, first.color.b, first.color.a,
            first.cov_xx, first.cov_yy, first.cov_zz,
            first.cov_xy, first.cov_xz, first.cov_yz)
        return

    rclpy, OgreGaussianMeshPublisher = create_publisher_node_class()
    rclpy.init()
    node = OgreGaussianMeshPublisher(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
