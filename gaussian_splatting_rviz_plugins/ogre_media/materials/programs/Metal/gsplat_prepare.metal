#include <metal_stdlib>
using namespace metal;

constant uint kRadix = 256;
constant float kSHC0 = 0.28209479177387814;
constant float kSHC1 = 0.4886025119029199;
constant float kSHC2[5] = {
  1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
  -1.0925484305920792, 0.5462742152960396};
constant float kSHC3[7] = {
  -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
  0.3731763325901154, -0.4570457994644658, 1.445305721320277,
  -0.5900435899266435};

struct SplatRecord
{
  packed_float3 position;
  float opacity;
  packed_float3 scale;
  float pad0;
  float4 quat;
};

struct DrawInstance
{
  float index;
  packed_float3 colour;
};

struct ViewParameters
{
  float4 world_rows[4];
  float4 view_rows[4];
  float4 frustum_planes[6];
  float4 local_camera;
  float4 local_sort_direction;
  float world_scale;
  float focal_y;
  float near_clip;
  float far_clip;
  float ortho_scale;
  float sigma_radius;
  float min_screen_radius;
  float eps2d;
  uint count;
  uint viewport_height;
  uint sh_coefficients;
  uint flags;
};

struct RadixParameters
{
  uint count;
  uint tile_count;
  uint elements_per_tile;
  uint shift;
};

struct DrawIndexedArguments
{
  uint index_count;
  uint instance_count;
  uint index_start;
  int base_vertex;
  uint base_instance;
};

static float3 transformPoint(constant float4 * rows, float3 point)
{
  const float4 p = float4(point, 1.0);
  return float3(dot(rows[0], p), dot(rows[1], p), dot(rows[2], p));
}

static uint orderedFloatKey(float value)
{
  const uint bits = as_type<uint>(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

kernel void reset_view(
  device atomic_uint * visible_count [[buffer(0)]],
  device DrawIndexedArguments * arguments [[buffer(1)]],
  uint id [[thread_position_in_grid]])
{
  if (id != 0) {
    return;
  }
  atomic_store_explicit(visible_count, 0u, memory_order_relaxed);
  arguments->index_count = 6u;
  arguments->instance_count = 0u;
  arguments->index_start = 0u;
  arguments->base_vertex = 0;
  arguments->base_instance = 0u;
}

kernel void make_depth_keys(
  device const SplatRecord * splats [[buffer(0)]],
  device uint2 * records [[buffer(1)]],
  device atomic_uint * visible_count [[buffer(2)]],
  constant ViewParameters & p [[buffer(3)]],
  uint i [[thread_position_in_grid]])
{
  if (i >= p.count) {
    return;
  }

  const SplatRecord splat = splats[i];
  const float alpha_cutoff = 1.0 / 255.0;
  bool visible = splat.opacity >= alpha_cutoff;
  const float visible_radius = visible ? min(
    p.sigma_radius, sqrt(2.0 * log(splat.opacity / alpha_cutoff))) : 0.0;
  const float max_sigma = max(splat.scale.x, max(splat.scale.y, splat.scale.z));
  const float world_radius = visible_radius * max_sigma * p.world_scale;
  const float3 world_position = transformPoint(p.world_rows, splat.position);

  if (visible && (p.flags & 1u)) {
    for (uint plane = 0; plane < 6; ++plane) {
      if (dot(p.frustum_planes[plane].xyz, world_position) +
        p.frustum_planes[plane].w < -world_radius)
      {
        visible = false;
        break;
      }
    }
  }

  if (visible && (p.flags & 1u) && p.min_screen_radius > 0.0 && p.viewport_height > 0u) {
    float screen_radius = 0.0;
    if (p.flags & 2u) {
      const float depth = -transformPoint(p.view_rows, world_position).z;
      if (depth <= 0.0) {
        visible = false;
      } else {
        screen_radius = p.focal_y * world_radius / max(depth - world_radius, p.near_clip);
      }
    } else {
      screen_radius = p.ortho_scale * world_radius;
    }
    if (visible) {
      const float blur_radius = visible_radius * sqrt(max(p.eps2d, 0.0));
      screen_radius = sqrt(screen_radius * screen_radius + blur_radius * blur_radius);
      visible = screen_radius >= p.min_screen_radius;
    }
  }

  if (visible) {
    uint key = orderedFloatKey(dot(float3(splat.position), p.local_sort_direction.xyz));
    // UINT_MAX is reserved for rejected records, which the ascending radix
    // moves behind all visible records.
    key = min(key, 0xfffffffeu);
    records[i] = uint2(key, i);
    atomic_fetch_add_explicit(visible_count, 1u, memory_order_relaxed);
  } else {
    records[i] = uint2(0xffffffffu, i);
  }
}

kernel void radix_histogram(
  device const uint2 * records [[buffer(0)]],
  device uint * histograms [[buffer(1)]],
  constant RadixParameters & parameters [[buffer(2)]],
  uint tile [[threadgroup_position_in_grid]],
  uint lane [[thread_position_in_threadgroup]],
  uint thread_count [[threads_per_threadgroup]])
{
  threadgroup atomic_uint local_histogram[kRadix];
  for (uint digit = lane; digit < kRadix; digit += thread_count) {
    atomic_store_explicit(&local_histogram[digit], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint begin = tile * parameters.elements_per_tile;
  const uint end = min(begin + parameters.elements_per_tile, parameters.count);
  for (uint i = begin + lane; i < end; i += thread_count) {
    const uint digit = (records[i].x >> parameters.shift) & 0xffu;
    atomic_fetch_add_explicit(&local_histogram[digit], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint digit = lane; digit < kRadix; digit += thread_count) {
    histograms[digit * parameters.tile_count + tile] =
      atomic_load_explicit(&local_histogram[digit], memory_order_relaxed);
  }
}

kernel void radix_scan_offsets(
  device const uint * histograms [[buffer(0)]],
  device uint * offsets [[buffer(1)]],
  device uint * totals [[buffer(2)]],
  constant RadixParameters & parameters [[buffer(3)]],
  uint digit [[thread_position_in_grid]])
{
  if (digit >= kRadix) {
    return;
  }
  uint running = 0;
  const uint base = digit * parameters.tile_count;
  for (uint tile = 0; tile < parameters.tile_count; ++tile) {
    offsets[base + tile] = running;
    running += histograms[base + tile];
  }
  totals[digit] = running;
}

kernel void radix_scan_digit_bases(
  device const uint * totals [[buffer(0)]],
  device uint * digit_bases [[buffer(1)]],
  uint id [[thread_position_in_grid]])
{
  if (id != 0) {
    return;
  }
  uint running = 0;
  for (uint digit = 0; digit < kRadix; ++digit) {
    digit_bases[digit] = running;
    running += totals[digit];
  }
}

kernel void radix_scatter(
  device const uint2 * input [[buffer(0)]],
  device uint2 * output [[buffer(1)]],
  device const uint * offsets [[buffer(2)]],
  device const uint * digit_bases [[buffer(3)]],
  constant RadixParameters & parameters [[buffer(4)]],
  uint tile [[threadgroup_position_in_grid]],
  uint lane [[thread_position_in_threadgroup]],
  uint thread_count [[threads_per_threadgroup]])
{
  threadgroup uint cursor[kRadix];
  for (uint digit = lane; digit < kRadix; digit += thread_count) {
    cursor[digit] = digit_bases[digit] + offsets[digit * parameters.tile_count + tile];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint begin = tile * parameters.elements_per_tile;
  const uint end = min(begin + parameters.elements_per_tile, parameters.count);
  const uint tile_size = end > begin ? end - begin : 0;
  for (uint chunk = 0; chunk < parameters.elements_per_tile; chunk += thread_count) {
    const uint local = chunk + lane;
    const bool active = local < tile_size;
    const uint2 record = active ? input[begin + local] : uint2(0u);
    const uint digit = active ? ((record.x >> parameters.shift) & 0xffu) : 0u;

    uint peers = uint((simd_vote::vote_t)simd_ballot(active));
    for (uint bit_index = 0; bit_index < 8; ++bit_index) {
      const bool bit = (digit >> bit_index) & 1u;
      const uint ballot = uint((simd_vote::vote_t)simd_ballot(bit));
      peers &= bit ? ballot : ~ballot;
    }
    const uint rank = popcount(peers & ((1u << lane) - 1u));
    const uint peer_count = popcount(peers);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint base = active ? cursor[digit] : 0u;
    if (active) {
      output[base + rank] = record;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active && rank == 0) {
      cursor[digit] += peer_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

static float3 evaluateSH(
  uint splat, float3 position,
  device const float * dc, device const float * rest,
  constant ViewParameters & p)
{
  float3 colour = kSHC0 * float3(dc[splat * 3u], dc[splat * 3u + 1u], dc[splat * 3u + 2u]) + 0.5;
  const float3 direction = normalize(position - p.local_camera.xyz);
  const float x = direction.x;
  const float y = direction.y;
  const float z = direction.z;
  const uint stride = p.sh_coefficients * 3u;
  device const float * coefficients = rest + splat * stride;

  if (p.sh_coefficients >= 3u) {
    colour += -kSHC1 * y * float3(coefficients[0], coefficients[1], coefficients[2]);
    colour +=  kSHC1 * z * float3(coefficients[3], coefficients[4], coefficients[5]);
    colour += -kSHC1 * x * float3(coefficients[6], coefficients[7], coefficients[8]);
  }
  if (p.sh_coefficients >= 8u) {
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    colour += kSHC2[0] * xy * float3(coefficients[9], coefficients[10], coefficients[11]);
    colour += kSHC2[1] * yz * float3(coefficients[12], coefficients[13], coefficients[14]);
    colour += kSHC2[2] * (2.0 * zz - xx - yy) *
      float3(coefficients[15], coefficients[16], coefficients[17]);
    colour += kSHC2[3] * xz * float3(coefficients[18], coefficients[19], coefficients[20]);
    colour += kSHC2[4] * (xx - yy) *
      float3(coefficients[21], coefficients[22], coefficients[23]);
  }
  if (p.sh_coefficients >= 15u) {
    const float xx = x * x, yy = y * y, zz = z * z, xy = x * y;
    colour += kSHC3[0] * y * (3.0 * xx - yy) *
      float3(coefficients[24], coefficients[25], coefficients[26]);
    colour += kSHC3[1] * xy * z *
      float3(coefficients[27], coefficients[28], coefficients[29]);
    colour += kSHC3[2] * y * (4.0 * zz - xx - yy) *
      float3(coefficients[30], coefficients[31], coefficients[32]);
    colour += kSHC3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) *
      float3(coefficients[33], coefficients[34], coefficients[35]);
    colour += kSHC3[4] * x * (4.0 * zz - xx - yy) *
      float3(coefficients[36], coefficients[37], coefficients[38]);
    colour += kSHC3[5] * z * (xx - yy) *
      float3(coefficients[39], coefficients[40], coefficients[41]);
    colour += kSHC3[6] * x * (xx - 3.0 * yy) *
      float3(coefficients[42], coefficients[43], coefficients[44]);
  }
  return max(colour, 0.0);
}

kernel void gather_instances(
  device const uint2 * sorted [[buffer(0)]],
  device const SplatRecord * splats [[buffer(1)]],
  device const float * sh_dc [[buffer(2)]],
  device const float * sh_rest [[buffer(3)]],
  device const atomic_uint * visible_count [[buffer(4)]],
  device DrawInstance * instances [[buffer(5)]],
  device DrawIndexedArguments * arguments [[buffer(6)]],
  constant ViewParameters & p [[buffer(7)]],
  uint output [[thread_position_in_grid]])
{
  const uint count = atomic_load_explicit(visible_count, memory_order_relaxed);
  if (output == 0u) {
    arguments->instance_count = count;
  }
  if (output >= count) {
    return;
  }
  const uint splat = sorted[output].y;
  DrawInstance instance;
  instance.index = float(splat);
  instance.colour = evaluateSH(splat, float3(splats[splat].position), sh_dc, sh_rest, p);
  instances[output] = instance;
}
