// GPU view preparation for the Gaussian splat display: cull, compact, sort and
// gather, encoded into the frame's command buffer with no CPU readback.
//
// The per-splat arithmetic mirrors prepareSplat() in splat_view.hpp and the
// structs mirror that header's layouts; test/verify_gpu_preparation.mm checks
// the two against each other.
//
// The tiled radix sort - radix_histogram, radix_scan_offsets,
// radix_scan_digit_bases and radix_scatter - is adapted from the GPU sorter in
// MetalSprocketsGaussianSplats (https://github.com/schwa/MetalSprocketsGaussianSplats),
// copyright (c) 2025 Jonathan Wight, and compact_scatter reuses its lane-ranked
// scatter. MIT licensed; see THIRD_PARTY_NOTICES.md in this package.

#include <metal_stdlib>
using namespace metal;

constant uint kRadix = 256;

// The scans run 256 threads wide and walk a row 256 entries at a time. Each
// thread looping over a whole row by itself, as these kernels once did, cost
// 4-5 ms of a 5-7 ms sort in the OpenGL equivalent on an RTX 4090.
constant uint kScanWidth = 256;

// Inclusive prefix sum of `value` across a threadgroup of kScanWidth threads
// (Hillis-Steele). Every thread has to call it, the same number of times.
static uint threadgroupInclusiveScan(uint value, uint lane, threadgroup uint * scratch)
{
  scratch[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = 1; offset < kScanWidth; offset <<= 1) {
    const uint add = lane >= offset ? scratch[lane - offset] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[lane] += add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint result = scratch[lane];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}
// Marks a rejected splat in the index half of its record. Keys stay exact, and
// no splat index can reach this value.
constant uint kRejected = 0xffffffffu;
// Ogre's FRUSTUM_PLANE_FAR.
constant uint kFrustumPlaneFar = 1;
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

// Mirrors ProjectionParameters and ProjectedInstance in splat_view.hpp.
struct ProjectionParameters
{
  float4 worldview_rows[4];
  float4 worldviewproj_rows[4];
  float4 viewport_size;
  float fovy;
  float eps2d;
  float antialiased;
  float sigma_radius;
};

struct ProjectedInstance
{
  packed_float3 centre;
  float visible_radius;
  float4 axes;
  float4 colour;
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

// Written once per preparation, by compact_scan, and read by every stage after
// it: the sort and the gather size themselves to the survivors without the CPU
// ever learning how many there are.
struct PreparationState
{
  uint visible;
  uint visible_tiles;
  uint gather_groups;
  uint reserved;
};

struct TileParameters
{
  uint count;              // splats the buffers were sized for
  uint tile_count;         // tiles in that capacity; the stride of the radix tables
  uint elements_per_tile;
  uint shift;              // the radix digit being sorted on
};

struct ScanParameters
{
  uint tile_count;
  uint elements_per_tile;
  uint gather_width;
  uint index_count;
};

struct DrawIndexedArguments
{
  uint index_count;
  uint instance_count;
  uint index_start;
  int base_vertex;
  uint base_instance;
};

// MTLDispatchThreadgroupsIndirectArguments.
struct DispatchArguments
{
  uint threadgroups[3];
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

// Stage 1: a depth key per splat, or the rejection mark, written in place so
// the buffer stays in index order for the compaction to preserve.
kernel void make_depth_keys(
  device const SplatRecord * splats [[buffer(0)]],
  device uint2 * records [[buffer(1)]],
  constant ViewParameters & p [[buffer(2)]],
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
      if (plane == kFrustumPlaneFar && p.far_clip == 0.0) {
        continue;
      }
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

  records[i] = visible ?
    uint2(orderedFloatKey(dot(float3(splat.position), p.local_sort_direction.xyz)), i) :
    uint2(0u, kRejected);
}

// Stage 2a: survivors per tile.
kernel void compact_count(
  device const uint2 * records [[buffer(0)]],
  device uint * tile_counts [[buffer(1)]],
  constant TileParameters & parameters [[buffer(2)]],
  uint tile [[thread_position_in_grid]])
{
  if (tile >= parameters.tile_count) {
    return;
  }
  const uint begin = tile * parameters.elements_per_tile;
  const uint end = min(begin + parameters.elements_per_tile, parameters.count);
  uint survivors = 0;
  for (uint i = begin; i < end; ++i) {
    survivors += records[i].y != kRejected ? 1u : 0u;
  }
  tile_counts[tile] = survivors;
}

// Stage 2b: where each tile's survivors start in the compacted buffer, and
// every count the later stages need - the visible total, how many tiles and
// gather threadgroups it spans, and the arguments Ogre's indirect draw reads.
// One thread; a scan over a few thousand tiles.
kernel void compact_scan(
  device const uint * tile_counts [[buffer(0)]],
  device uint * tile_offsets [[buffer(1)]],
  device PreparationState * state [[buffer(2)]],
  device DispatchArguments * tile_dispatch [[buffer(3)]],
  device DispatchArguments * gather_dispatch [[buffer(4)]],
  device DrawIndexedArguments * draw [[buffer(5)]],
  constant ScanParameters & parameters [[buffer(6)]],
  uint lane [[thread_index_in_threadgroup]])
{
  threadgroup uint scratch[kScanWidth];
  threadgroup uint carry;
  if (lane == 0) {
    carry = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint start = 0; start < parameters.tile_count; start += kScanWidth) {
    const uint tile = start + lane;
    const uint count = tile < parameters.tile_count ? tile_counts[tile] : 0u;
    const uint inclusive = threadgroupInclusiveScan(count, lane, scratch);
    if (tile < parameters.tile_count) {
      tile_offsets[tile] = carry + inclusive - count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == kScanWidth - 1) {
      carry += inclusive;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane != 0) {
    return;
  }
  const uint visible = carry;
  const uint visible_tiles =
    (visible + parameters.elements_per_tile - 1u) / parameters.elements_per_tile;
  const uint gather_groups = (visible + parameters.gather_width - 1u) / parameters.gather_width;

  state->visible = visible;
  state->visible_tiles = visible_tiles;
  state->gather_groups = gather_groups;
  state->reserved = 0u;

  // At least one threadgroup even when nothing is visible, so that no indirect
  // dispatch is empty; every kernel bounds itself by state->visible.
  tile_dispatch->threadgroups[0] = max(visible_tiles, 1u);
  tile_dispatch->threadgroups[1] = 1u;
  tile_dispatch->threadgroups[2] = 1u;
  gather_dispatch->threadgroups[0] = max(gather_groups, 1u);
  gather_dispatch->threadgroups[1] = 1u;
  gather_dispatch->threadgroups[2] = 1u;

  draw->index_count = parameters.index_count;
  draw->instance_count = visible;
  draw->index_start = 0u;
  draw->base_vertex = 0;
  draw->base_instance = 0u;
}

// Stage 2c: moves each tile's survivors, in index order, to the tile's offset
// in the compacted buffer. Lanes of a threadgroup rank themselves with
// simd_ballot exactly as radix_scatter does, with a single bucket.
kernel void compact_scatter(
  device const uint2 * records [[buffer(0)]],
  device uint2 * compacted [[buffer(1)]],
  device const uint * tile_offsets [[buffer(2)]],
  constant TileParameters & parameters [[buffer(3)]],
  uint tile [[threadgroup_position_in_grid]],
  uint lane [[thread_position_in_threadgroup]],
  uint thread_count [[threads_per_threadgroup]])
{
  threadgroup uint cursor[1];
  if (lane == 0) {
    cursor[0] = tile_offsets[tile];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint begin = tile * parameters.elements_per_tile;
  const uint end = min(begin + parameters.elements_per_tile, parameters.count);
  const uint tile_size = end > begin ? end - begin : 0;
  for (uint chunk = 0; chunk < parameters.elements_per_tile; chunk += thread_count) {
    const uint local = chunk + lane;
    const uint2 record = local < tile_size ? records[begin + local] : uint2(0u, kRejected);
    const bool active = record.y != kRejected;

    const uint peers = uint((simd_vote::vote_t)simd_ballot(active));
    const uint rank = popcount(peers & ((1u << lane) - 1u));
    const uint peer_count = popcount(peers);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint base = cursor[0];
    if (active) {
      compacted[base + rank] = record;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active && rank == 0) {
      cursor[0] += peer_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Stage 3: stable LSD radix over the compacted survivors only.
kernel void radix_histogram(
  device const uint2 * records [[buffer(0)]],
  device uint * histograms [[buffer(1)]],
  constant TileParameters & parameters [[buffer(2)]],
  device const PreparationState & state [[buffer(3)]],
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
  const uint end = min(begin + parameters.elements_per_tile, state.visible);
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
  constant TileParameters & parameters [[buffer(3)]],
  device const PreparationState & state [[buffer(4)]],
  uint digit [[threadgroup_position_in_grid]],
  uint lane [[thread_index_in_threadgroup]])
{
  // One threadgroup per digit, scanning that digit's count in every tile.
  threadgroup uint scratch[kScanWidth];
  threadgroup uint carry;
  if (lane == 0) {
    carry = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint base = digit * parameters.tile_count;
  const uint tiles = state.visible_tiles;
  for (uint start = 0; start < tiles; start += kScanWidth) {
    const uint tile = start + lane;
    const uint count = tile < tiles ? histograms[base + tile] : 0u;
    const uint inclusive = threadgroupInclusiveScan(count, lane, scratch);
    if (tile < tiles) {
      offsets[base + tile] = carry + inclusive - count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == kScanWidth - 1) {
      carry += inclusive;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    totals[digit] = carry;
  }
}

kernel void radix_scan_digit_bases(
  device const uint * totals [[buffer(0)]],
  device uint * digit_bases [[buffer(1)]],
  uint digit [[thread_index_in_threadgroup]])
{
  threadgroup uint scratch[kScanWidth];
  const uint total = totals[digit];
  digit_bases[digit] = threadgroupInclusiveScan(total, digit, scratch) - total;
}

kernel void radix_scatter(
  device const uint2 * input [[buffer(0)]],
  device uint2 * output [[buffer(1)]],
  device const uint * offsets [[buffer(2)]],
  device const uint * digit_bases [[buffer(3)]],
  constant TileParameters & parameters [[buffer(4)]],
  device const PreparationState & state [[buffer(5)]],
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
  const uint end = min(begin + parameters.elements_per_tile, state.visible);
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
  // Vector3::normalise(), guard included, as in prepareSplat(): a splat at the
  // camera keeps a zero direction instead of a NaN one.
  float3 direction = position - p.local_camera.xyz;
  const float length = sqrt(dot(direction, direction));
  if (length > 0.0) {
    direction *= 1.0 / length;
  }
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

// gsplat_vp from gsplat.metal, less the corner displacement, done once per
// splat here instead of once per quad vertex; projectSplat() in splat_view.hpp
// is the reference it is checked against.
static ProjectedInstance projectSplat(
  SplatRecord splat, float3 colour, constant ProjectionParameters & q)
{
  ProjectedInstance out;
  out.centre = packed_float3(0.0, 0.0, 2.0);
  out.visible_radius = 0.0;
  out.axes = float4(0.0);
  out.colour = float4(0.0);

  const float4 p = float4(float3(splat.position), 1.0);
  const float clip_w = dot(q.worldviewproj_rows[3], p);
  if (clip_w <= 0.0) {
    return out;
  }

  const float x = splat.quat.x;
  const float y = splat.quat.y;
  const float z = splat.quat.z;
  const float w = splat.quat.w;
  const float3x3 rotation = float3x3(
    float3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
    float3(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
    float3(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
  const float3 s = float3(splat.scale);
  const float3x3 m = float3x3(rotation[0] * s.x, rotation[1] * s.y, rotation[2] * s.z);
  const float3x3 sigma = m * transpose(m);

  const float tanFovy = tan(q.fovy * 0.5);
  const float tanFovx = tanFovy * q.viewport_size.x / q.viewport_size.y;
  const float focalY = q.viewport_size.y / (2.0 * tanFovy);
  const float focalX = q.viewport_size.x / (2.0 * tanFovx);
  float3 t = float3(
    dot(q.worldview_rows[0], p), dot(q.worldview_rows[1], p), dot(q.worldview_rows[2], p));
  const float limx = 1.3 * tanFovx;
  const float limy = 1.3 * tanFovy;
  t.x = clamp(t.x / t.z, -limx, limx) * t.z;
  t.y = clamp(t.y / t.z, -limy, limy) * t.z;
  const float3 t0 = (focalX / t.z) * q.worldview_rows[0].xyz +
    (-(focalX * t.x) / (t.z * t.z)) * q.worldview_rows[2].xyz;
  const float3 t1 = (focalY / t.z) * q.worldview_rows[1].xyz +
    (-(focalY * t.y) / (t.z * t.z)) * q.worldview_rows[2].xyz;
  float a = dot(t0, sigma * t0);
  const float b = dot(t0, sigma * t1);
  float c = dot(t1, sigma * t1);

  const float detOriginal = a * c - b * b;
  a += q.eps2d;
  c += q.eps2d;
  const float detBlurred = a * c - b * b;
  const float compensation = sqrt(max(detOriginal / detBlurred, 0.0));
  const float alpha = splat.opacity * mix(1.0, compensation, q.antialiased);
  const float alphaCutoff = 1.0 / 255.0;
  if (alpha < alphaCutoff) {
    return out;
  }

  const float visibleRadius = min(q.sigma_radius, sqrt(2.0 * log(alpha / alphaCutoff)));
  // The eigen decomposition as in gsplat_vp, whose comments say why its
  // formulas avoid cancellation.
  const float mid = 0.5 * (a + c);
  const float halfDifference = 0.5 * (a - c);
  const float root = sqrt(halfDifference * halfDifference + b * b);
  const float lambda1 = max(mid + root, 0.01);
  const float lambda2 = max(mid - root, 0.01);
  float2 axis1 = halfDifference >= 0.0 ?
    float2(halfDifference + root, b) : float2(b, root - halfDifference);
  const float axisLength = length(axis1);
  axis1 = axisLength > 0.0 ? axis1 / axisLength : float2(1.0, 0.0);
  const float2 axis2 = float2(-axis1.y, axis1.x);

  out.centre = packed_float3(float3(
    dot(q.worldviewproj_rows[0], p), dot(q.worldviewproj_rows[1], p),
    dot(q.worldviewproj_rows[2], p)) / clip_w);
  out.visible_radius = visibleRadius;
  out.axes = float4(
    axis1 * (visibleRadius * sqrt(lambda1)) * 2.0 / q.viewport_size.xy,
    axis2 * (visibleRadius * sqrt(lambda2)) * 2.0 / q.viewport_size.xy);
  out.colour = float4(colour, alpha);
  return out;
}

// Stage 3: every survivor coloured and projected, between compaction and the
// sort. The compacted list is still in index order here, so the splat records
// and the spherical harmonics - by far the scene's largest arrays - are read in
// the order they are stored. After the sort the same reads land in depth order,
// scattered over the whole scene: on an RTX 4090 with the full Garden scene,
// evaluating SH there made the gather 24 times slower. Each result goes to the
// survivor's compacted slot, and the slot replaces the splat index as the
// sort's value, so the gather only permutes a buffer the size of the visible
// set. The sort is stable and slots follow index order, so the draw order is
// unchanged. slot_splats keeps each slot's splat index for diagnostics.
kernel void shade_project(
  device uint2 * compacted [[buffer(0)]],
  device const SplatRecord * splats [[buffer(1)]],
  device const float * sh_dc [[buffer(2)]],
  device const float * sh_rest [[buffer(3)]],
  device const PreparationState & state [[buffer(4)]],
  device ProjectedInstance * projected [[buffer(5)]],
  constant ViewParameters & p [[buffer(6)]],
  constant ProjectionParameters & q [[buffer(7)]],
  device uint * slot_splats [[buffer(8)]],
  uint slot [[thread_position_in_grid]])
{
  if (slot >= state.visible) {
    return;
  }
  const uint index = compacted[slot].y;
  const SplatRecord splat = splats[index];
  const float3 colour = evaluateSH(index, float3(splat.position), sh_dc, sh_rest, p);
  projected[slot] = projectSplat(splat, colour, q);
  slot_splats[slot] = index;
  compacted[slot].y = slot;
}

// Stage 5: the draw stream in depth order, a permutation of what shade_project
// wrote, so the vertex program only has to place each quad's corners.
kernel void gather_instances(
  device const uint2 * sorted [[buffer(0)]],
  device const ProjectedInstance * projected [[buffer(1)]],
  device const PreparationState & state [[buffer(4)]],
  device ProjectedInstance * instances [[buffer(5)]],
  uint output [[thread_position_in_grid]])
{
  if (output >= state.visible) {
    return;
  }
  instances[output] = projected[sorted[output].y];
}

// Tile rasterisation: the prepared splats composited front to back by compute,
// one threadgroup per 16x16 tile of pixels, in place of drawing a quad per
// splat. Each pixel stops once it is opaque, which blending in the render
// pipeline cannot do, and blends in registers rather than through the render
// target. This is the rasteriser of the 3DGS reference implementation, adapted
// to take the draw stream gather_instances already wrote in depth order:
//
// raster_bin_count and raster_bin_scan count the tiles each splat's quad
// touches and where each splat's (tile, draw position) pairs start;
// raster_bin_emit writes the pairs in draw order; the radix, sorting on the
// tile alone, is stable, so each tile's pairs stay in draw order;
// raster_tile_ranges finds each tile's run; raster_tiles composites it.

constant uint kRasterTile = 16;
// A pixel stops once less than this much of what lies behind it would show,
// as in the reference rasteriser.
constant float kTransmittanceEnd = 0.0001;

struct RasterParameters
{
  uint width;
  uint height;
  uint tiles_x;
  uint tiles_y;
  uint pair_capacity;
  uint pair_tile_capacity;  // radix tiles in pair_capacity; the radix tables' stride
  uint flip_y;              // pixel rows run downward from NDC y = +1
  float depth_min;          // the clip volume's near depth: 0 on Metal, -1 on OpenGL
};

// A splat's quad in pixels: its centre, and each axis scaled by the inverse of
// its squared length, so that the dot product with a pixel's offset from the
// centre gives the quad coordinate the vertex program would have interpolated.
struct RasterSplat
{
  float2 centre;
  float2 u;
  float2 v;
  float half_radius_squared;  // 0.5 visible_radius^2, for the Gaussian's exponent
  float opacity;
  float3 colour;
};

static RasterSplat rasterSplat(ProjectedInstance q, constant RasterParameters & r)
{
  const float2 size = float2(r.width, r.height);
  float2 centre = (float2(q.centre.xy) * 0.5 + 0.5) * size;
  float2 a1 = q.axes.xy * 0.5 * size;
  float2 a2 = q.axes.zw * 0.5 * size;
  if (r.flip_y) {
    centre.y = size.y - centre.y;
    a1.y = -a1.y;
    a2.y = -a2.y;
  }
  RasterSplat s;
  s.centre = centre;
  const float l1 = dot(a1, a1);
  const float l2 = dot(a2, a2);
  s.u = l1 > 0.0 ? a1 / l1 : float2(0.0);
  s.v = l2 > 0.0 ? a2 / l2 : float2(0.0);
  s.half_radius_squared = 0.5 * q.visible_radius * q.visible_radius;
  s.opacity = l1 > 0.0 && l2 > 0.0 ? q.colour.w : 0.0;
  s.colour = q.colour.xyz;
  return s;
}

// The tiles holding a pixel centre the quad covers, as (x0, y0, x1, y1), or
// false for a splat that draws nothing: rejected by the projection, clipped by
// depth as the whole quad would be, or off screen.
static bool tileBounds(ProjectedInstance q, constant RasterParameters & r, thread uint4 & tiles)
{
  if (q.visible_radius <= 0.0 || q.centre.z > 1.0 || q.centre.z < r.depth_min) {
    return false;
  }
  const float2 size = float2(r.width, r.height);
  float2 centre = (float2(q.centre.xy) * 0.5 + 0.5) * size;
  if (r.flip_y) {
    centre.y = size.y - centre.y;
  }
  const float2 extent = (abs(q.axes.xy) + abs(q.axes.zw)) * 0.5 * size;
  const float2 low = ceil(centre - extent - 0.5);
  const float2 high = floor(centre + extent - 0.5);
  if (!all(low <= high) || !all(high >= 0.0) || !all(low < size)) {
    return false;
  }
  const uint2 first = uint2(max(low, 0.0)) / kRasterTile;
  const uint2 last = uint2(min(high, size - 1.0)) / kRasterTile;
  tiles = uint4(first, last);
  return true;
}

static uint tileCount(uint4 tiles)
{
  return (tiles.z - tiles.x + 1u) * (tiles.w - tiles.y + 1u);
}

kernel void raster_bin_count(
  device const ProjectedInstance * instances [[buffer(0)]],
  device const PreparationState & state [[buffer(1)]],
  device uint * block_sums [[buffer(2)]],
  constant RasterParameters & r [[buffer(3)]],
  uint i [[thread_position_in_grid]],
  uint lane [[thread_index_in_threadgroup]],
  uint block [[threadgroup_position_in_grid]])
{
  threadgroup uint scratch[kScanWidth];
  uint count = 0;
  uint4 tiles;
  if (i < state.visible && tileBounds(instances[i], r, tiles)) {
    count = tileCount(tiles);
  }
  const uint inclusive = threadgroupInclusiveScan(count, lane, scratch);
  if (lane == kScanWidth - 1) {
    block_sums[block] = inclusive;
  }
}

kernel void raster_bin_scan(
  device const uint * block_sums [[buffer(0)]],
  device uint * block_offsets [[buffer(1)]],
  device const PreparationState & state [[buffer(2)]],
  device PreparationState * raster_state [[buffer(3)]],
  device DispatchArguments * sort_dispatch [[buffer(4)]],
  device DispatchArguments * pair_dispatch [[buffer(5)]],
  constant RasterParameters & r [[buffer(6)]],
  uint lane [[thread_index_in_threadgroup]])
{
  threadgroup uint scratch[kScanWidth];
  threadgroup uint carry;
  if (lane == 0) {
    carry = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint blocks = state.gather_groups;
  for (uint start = 0; start < blocks; start += kScanWidth) {
    const uint block = start + lane;
    const uint count = block < blocks ? block_sums[block] : 0u;
    const uint inclusive = threadgroupInclusiveScan(count, lane, scratch);
    if (block < blocks) {
      block_offsets[block] = carry + inclusive - count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == kScanWidth - 1) {
      carry += inclusive;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane != 0) {
    return;
  }
  // Pairs past the capacity are dropped; `reserved` keeps the full count so a
  // caller can tell.
  const uint pairs = min(carry, r.pair_capacity);
  const uint pair_tiles = (pairs + 1023u) / 1024u;
  const uint pair_groups = (pairs + kScanWidth - 1u) / kScanWidth;
  raster_state->visible = pairs;
  raster_state->visible_tiles = pair_tiles;
  raster_state->gather_groups = pair_groups;
  raster_state->reserved = carry;
  sort_dispatch->threadgroups[0] = max(pair_tiles, 1u);
  sort_dispatch->threadgroups[1] = 1u;
  sort_dispatch->threadgroups[2] = 1u;
  pair_dispatch->threadgroups[0] = max(pair_groups, 1u);
  pair_dispatch->threadgroups[1] = 1u;
  pair_dispatch->threadgroups[2] = 1u;
}

kernel void raster_bin_emit(
  device const ProjectedInstance * instances [[buffer(0)]],
  device const PreparationState & state [[buffer(1)]],
  device const uint * block_offsets [[buffer(2)]],
  constant RasterParameters & r [[buffer(3)]],
  device uint2 * pairs [[buffer(4)]],
  uint i [[thread_position_in_grid]],
  uint lane [[thread_index_in_threadgroup]],
  uint block [[threadgroup_position_in_grid]])
{
  threadgroup uint scratch[kScanWidth];
  uint count = 0;
  uint4 tiles = uint4(0);
  if (i < state.visible && tileBounds(instances[i], r, tiles)) {
    count = tileCount(tiles);
  }
  const uint inclusive = threadgroupInclusiveScan(count, lane, scratch);
  if (count == 0) {
    return;
  }
  uint cursor = block_offsets[block] + inclusive - count;
  for (uint y = tiles.y; y <= tiles.w; ++y) {
    for (uint x = tiles.x; x <= tiles.z; ++x) {
      if (cursor < r.pair_capacity) {
        pairs[cursor] = uint2(y * r.tiles_x + x, i);
      }
      ++cursor;
    }
  }
}

// Each tile's run of pairs as a start and an end, two words per tile. They are
// written by different threads, so they are stored as separate words: a store
// to one half of a uint2 may be a store of the whole vector.
kernel void raster_clear_ranges(
  device uint * ranges [[buffer(0)]],
  constant RasterParameters & r [[buffer(1)]],
  uint word [[thread_position_in_grid]])
{
  if (word < 2u * r.tiles_x * r.tiles_y) {
    ranges[word] = 0u;
  }
}

kernel void raster_tile_ranges(
  device const uint2 * pairs [[buffer(0)]],
  device const PreparationState & raster_state [[buffer(1)]],
  device uint * ranges [[buffer(2)]],
  uint j [[thread_position_in_grid]])
{
  const uint count = raster_state.visible;
  if (j >= count) {
    return;
  }
  const uint tile = pairs[j].x;
  if (j == 0 || pairs[j - 1].x != tile) {
    ranges[2u * tile] = j;
  }
  if (j + 1 == count || pairs[j + 1].x != tile) {
    ranges[2u * tile + 1u] = j + 1;
  }
}

// One threadgroup per tile, one thread per pixel. The tile's pairs arrive a
// batch of 256 at a time in shared memory. Most splats cover a few pixels of
// the 256, so each SIMD group, whose 32 threads hold an 8x4 block of the tile,
// first lists which splats of the batch reach its block at all and evaluates
// only those: in the full Garden scene, 13 in 14 splats in a tile's range
// cover no given pixel.
kernel void raster_tiles(
  device const uint2 * pairs [[buffer(0)]],
  device const uint * ranges [[buffer(1)]],
  device const ProjectedInstance * instances [[buffer(2)]],
  constant RasterParameters & r [[buffer(3)]],
  metal::texture2d<float, metal::access::write> output [[texture(0)]],
  uint2 tile [[threadgroup_position_in_grid]],
  uint lane [[thread_index_in_threadgroup]],
  uint group [[simdgroup_index_in_threadgroup]],
  uint simd_lane [[thread_index_in_simdgroup]])
{
  threadgroup float4 batch_quad[kScanWidth];    // centre, u
  threadgroup float4 batch_shape[kScanWidth];   // v, half_radius_squared, opacity
  threadgroup float4 batch_colour[kScanWidth];
  threadgroup short4 batch_box[kScanWidth];     // covered pixels, first and last, inclusive
  threadgroup ushort block_list[kScanWidth / 32][kScanWidth];
  threadgroup atomic_uint finished;

  const uint2 block = tile * kRasterTile + uint2((group & 1u) * 8u, (group >> 1u) * 4u);
  const uint2 pixel = block + uint2(simd_lane & 7u, simd_lane >> 3u);
  const bool inside = pixel.x < r.width && pixel.y < r.height;
  const float2 position = float2(pixel) + 0.5;
  const uint index = tile.y * r.tiles_x + tile.x;
  const uint2 range = uint2(ranges[2u * index], ranges[2u * index + 1u]);
  float transmittance = 1.0;
  float3 colour = float3(0.0);
  bool done = !inside;

  // The pairs are in draw order, back to front; walk them from the end.
  for (uint top = range.y; top > range.x; ) {
    const uint batch = min(kScanWidth, top - range.x);
    if (lane == 0) {
      atomic_store_explicit(&finished, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < batch) {
      const ProjectedInstance q = instances[pairs[top - 1u - lane].y];
      const RasterSplat s = rasterSplat(q, r);
      batch_quad[lane] = float4(s.centre, s.u);
      batch_shape[lane] = float4(s.v, s.half_radius_squared, s.opacity);
      batch_colour[lane] = float4(s.colour, 0.0);
      const float2 extent = (abs(q.axes.xy) + abs(q.axes.zw)) * 0.5 * float2(r.width, r.height);
      const float2 low = clamp(ceil(s.centre - extent - 0.5), -1.0, 32767.0);
      const float2 high = clamp(floor(s.centre + extent - 0.5), -1.0, 32767.0);
      batch_box[lane] = short4(short2(low), short2(high));
    }
    if (done) {
      atomic_fetch_add_explicit(&finished, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&finished, memory_order_relaxed) == kScanWidth) {
      break;
    }

    // This SIMD group's list, in batch order.
    uint listed = 0;
    const short4 mine = short4(short2(block), short2(block + uint2(7u, 3u)));
    for (uint first = 0; first < batch; first += 32u) {
      const uint k = first + simd_lane;
      bool reaches = false;
      if (k < batch) {
        const short4 box = batch_box[k];
        reaches = box.x <= mine.z && box.z >= mine.x && box.y <= mine.w && box.w >= mine.y;
      }
      const uint slot = listed + simd_prefix_exclusive_sum(reaches ? 1u : 0u);
      if (reaches) {
        block_list[group][slot] = ushort(k);
      }
      listed += simd_sum(reaches ? 1u : 0u);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint n = 0; n < listed && !done; ++n) {
      const uint k = block_list[group][n];
      const float4 quad = batch_quad[k];
      const float4 shape = batch_shape[k];
      const float2 offset = position - quad.xy;
      const float cu = dot(offset, quad.zw);
      const float cv = dot(offset, shape.xy);
      if (abs(cu) > 1.0 || abs(cv) > 1.0) {
        continue;
      }
      const float alpha = min(0.99, shape.w * exp(-shape.z * (cu * cu + cv * cv)));
      if (alpha < 1.0 / 255.0) {
        continue;
      }
      const float next = transmittance * (1.0 - alpha);
      if (next < kTransmittanceEnd) {
        done = true;
        break;
      }
      colour += batch_colour[k].xyz * (alpha * transmittance);
      transmittance = next;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    top -= batch;
  }
  if (inside) {
    output.write(float4(colour, 1.0 - transmittance), pixel);
  }
}
