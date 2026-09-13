// Prototype derived from the GPU radix structure in MetalSprocketsGaussianSplats.
// See tools/THIRD_PARTY_NOTICES.md.

#include <metal_stdlib>
using namespace metal;

constant uint kRadix = 256;

struct RadixParameters
{
  uint count;
  uint tile_count;
  uint elements_per_tile;
  uint shift;
};

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
