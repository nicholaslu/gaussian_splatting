// The radix dispatch sequence below is adapted, with the kernels it drives in
// gsplat_prepare.metal, from MetalSprocketsGaussianSplats, copyright (c) 2025
// Jonathan Wight, MIT licensed; see THIRD_PARTY_NOTICES.md in this package.

#import <Foundation/Foundation.h>

#include "metal_view_preparation_core.h"

#include <algorithm>
#include <cstring>
#include <string>

namespace gaussian_splatting_rviz_plugins
{
namespace
{

// The radix and the compaction rank the lanes of a threadgroup against each
// other with simd_ballot, which only means anything when a threadgroup is
// exactly one SIMD group. makePipeline() refuses a GPU where that does not
// hold rather than letting it sort wrongly.
constexpr std::uint32_t kThreadsPerTile = 32;
constexpr std::uint32_t kElementsPerTile = 1024;
constexpr std::uint32_t kRadix = 256;
// Threads in each scan threadgroup; kScanWidth in gsplat_prepare.metal.
constexpr std::uint32_t kScanWidth = 256;
// Slots finish_order gathers from per pass: 64 MiB of DrawnSplat.
constexpr std::uint32_t kFinishSlotsPerPass = 1u << 20;

// Mirror the structs of the same names in gsplat_prepare.metal.
struct TileParameters
{
  std::uint32_t count;
  std::uint32_t tile_count;
  std::uint32_t elements_per_tile;
  std::uint32_t shift;
};

struct ScanParameters
{
  std::uint32_t tile_count;
  std::uint32_t elements_per_tile;
  std::uint32_t gather_width;
  std::uint32_t index_count;
};

void barrier(id<MTLComputeCommandEncoder> encoder)
{
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

MTLSize linearWidth(id<MTLComputePipelineState> pipeline, NSUInteger cap)
{
  return MTLSizeMake(std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, cap), 1, 1);
}

}  // namespace

MetalViewPreparationCore::MetalViewPreparationCore(
  id<MTLDevice> device, const std::string & shader_path)
: device_(device)
{
  if (!device_) {
    error_ = "no Metal device";
    return;
  }
  static_assert(sizeof(State) == 16, "PreparationState layout");

  @autoreleasepool {
    NSError * error = nil;
    NSString * source = [NSString stringWithContentsOfFile:
      [NSString stringWithUTF8String:shader_path.c_str()]
      encoding:NSUTF8StringEncoding error:&error];
    if (!source) {
      setError("reading " + shader_path, error);
      return;
    }
    library_ = [device_ newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
    if (!library_) {
      setError("compiling Metal preparation shader", error);
      return;
    }
    keys_pipeline_ = makePipeline(@"make_depth_keys", Dispatch::kLinear);
    compact_count_pipeline_ = makePipeline(@"compact_count", Dispatch::kLinear);
    compact_scan_pipeline_ = makePipeline(@"compact_scan", Dispatch::kScan);
    compact_scatter_pipeline_ = makePipeline(@"compact_scatter", Dispatch::kSimdRanked);
    histogram_pipeline_ = makePipeline(@"radix_histogram", Dispatch::kTiled);
    offset_pipeline_ = makePipeline(@"radix_scan_offsets", Dispatch::kScan);
    base_pipeline_ = makePipeline(@"radix_scan_digit_bases", Dispatch::kScan);
    scatter_pipeline_ = makePipeline(@"radix_scatter", Dispatch::kSimdRanked);
    shade_pipeline_ = makePipeline(@"shade_slots", Dispatch::kLinear);
    gather_pipeline_ = makePipeline(@"reproject_gather", Dispatch::kLinear);
    finish_pipeline_ = makePipeline(@"finish_order", Dispatch::kLinear);
    recolour_pipeline_ = makePipeline(@"recolour_order", Dispatch::kLinear);
  }
}

bool MetalViewPreparationCore::ready() const
{
  return keys_pipeline_ && compact_count_pipeline_ && compact_scan_pipeline_ &&
         compact_scatter_pipeline_ && histogram_pipeline_ && offset_pipeline_ &&
         base_pipeline_ && scatter_pipeline_ && shade_pipeline_ && gather_pipeline_ &&
         finish_pipeline_ && recolour_pipeline_;
}

const std::string & MetalViewPreparationCore::error() const
{
  return error_;
}

bool MetalViewPreparationCore::configure(std::uint32_t count, std::uint32_t index_count)
{
  clear();
  if (!ready()) {
    return false;
  }
  if (count == 0) {
    error_ = "Metal preparation requires a non-zero splat count";
    return false;
  }

  count_ = count;
  index_count_ = index_count;
  tile_count_ = (count + kElementsPerTile - 1u) / kElementsPerTile;
  // Both survivor stages run from the same indirect arguments, sized for this.
  gather_width_ = static_cast<std::uint32_t>(std::min<NSUInteger>(
      std::min(gather_pipeline_.maxTotalThreadsPerThreadgroup,
      shade_pipeline_.maxTotalThreadsPerThreadgroup), 256u));

  const NSUInteger pairs = static_cast<NSUInteger>(count) * sizeof(std::uint32_t) * 2u;
  const NSUInteger per_tile = static_cast<NSUInteger>(tile_count_) * sizeof(std::uint32_t);
  const NSUInteger tables = per_tile * kRadix;
  staged_ = newBuffer(pairs, "staged keys");
  tile_counts_ = newBuffer(per_tile, "tile survivor counts");
  tile_offsets_ = newBuffer(per_tile, "tile survivor offsets");
  histograms_ = newBuffer(tables, "radix histograms");
  offsets_ = newBuffer(tables, "radix offsets");
  totals_ = newBuffer(kRadix * sizeof(std::uint32_t), "radix totals");
  digit_bases_ = newBuffer(kRadix * sizeof(std::uint32_t), "radix digit bases");
  const NSUInteger survivors = static_cast<NSUInteger>(count) * sizeof(DrawnSplat);
  pairs_ = newBuffer(pairs, "key/slot pairs");
  slots_ = newBuffer(survivors, "survivors by slot");
  drawn_ = newBuffer(survivors, "survivors in draw order");
  colours_ = newBuffer(static_cast<NSUInteger>(count) * sizeof(float) * 4u, "drawn colours");
  slot_draws_ = newBuffer(
    static_cast<NSUInteger>(count) * sizeof(std::uint32_t) * 2u, "slot splats and positions");
  bool allocated = staged_ && tile_counts_ && tile_offsets_ && histograms_ && offsets_ &&
    totals_ && digit_bases_ && pairs_ && slots_ && drawn_ && colours_ && slot_draws_;
  for (Order & order : orders_) {
    order.state = newBuffer(sizeof(State), "order state");
    order.tile_dispatch = newBuffer(sizeof(std::uint32_t) * 3u, "order tile dispatch");
    order.gather_dispatch = newBuffer(sizeof(std::uint32_t) * 3u, "order gather dispatch");
    allocated = allocated && order.state && order.tile_dispatch && order.gather_dispatch;
    if (order.state) {
      std::memset(order.state.contents, 0, sizeof(State));
    }
  }
  if (!allocated) {
    const std::string failure = error_;
    clear();
    error_ = failure;
    return false;
  }
  error_.clear();
  return true;
}

void MetalViewPreparationCore::clear()
{
  count_ = 0;
  tile_count_ = 0;
  index_count_ = 0;
  gather_width_ = 0;
  splats_ = nil;
  sh_dc_ = nil;
  sh_rest_ = nil;
  staged_ = nil;
  tile_counts_ = nil;
  tile_offsets_ = nil;
  histograms_ = nil;
  offsets_ = nil;
  totals_ = nil;
  digit_bases_ = nil;
  pairs_ = nil;
  slots_ = nil;
  drawn_ = nil;
  colours_ = nil;
  slot_draws_ = nil;
  colour_frame_ = 0;
  refreshable_ = false;
  for (Order & order : orders_) {
    order = Order{};
  }
  current_ = -1;
  building_ = -1;
  build_step_ = 0;
}

bool MetalViewPreparationCore::uploadStaticData(
  const SplatRecord * records, std::size_t record_count,
  const float * sh_dc, std::size_t sh_dc_floats,
  const float * sh_rest, std::size_t sh_rest_floats,
  id<MTLCommandBuffer> last_user)
{
  if (!ready() || count_ == 0) {
    error_ = "Metal static upload called before configuration";
    return false;
  }
  if (!records || record_count != count_ || !sh_dc ||
    sh_dc_floats != static_cast<std::size_t>(count_) * 3u)
  {
    error_ = "Metal static upload does not match the configured splat count";
    return false;
  }

  // A pass that has been committed but has not finished may be reading these
  // buffers, so wait that one out. One still being encoded has not started,
  // and will simply read the new data.
  if (last_user) {
    const MTLCommandBufferStatus status = last_user.status;
    if (status == MTLCommandBufferStatusCommitted || status == MTLCommandBufferStatusScheduled) {
      [last_user waitUntilCompleted];
    }
  }

  const NSUInteger record_bytes = static_cast<NSUInteger>(count_) * sizeof(SplatRecord);
  const NSUInteger dc_bytes = sh_dc_floats * sizeof(float);
  const NSUInteger rest_bytes = sh_rest_floats * sizeof(float);
  if (!fit(splats_, record_bytes, "splat records") || !fit(sh_dc_, dc_bytes, "SH DC") ||
    !fit(sh_rest_, rest_bytes, "SH rest"))
  {
    return false;
  }
  std::memcpy(splats_.contents, records, record_bytes);
  std::memcpy(sh_dc_.contents, sh_dc, dc_bytes);
  if (sh_rest && rest_bytes > 0) {
    std::memcpy(sh_rest_.contents, sh_rest, rest_bytes);
  }
  // The orders' colours and survivors belong to the old data.
  current_ = -1;
  building_ = -1;
  build_step_ = 0;
  error_.clear();
  return true;
}

bool MetalViewPreparationCore::encodeCull(
  id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters)
{
  if (!ready() || count_ == 0 || !splats_ || !sh_dc_ || !sh_rest_ ||
    parameters.count != count_)
  {
    error_ = "Metal view preparation is not fully configured";
    return false;
  }
  const NSUInteger rest_needed =
    static_cast<NSUInteger>(count_) * parameters.sh_coefficients * 3u * sizeof(float);
  if (parameters.sh_coefficients > 15u || sh_rest_.length < rest_needed) {
    error_ = "uploaded SH coefficients do not cover the requested degree";
    return false;
  }
  // The order not being drawn from.
  building_ = current_ == 0 ? 1 : 0;
  build_step_ = 1;
  build_view_ = parameters;
  const Order & order = orders_[building_];

  [encoder setComputePipelineState:keys_pipeline_];
  [encoder setBuffer:splats_ offset:0 atIndex:0];
  [encoder setBuffer:staged_ offset:0 atIndex:1];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(count_, 1, 1)
     threadsPerThreadgroup:linearWidth(keys_pipeline_, 256u)];
  barrier(encoder);

  const TileParameters tiles{count_, tile_count_, kElementsPerTile, 0u};
  [encoder setComputePipelineState:compact_count_pipeline_];
  [encoder setBuffer:staged_ offset:0 atIndex:0];
  [encoder setBuffer:tile_counts_ offset:0 atIndex:1];
  [encoder setBytes:&tiles length:sizeof(tiles) atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(tile_count_, 1, 1)
     threadsPerThreadgroup:linearWidth(compact_count_pipeline_, 256u)];
  barrier(encoder);

  const ScanParameters scan{tile_count_, kElementsPerTile, gather_width_, index_count_};
  [encoder setComputePipelineState:compact_scan_pipeline_];
  [encoder setBuffer:tile_counts_ offset:0 atIndex:0];
  [encoder setBuffer:tile_offsets_ offset:0 atIndex:1];
  [encoder setBuffer:order.state offset:0 atIndex:2];
  [encoder setBuffer:order.tile_dispatch offset:0 atIndex:3];
  [encoder setBuffer:order.gather_dispatch offset:0 atIndex:4];
  [encoder setBytes:&scan length:sizeof(scan) atIndex:6];
  [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kScanWidth, 1, 1)];
  barrier(encoder);

  [encoder setComputePipelineState:compact_scatter_pipeline_];
  [encoder setBuffer:staged_ offset:0 atIndex:0];
  [encoder setBuffer:pairs_ offset:0 atIndex:1];
  [encoder setBuffer:tile_offsets_ offset:0 atIndex:2];
  [encoder setBytes:&tiles length:sizeof(tiles) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(tile_count_, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kThreadsPerTile, 1, 1)];
  barrier(encoder);
  return true;
}

void MetalViewPreparationCore::encodeShade(
  id<MTLComputeCommandEncoder> encoder, std::uint32_t index, std::uint32_t count)
{
  if (building_ < 0 || count == 0 || index >= count) {
    return;
  }
  const Order & order = orders_[building_];
  // Rewrites the compacted pairs, the sort's input, to carry slots.
  [encoder setComputePipelineState:shade_pipeline_];
  [encoder setBuffer:pairs_ offset:0 atIndex:0];
  [encoder setBuffer:splats_ offset:0 atIndex:1];
  [encoder setBuffer:sh_dc_ offset:0 atIndex:2];
  [encoder setBuffer:sh_rest_ offset:0 atIndex:3];
  [encoder setBuffer:order.state offset:0 atIndex:4];
  [encoder setBuffer:slots_ offset:0 atIndex:5];
  [encoder setBytes:&build_view_ length:sizeof(build_view_) atIndex:6];
  const std::uint32_t part[2] = {index, count};
  [encoder setBytes:part length:sizeof(part) atIndex:7];
  // Sized for every survivor; threads past the part return at once.
  [encoder dispatchThreadgroupsWithIndirectBuffer:order.gather_dispatch indirectBufferOffset:0
                            threadsPerThreadgroup:MTLSizeMake(gather_width_, 1, 1)];
  barrier(encoder);
  build_step_ = index + 1 == count ? 3 : 2;
}

void MetalViewPreparationCore::encodeRadix(
  id<MTLComputeCommandEncoder> encoder, const Order & order, std::uint32_t first_bit,
  std::uint32_t end_bit)
{
  id<MTLBuffer> input = pairs_;
  id<MTLBuffer> output = staged_;
  for (std::uint32_t shift = first_bit; shift < end_bit; shift += 8u) {
    const TileParameters radix{count_, tile_count_, kElementsPerTile, shift};

    [encoder setComputePipelineState:histogram_pipeline_];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:histograms_ offset:0 atIndex:1];
    [encoder setBytes:&radix length:sizeof(radix) atIndex:2];
    [encoder setBuffer:order.state offset:0 atIndex:3];
    [encoder dispatchThreadgroupsWithIndirectBuffer:order.tile_dispatch indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(kThreadsPerTile, 1, 1)];
    barrier(encoder);

    [encoder setComputePipelineState:offset_pipeline_];
    [encoder setBuffer:histograms_ offset:0 atIndex:0];
    [encoder setBuffer:offsets_ offset:0 atIndex:1];
    [encoder setBuffer:totals_ offset:0 atIndex:2];
    [encoder setBytes:&radix length:sizeof(radix) atIndex:3];
    [encoder setBuffer:order.state offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(kRadix, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kScanWidth, 1, 1)];
    barrier(encoder);

    [encoder setComputePipelineState:base_pipeline_];
    [encoder setBuffer:totals_ offset:0 atIndex:0];
    [encoder setBuffer:digit_bases_ offset:0 atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kScanWidth, 1, 1)];
    barrier(encoder);

    [encoder setComputePipelineState:scatter_pipeline_];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:offsets_ offset:0 atIndex:2];
    [encoder setBuffer:digit_bases_ offset:0 atIndex:3];
    [encoder setBytes:&radix length:sizeof(radix) atIndex:4];
    [encoder setBuffer:order.state offset:0 atIndex:5];
    [encoder dispatchThreadgroupsWithIndirectBuffer:order.tile_dispatch indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(kThreadsPerTile, 1, 1)];
    barrier(encoder);
    std::swap(input, output);
  }
}

bool MetalViewPreparationCore::encodeSort(
  id<MTLComputeCommandEncoder> encoder, std::uint32_t key_bits)
{
  // Whole byte pairs, so that the pairs end back in the order's own buffer.
  if (!ready() || count_ == 0 || key_bits == 0u || key_bits > 32u || key_bits % 16u != 0u) {
    error_ = "Metal sort is not configured, or the key width is not 16 or 32 bits";
    return false;
  }
  // With no build under way, as when benchmarking, the newest order again.
  const int sorting = building_ >= 0 ? building_ : current_;
  if (sorting < 0) {
    error_ = "no order to sort";
    return false;
  }
  encodeRadix(encoder, orders_[sorting], 0u, key_bits);
  if (building_ >= 0) {
    completeBuild(encoder);
  }
  return true;
}

bool MetalViewPreparationCore::encodeGather(
  id<MTLComputeCommandEncoder> encoder, const ProjectionParameters & projection,
  id<MTLBuffer> instances, id<MTLBuffer> draw_arguments)
{
  if (current_ < 0 || !instances || !draw_arguments ||
    instances.length < static_cast<NSUInteger>(count_) * sizeof(ProjectedInstance))
  {
    error_ = "no complete order, or an instance buffer smaller than the splat count";
    return false;
  }
  const Order & order = orders_[current_];
  [encoder setComputePipelineState:gather_pipeline_];
  [encoder setBuffer:drawn_ offset:0 atIndex:0];
  [encoder setBuffer:colours_ offset:0 atIndex:1];
  [encoder setBuffer:order.state offset:0 atIndex:4];
  const std::uint32_t refreshed = refreshable_ && colour_parts_ > 0 ? 1u : 0u;
  [encoder setBytes:&refreshed length:sizeof(refreshed) atIndex:9];
  [encoder setBuffer:instances offset:0 atIndex:5];
  [encoder setBytes:&projection length:sizeof(projection) atIndex:6];
  [encoder setBuffer:draw_arguments offset:0 atIndex:7];
  [encoder setBytes:&index_count_ length:sizeof(index_count_) atIndex:8];
  [encoder dispatchThreadgroupsWithIndirectBuffer:order.gather_dispatch indirectBufferOffset:0
                            threadsPerThreadgroup:MTLSizeMake(gather_width_, 1, 1)];
  return true;
}

bool MetalViewPreparationCore::encode(
  id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
  const ProjectionParameters & projection, id<MTLBuffer> instances,
  id<MTLBuffer> draw_arguments)
{
  if (!encodeCull(encoder, parameters)) {
    return false;
  }
  encodeShade(encoder);
  if (!encodeSort(encoder, 32u) ||
    !encodeGather(encoder, projection, instances, draw_arguments))
  {
    return false;
  }
  error_.clear();
  return true;
}

bool MetalViewPreparationCore::encodeIncremental(
  id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
  const ProjectionParameters & projection, id<MTLBuffer> instances,
  id<MTLBuffer> draw_arguments)
{
  if (current_ < 0) {
    return encode(encoder, parameters, projection, instances, draw_arguments);
  }
  if (encodeBuildStep(encoder, parameters) == BuildStep::kFailed) {
    return false;
  }
  if (colour_parts_ > 0 && refreshable_) {
    encodeRecolour(encoder, parameters, colour_frame_++ % colour_parts_, colour_parts_);
  }
  if (!encodeGather(encoder, projection, instances, draw_arguments)) {
    return false;
  }
  error_.clear();
  return true;
}

void MetalViewPreparationCore::encodeRecolour(
  id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
  std::uint32_t index, std::uint32_t count)
{
  if (current_ < 0 || !refreshable_ || count == 0 || index >= count) {
    return;
  }
  const Order & order = orders_[current_];
  [encoder setComputePipelineState:recolour_pipeline_];
  [encoder setBuffer:slot_draws_ offset:0 atIndex:0];
  [encoder setBuffer:splats_ offset:0 atIndex:2];
  [encoder setBuffer:sh_dc_ offset:0 atIndex:3];
  [encoder setBuffer:sh_rest_ offset:0 atIndex:4];
  [encoder setBuffer:order.state offset:0 atIndex:5];
  [encoder setBuffer:colours_ offset:0 atIndex:6];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:7];
  const std::uint32_t part[2] = {index, count};
  [encoder setBytes:part length:sizeof(part) atIndex:8];
  // Sized for every survivor; threads past the part return at once.
  [encoder dispatchThreadgroupsWithIndirectBuffer:order.gather_dispatch indirectBufferOffset:0
                            threadsPerThreadgroup:MTLSizeMake(gather_width_, 1, 1)];
  barrier(encoder);
}

MetalViewPreparationCore::BuildStep MetalViewPreparationCore::encodeBuildStep(
  id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters)
{
  // Five steps of comparable cost on the full Garden scene: culling and
  // compaction, each half of the colour, and each half of the sort.
  if (building_ < 0) {
    return encodeCull(encoder, parameters) ? BuildStep::kCull : BuildStep::kFailed;
  }
  if (build_step_ == 1 || build_step_ == 2) {
    encodeShade(encoder, static_cast<std::uint32_t>(build_step_ - 1), 2u);
    return BuildStep::kShade;
  }
  if (build_step_ == 3) {
    encodeRadix(encoder, orders_[building_], 0u, 16u);
    build_step_ = 4;
    return BuildStep::kSortLow;
  }
  encodeRadix(encoder, orders_[building_], 16u, 32u);
  completeBuild(encoder);
  return BuildStep::kSortHigh;
}

void MetalViewPreparationCore::completeBuild(id<MTLComputeCommandEncoder> encoder)
{
  const Order & order = orders_[building_];
  [encoder setComputePipelineState:finish_pipeline_];
  [encoder setBuffer:pairs_ offset:0 atIndex:0];
  [encoder setBuffer:slots_ offset:0 atIndex:1];
  [encoder setBuffer:slot_draws_ offset:0 atIndex:2];
  [encoder setBuffer:order.state offset:0 atIndex:4];
  [encoder setBuffer:drawn_ offset:0 atIndex:5];
  [encoder setBuffer:colours_ offset:0 atIndex:7];
  const std::uint32_t refreshable = colour_parts_ > 0 ? 1u : 0u;
  [encoder setBytes:&refreshable length:sizeof(refreshable) atIndex:8];
  // The passes write disjoint positions, so they need no barrier between them.
  for (std::uint32_t first = 0; first < count_; first += kFinishSlotsPerPass) {
    const std::uint32_t range[2] = {first, first + std::min(kFinishSlotsPerPass, count_ - first)};
    [encoder setBytes:range length:sizeof(range) atIndex:6];
    [encoder dispatchThreadgroupsWithIndirectBuffer:order.gather_dispatch indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(gather_width_, 1, 1)];
  }
  barrier(encoder);
  refreshable_ = refreshable != 0u;
  current_ = building_;
  building_ = -1;
  build_step_ = 0;
}

void MetalViewPreparationCore::primeSortInput(std::uint32_t visible)
{
  building_ = current_ == 0 ? 1 : 0;
  build_step_ = 3;
  const Order & order = orders_[building_];
  auto * state = static_cast<State *>(order.state.contents);
  state->visible = std::min(visible, count_);
  state->visible_tiles = (state->visible + kElementsPerTile - 1u) / kElementsPerTile;
  state->gather_groups = 0;
  state->reserved = 0;
  auto * dispatch = static_cast<std::uint32_t *>(order.tile_dispatch.contents);
  dispatch[0] = std::max<std::uint32_t>(state->visible_tiles, 1u);
  dispatch[1] = 1u;
  dispatch[2] = 1u;
}

id<MTLBuffer> MetalViewPreparationCore::stateBuffer() const
{
  return current_ >= 0 ? orders_[current_].state : nil;
}

NSUInteger MetalViewPreparationCore::scratchBytes() const
{
  NSUInteger total = 0;
  for (id<MTLBuffer> buffer : {staged_, pairs_, slots_, drawn_, colours_, slot_draws_,
      tile_counts_, tile_offsets_, histograms_, offsets_, totals_, digit_bases_})
  {
    total += buffer ? buffer.length : 0u;
  }
  for (const Order & order : orders_) {
    for (id<MTLBuffer> buffer : {order.state, order.tile_dispatch, order.gather_dispatch})
    {
      total += buffer ? buffer.length : 0u;
    }
  }
  return total;
}

id<MTLComputePipelineState> MetalViewPreparationCore::makePipeline(
  NSString * name, Dispatch dispatch)
{
  id<MTLFunction> function = [library_ newFunctionWithName:name];
  if (!function) {
    if (error_.empty()) {
      error_ = "Metal preparation shader has no function " + std::string(name.UTF8String);
    }
    return nil;
  }
  NSError * error = nil;
  id<MTLComputePipelineState> pipeline =
    [device_ newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) {
    if (error_.empty()) {
      setError(std::string("creating pipeline ") + name.UTF8String, error);
    }
    return nil;
  }
  if (dispatch != Dispatch::kLinear && pipeline.maxTotalThreadsPerThreadgroup < kThreadsPerTile) {
    if (error_.empty()) {
      error_ = std::string(name.UTF8String) +
        " needs 32 threads per threadgroup; this GPU allows " +
        std::to_string(pipeline.maxTotalThreadsPerThreadgroup);
    }
    return nil;
  }
  if (dispatch == Dispatch::kScan && pipeline.maxTotalThreadsPerThreadgroup < kScanWidth) {
    if (error_.empty()) {
      error_ = std::string(name.UTF8String) + " scans with 256 threads per threadgroup; this GPU "
        "allows " + std::to_string(pipeline.maxTotalThreadsPerThreadgroup);
    }
    return nil;
  }
  if (dispatch == Dispatch::kSimdRanked && pipeline.threadExecutionWidth != kThreadsPerTile) {
    if (error_.empty()) {
      error_ = std::string(name.UTF8String) + " ranks lanes within a SIMD group of 32; this GPU's "
        "execution width is " + std::to_string(pipeline.threadExecutionWidth);
    }
    return nil;
  }
  return pipeline;
}

id<MTLBuffer> MetalViewPreparationCore::newBuffer(NSUInteger bytes, const char * label)
{
  id<MTLBuffer> buffer = [device_ newBufferWithLength:std::max<NSUInteger>(bytes, 4u)
                                              options:MTLResourceStorageModeShared];
  if (!buffer) {
    error_ = "Metal allocation failed for " + std::string(label) + " (" +
      std::to_string(bytes) + " bytes)";
  } else {
    buffer.label = [NSString stringWithUTF8String:label];
  }
  return buffer;
}

bool MetalViewPreparationCore::fit(
  id<MTLBuffer> __strong & buffer, NSUInteger bytes, const char * label)
{
  const NSUInteger length = std::max<NSUInteger>(bytes, 4u);
  if (buffer && buffer.length == length) {
    return true;
  }
  // Released before the replacement is allocated, so an old and a new copy of
  // the scene are never both held at once.
  buffer = nil;
  buffer = newBuffer(length, label);
  return buffer != nil;
}

void MetalViewPreparationCore::setError(const std::string & operation, NSError * error)
{
  error_ = operation;
  if (error) {
    error_ += ": ";
    error_ += error.localizedDescription.UTF8String;
  }
}

}  // namespace gaussian_splatting_rviz_plugins
