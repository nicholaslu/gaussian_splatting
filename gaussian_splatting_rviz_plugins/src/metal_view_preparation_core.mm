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
    compact_scan_pipeline_ = makePipeline(@"compact_scan", Dispatch::kLinear);
    compact_scatter_pipeline_ = makePipeline(@"compact_scatter", Dispatch::kSimdRanked);
    histogram_pipeline_ = makePipeline(@"radix_histogram", Dispatch::kTiled);
    offset_pipeline_ = makePipeline(@"radix_scan_offsets", Dispatch::kLinear);
    base_pipeline_ = makePipeline(@"radix_scan_digit_bases", Dispatch::kLinear);
    scatter_pipeline_ = makePipeline(@"radix_scatter", Dispatch::kSimdRanked);
    shade_pipeline_ = makePipeline(@"shade_project", Dispatch::kLinear);
    gather_pipeline_ = makePipeline(@"gather_instances", Dispatch::kLinear);
  }
}

bool MetalViewPreparationCore::ready() const
{
  return keys_pipeline_ && compact_count_pipeline_ && compact_scan_pipeline_ &&
         compact_scatter_pipeline_ && histogram_pipeline_ && offset_pipeline_ &&
         base_pipeline_ && scatter_pipeline_ && shade_pipeline_ && gather_pipeline_;
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
  keys_a_ = newBuffer(pairs, "compacted keys");
  keys_b_ = newBuffer(pairs, "staged keys");
  tile_counts_ = newBuffer(per_tile, "tile survivor counts");
  tile_offsets_ = newBuffer(per_tile, "tile survivor offsets");
  histograms_ = newBuffer(tables, "radix histograms");
  offsets_ = newBuffer(tables, "radix offsets");
  totals_ = newBuffer(kRadix * sizeof(std::uint32_t), "radix totals");
  digit_bases_ = newBuffer(kRadix * sizeof(std::uint32_t), "radix digit bases");
  state_ = newBuffer(sizeof(State), "preparation state");
  tile_dispatch_ = newBuffer(sizeof(std::uint32_t) * 3u, "tile dispatch arguments");
  gather_dispatch_ = newBuffer(sizeof(std::uint32_t) * 3u, "gather dispatch arguments");
  projected_ = newBuffer(
    static_cast<NSUInteger>(count) * sizeof(ProjectedInstance), "projected survivors");
  slot_splats_ = newBuffer(static_cast<NSUInteger>(count) * sizeof(std::uint32_t), "slot splats");
  if (!keys_a_ || !keys_b_ || !tile_counts_ || !tile_offsets_ || !histograms_ || !offsets_ ||
    !totals_ || !digit_bases_ || !state_ || !tile_dispatch_ || !gather_dispatch_ ||
    !projected_ || !slot_splats_)
  {
    const std::string failure = error_;
    clear();
    error_ = failure;
    return false;
  }
  std::memset(state_.contents, 0, sizeof(State));
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
  keys_a_ = nil;
  keys_b_ = nil;
  tile_counts_ = nil;
  tile_offsets_ = nil;
  histograms_ = nil;
  offsets_ = nil;
  totals_ = nil;
  digit_bases_ = nil;
  state_ = nil;
  tile_dispatch_ = nil;
  gather_dispatch_ = nil;
  projected_ = nil;
  slot_splats_ = nil;
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
  error_.clear();
  return true;
}

bool MetalViewPreparationCore::encodeCull(
  id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
  id<MTLBuffer> draw_arguments)
{
  if (!ready() || count_ == 0 || !splats_ || !sh_dc_ || !sh_rest_ || !draw_arguments ||
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

  [encoder setComputePipelineState:keys_pipeline_];
  [encoder setBuffer:splats_ offset:0 atIndex:0];
  [encoder setBuffer:keys_b_ offset:0 atIndex:1];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(count_, 1, 1)
     threadsPerThreadgroup:linearWidth(keys_pipeline_, 256u)];
  barrier(encoder);

  const TileParameters tiles{count_, tile_count_, kElementsPerTile, 0u};
  [encoder setComputePipelineState:compact_count_pipeline_];
  [encoder setBuffer:keys_b_ offset:0 atIndex:0];
  [encoder setBuffer:tile_counts_ offset:0 atIndex:1];
  [encoder setBytes:&tiles length:sizeof(tiles) atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(tile_count_, 1, 1)
     threadsPerThreadgroup:linearWidth(compact_count_pipeline_, 256u)];
  barrier(encoder);

  const ScanParameters scan{tile_count_, kElementsPerTile, gather_width_, index_count_};
  [encoder setComputePipelineState:compact_scan_pipeline_];
  [encoder setBuffer:tile_counts_ offset:0 atIndex:0];
  [encoder setBuffer:tile_offsets_ offset:0 atIndex:1];
  [encoder setBuffer:state_ offset:0 atIndex:2];
  [encoder setBuffer:tile_dispatch_ offset:0 atIndex:3];
  [encoder setBuffer:gather_dispatch_ offset:0 atIndex:4];
  [encoder setBuffer:draw_arguments offset:0 atIndex:5];
  [encoder setBytes:&scan length:sizeof(scan) atIndex:6];
  [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  barrier(encoder);

  [encoder setComputePipelineState:compact_scatter_pipeline_];
  [encoder setBuffer:keys_b_ offset:0 atIndex:0];
  [encoder setBuffer:keys_a_ offset:0 atIndex:1];
  [encoder setBuffer:tile_offsets_ offset:0 atIndex:2];
  [encoder setBytes:&tiles length:sizeof(tiles) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(tile_count_, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kThreadsPerTile, 1, 1)];
  barrier(encoder);
  return true;
}

id<MTLBuffer> MetalViewPreparationCore::encodeSort(
  id<MTLComputeCommandEncoder> encoder, std::uint32_t key_bits)
{
  if (!ready() || count_ == 0 || key_bits == 0u || key_bits > 32u || key_bits % 8u != 0u) {
    error_ = "Metal sort is not configured, or the key width is not a whole number of bytes";
    return nil;
  }
  id<MTLBuffer> input = keys_a_;
  id<MTLBuffer> output = keys_b_;
  for (std::uint32_t shift = 0; shift < key_bits; shift += 8u) {
    const TileParameters radix{count_, tile_count_, kElementsPerTile, shift};

    [encoder setComputePipelineState:histogram_pipeline_];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:histograms_ offset:0 atIndex:1];
    [encoder setBytes:&radix length:sizeof(radix) atIndex:2];
    [encoder setBuffer:state_ offset:0 atIndex:3];
    [encoder dispatchThreadgroupsWithIndirectBuffer:tile_dispatch_ indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(kThreadsPerTile, 1, 1)];
    barrier(encoder);

    [encoder setComputePipelineState:offset_pipeline_];
    [encoder setBuffer:histograms_ offset:0 atIndex:0];
    [encoder setBuffer:offsets_ offset:0 atIndex:1];
    [encoder setBuffer:totals_ offset:0 atIndex:2];
    [encoder setBytes:&radix length:sizeof(radix) atIndex:3];
    [encoder setBuffer:state_ offset:0 atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(kRadix, 1, 1)
       threadsPerThreadgroup:linearWidth(offset_pipeline_, 64u)];
    barrier(encoder);

    [encoder setComputePipelineState:base_pipeline_];
    [encoder setBuffer:totals_ offset:0 atIndex:0];
    [encoder setBuffer:digit_bases_ offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    barrier(encoder);

    [encoder setComputePipelineState:scatter_pipeline_];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:offsets_ offset:0 atIndex:2];
    [encoder setBuffer:digit_bases_ offset:0 atIndex:3];
    [encoder setBytes:&radix length:sizeof(radix) atIndex:4];
    [encoder setBuffer:state_ offset:0 atIndex:5];
    [encoder dispatchThreadgroupsWithIndirectBuffer:tile_dispatch_ indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(kThreadsPerTile, 1, 1)];
    barrier(encoder);
    std::swap(input, output);
  }
  return input;
}

void MetalViewPreparationCore::encodeShade(
  id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
  const ProjectionParameters & projection)
{
  // Rewrites the compacted pairs in keys_a_, the sort's input, to carry slots.
  [encoder setComputePipelineState:shade_pipeline_];
  [encoder setBuffer:keys_a_ offset:0 atIndex:0];
  [encoder setBuffer:splats_ offset:0 atIndex:1];
  [encoder setBuffer:sh_dc_ offset:0 atIndex:2];
  [encoder setBuffer:sh_rest_ offset:0 atIndex:3];
  [encoder setBuffer:state_ offset:0 atIndex:4];
  [encoder setBuffer:projected_ offset:0 atIndex:5];
  [encoder setBytes:&parameters length:sizeof(parameters) atIndex:6];
  [encoder setBytes:&projection length:sizeof(projection) atIndex:7];
  [encoder setBuffer:slot_splats_ offset:0 atIndex:8];
  [encoder dispatchThreadgroupsWithIndirectBuffer:gather_dispatch_ indirectBufferOffset:0
                            threadsPerThreadgroup:MTLSizeMake(gather_width_, 1, 1)];
  barrier(encoder);
}

void MetalViewPreparationCore::encodeGather(
  id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> sorted, id<MTLBuffer> instances)
{
  [encoder setComputePipelineState:gather_pipeline_];
  [encoder setBuffer:sorted offset:0 atIndex:0];
  [encoder setBuffer:projected_ offset:0 atIndex:1];
  [encoder setBuffer:state_ offset:0 atIndex:4];
  [encoder setBuffer:instances offset:0 atIndex:5];
  [encoder dispatchThreadgroupsWithIndirectBuffer:gather_dispatch_ indirectBufferOffset:0
                            threadsPerThreadgroup:MTLSizeMake(gather_width_, 1, 1)];
}

id<MTLBuffer> MetalViewPreparationCore::encode(
  id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
  const ProjectionParameters & projection, id<MTLBuffer> instances,
  id<MTLBuffer> draw_arguments)
{
  if (!instances ||
    instances.length < static_cast<NSUInteger>(count_) * sizeof(ProjectedInstance))
  {
    error_ = "instance buffer is missing or smaller than the splat count";
    return nil;
  }
  if (!encodeCull(encoder, parameters, draw_arguments)) {
    return nil;
  }
  encodeShade(encoder, parameters, projection);
  id<MTLBuffer> sorted = encodeSort(encoder, 32u);
  if (!sorted) {
    return nil;
  }
  encodeGather(encoder, sorted, instances);
  error_.clear();
  return sorted;
}

void MetalViewPreparationCore::primeSortInput(std::uint32_t visible)
{
  auto * state = static_cast<State *>(state_.contents);
  state->visible = std::min(visible, count_);
  state->visible_tiles = (state->visible + kElementsPerTile - 1u) / kElementsPerTile;
  state->gather_groups = 0;
  state->reserved = 0;
  auto * dispatch = static_cast<std::uint32_t *>(tile_dispatch_.contents);
  dispatch[0] = std::max<std::uint32_t>(state->visible_tiles, 1u);
  dispatch[1] = 1u;
  dispatch[2] = 1u;
}

NSUInteger MetalViewPreparationCore::scratchBytes() const
{
  NSUInteger total = 0;
  for (id<MTLBuffer> buffer : {keys_a_, keys_b_, tile_counts_, tile_offsets_, histograms_,
      offsets_, totals_, digit_bases_, state_, tile_dispatch_, gather_dispatch_, projected_,
      slot_splats_})
  {
    total += buffer ? buffer.length : 0u;
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
