#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "gaussian_splatting_rviz_plugins/splat_view.hpp"

#if !__has_feature(objc_arc)
#error "metal_view_preparation_core.h relies on ARC; compile with -fobjc-arc"
#endif

namespace gaussian_splatting_rviz_plugins
{

// GPU view preparation with no Ogre in it. The display's Ogre adapter and the
// offline verifier both encode through this class, so what the verifier checks
// is the dispatch sequence the display runs.
//
// One preparation: make_depth_keys writes a key, or a rejection mark, per
// splat; compact_count, compact_scan and compact_scatter pack the survivors in
// index order into the front of the key buffer and write the visible count,
// the indirect dispatch sizes and the draw arguments; shade_project colours and
// projects them while they are still in index order, and makes each one's
// compacted slot the sort's value; the radix sorts only those survivors;
// gather_instances permutes the projected survivors into depth order. Nothing
// is read back on the CPU.
class MetalViewPreparationCore
{
public:
  // Mirrors PreparationState in gsplat_prepare.metal. Written once per
  // preparation, so it is safe to read after the command buffer completes.
  struct State
  {
    std::uint32_t visible;
    std::uint32_t visible_tiles;
    std::uint32_t gather_groups;
    std::uint32_t reserved;
  };

  MetalViewPreparationCore(id<MTLDevice> device, const std::string & shader_path);

  bool ready() const;
  const std::string & error() const;

  // Sizes the scratch buffers for `count` splats, drawn with `index_count`
  // indices each. Drops any scene data.
  bool configure(std::uint32_t count, std::uint32_t index_count);
  void clear();

  // Copies scene data in, over the existing buffers when the sizes match.
  // `last_user` is the most recent command buffer that read them: if it is
  // already on the GPU the copy waits for it rather than racing it.
  bool uploadStaticData(
    const SplatRecord * records, std::size_t record_count,
    const float * sh_dc, std::size_t sh_dc_floats,
    const float * sh_rest, std::size_t sh_rest_floats,
    id<MTLCommandBuffer> last_user);

  // The stages, separately so each can be timed in a command buffer of its
  // own. encode() runs all of them on one encoder, which is what the display
  // does. encodeSort() and encode() return the buffer holding the sorted
  // key/slot pairs, or nil on failure; slotSplatBuffer() maps a slot back to
  // its splat index.
  bool encodeCull(
    id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
    id<MTLBuffer> draw_arguments);
  void encodeShade(
    id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
    const ProjectionParameters & projection);
  id<MTLBuffer> encodeSort(id<MTLComputeCommandEncoder> encoder, std::uint32_t key_bits = 32);
  void encodeGather(
    id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> sorted, id<MTLBuffer> instances);
  id<MTLBuffer> encode(
    id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
    const ProjectionParameters & projection, id<MTLBuffer> instances,
    id<MTLBuffer> draw_arguments);

  // For benchmarking the sort alone: makes the first `visible` pairs already
  // in keyBuffer() the sort's input, as the cull stage would have.
  void primeSortInput(std::uint32_t visible);

  std::uint32_t count() const {return count_;}
  id<MTLBuffer> stateBuffer() const {return state_;}
  id<MTLBuffer> keyBuffer() const {return keys_a_;}
  id<MTLBuffer> splatBuffer() const {return splats_;}
  id<MTLBuffer> slotSplatBuffer() const {return slot_splats_;}
  NSUInteger scratchBytes() const;

private:
  enum class Dispatch { kLinear, kTiled, kSimdRanked, kScan };

  id<MTLComputePipelineState> makePipeline(NSString * name, Dispatch dispatch);
  id<MTLBuffer> newBuffer(NSUInteger bytes, const char * label);
  bool fit(id<MTLBuffer> __strong & buffer, NSUInteger bytes, const char * label);
  void setError(const std::string & operation, NSError * error);

  id<MTLDevice> device_ = nil;
  id<MTLLibrary> library_ = nil;
  id<MTLComputePipelineState> keys_pipeline_ = nil;
  id<MTLComputePipelineState> compact_count_pipeline_ = nil;
  id<MTLComputePipelineState> compact_scan_pipeline_ = nil;
  id<MTLComputePipelineState> compact_scatter_pipeline_ = nil;
  id<MTLComputePipelineState> histogram_pipeline_ = nil;
  id<MTLComputePipelineState> offset_pipeline_ = nil;
  id<MTLComputePipelineState> base_pipeline_ = nil;
  id<MTLComputePipelineState> scatter_pipeline_ = nil;
  id<MTLComputePipelineState> shade_pipeline_ = nil;
  id<MTLComputePipelineState> gather_pipeline_ = nil;

  std::uint32_t count_ = 0;
  std::uint32_t tile_count_ = 0;
  std::uint32_t index_count_ = 0;
  std::uint32_t gather_width_ = 0;

  id<MTLBuffer> splats_ = nil;
  id<MTLBuffer> sh_dc_ = nil;
  id<MTLBuffer> sh_rest_ = nil;
  id<MTLBuffer> keys_a_ = nil;
  id<MTLBuffer> keys_b_ = nil;
  id<MTLBuffer> tile_counts_ = nil;
  id<MTLBuffer> tile_offsets_ = nil;
  id<MTLBuffer> histograms_ = nil;
  id<MTLBuffer> offsets_ = nil;
  id<MTLBuffer> totals_ = nil;
  id<MTLBuffer> digit_bases_ = nil;
  id<MTLBuffer> state_ = nil;
  id<MTLBuffer> tile_dispatch_ = nil;
  id<MTLBuffer> gather_dispatch_ = nil;
  // Each survivor projected, in compacted slot order, for the gather to permute,
  // and the splat index behind each slot.
  id<MTLBuffer> projected_ = nil;
  id<MTLBuffer> slot_splats_ = nil;
  std::string error_;
};

}  // namespace gaussian_splatting_rviz_plugins
