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

// One survivor of a Metal draw order: its colour and the splat it is. Leaner
// than the OpenGL path's DrawnSplat, which carries the record too; see
// OrderedSplat in gsplat_prepare.metal, which this mirrors.
struct OrderedSplat
{
  float colour[3];
  std::uint32_t splat;
};
static_assert(sizeof(OrderedSplat) == 16, "one quadruple");

// GPU view preparation with no Ogre in it. The display's Ogre adapter and the
// offline verifier both encode through this class, so what the verifier checks
// is the dispatch sequence the display runs.
//
// Preparation builds a draw order and draws from it. Building an order for a
// view: make_depth_keys writes a key, or a rejection mark, per splat;
// compact_count, compact_scan and compact_scatter pack the survivors in index
// order and count them; shade_slots colours them while they are still in
// index order, and makes each one's compacted slot the sort's value; the radix
// sorts only those survivors; finish_order lays the survivors and their
// colours out in draw order. Drawing: reproject_gather projects them in draw
// order for the frame's camera, and writes the indirect draw arguments;
// recolour_order can first evaluate their colours again for the frame's
// camera. Nothing is read back on the CPU.
//
// The next order can be built while the last is drawn from: encode() builds
// one for its view and draws from it in the same frame, encodeIncremental()
// spreads a build over five frames - culling, two halves of the colour, and
// two halves of the sort - drawing every frame from the newest complete order.
// Only the orders' counts are kept twice; the survivors in draw order are
// replaced when a build completes, by which point nothing will read the
// previous ones again.
class MetalViewPreparationCore
{
public:
  // Mirrors PreparationState in gsplat_prepare.metal: an order's survivor count
  // and the dispatch sizes it implies.
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
  // indices each. Drops any scene data and both orders.
  bool configure(std::uint32_t count, std::uint32_t index_count);
  void clear();

  // Copies scene data in, over the existing buffers when the sizes match.
  // `last_user` is the most recent command buffer that read them: if it is
  // already on the GPU the copy waits for it rather than racing it. Both orders
  // are dropped, since their colours came from the old data.
  bool uploadStaticData(
    const SplatRecord * records, std::size_t record_count,
    const float * sh_dc, std::size_t sh_dc_floats,
    const float * sh_rest, std::size_t sh_rest_floats,
    id<MTLCommandBuffer> last_user);

  // One preparation with an order built for `parameters` in this frame: the
  // same draw stream as the CPU path gives for that view.
  bool encode(
    id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
    const ProjectionParameters & projection, id<MTLBuffer> instances,
    id<MTLBuffer> draw_arguments);
  // One of five steps of building an order - started for `parameters` if
  // none is under way - then the draw stream from the newest complete order,
  // projected with `projection`, and with its colours evaluated again for
  // `parameters` as setColourRefresh() asks. With no complete order yet, it is
  // encode().
  bool encodeIncremental(
    id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
    const ProjectionParameters & projection, id<MTLBuffer> instances,
    id<MTLBuffer> draw_arguments);
  // How incremental preparation refreshes colours: 0 not at all, so they lag
  // with the order; 1 every colour every frame; 2 half of them a frame in
  // turn. Orders completed while it is 0 cannot be refreshed, which saves
  // their completion some writes.
  void setColourRefresh(std::uint32_t parts) {colour_parts_ = parts;}
  // The build step encodeIncremental() takes, without the draw: the step it
  // ran, or kFailed. Needs a complete order to draw from meanwhile.
  enum class BuildStep { kFailed, kCull, kShade, kSortLow, kSortHigh };
  BuildStep encodeBuildStep(
    id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters);
  bool hasOrder() const {return current_ >= 0;}
  bool building() const {return building_ >= 0;}

  // The stages, separately so each can be timed in a command buffer of its
  // own; encode() is all four. encodeCull() starts an order; encodeSort()
  // completes it, and returns false on failure.
  bool encodeCull(id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters);
  // All the survivors, or part `index` of `count` equal parts of them.
  void encodeShade(
    id<MTLComputeCommandEncoder> encoder, std::uint32_t index = 0, std::uint32_t count = 1);
  bool encodeSort(id<MTLComputeCommandEncoder> encoder, std::uint32_t key_bits = 32);
  bool encodeGather(
    id<MTLComputeCommandEncoder> encoder, const ProjectionParameters & projection,
    id<MTLBuffer> instances, id<MTLBuffer> draw_arguments);
  // The newest complete order's colours evaluated again for `parameters`:
  // all of them, or part `index` of `count` equal parts.
  void encodeRecolour(
    id<MTLComputeCommandEncoder> encoder, const ViewParameters & parameters,
    std::uint32_t index = 0, std::uint32_t count = 1);

  // For benchmarking the sort alone: makes the first `visible` pairs already
  // in keyBuffer() the input of a build that encodeSort() completes.
  void primeSortInput(std::uint32_t visible);

  std::uint32_t count() const {return count_;}
  // The newest complete order: its state, and its survivors in draw order as
  // OrderedSplat.
  id<MTLBuffer> stateBuffer() const;
  id<MTLBuffer> drawnBuffer() const {return drawn_;}
  // The key/slot pairs the sort works on.
  id<MTLBuffer> keyBuffer() const {return pairs_;}
  id<MTLBuffer> splatBuffer() const {return splats_;}
  NSUInteger scratchBytes() const;

private:
  enum class Dispatch { kLinear, kTiled, kSimdRanked, kScan };

  // What is kept per order: its survivor count and the dispatch sizes it
  // implies.
  struct Order
  {
    id<MTLBuffer> state = nil;
    id<MTLBuffer> tile_dispatch = nil;
    id<MTLBuffer> gather_dispatch = nil;
  };

  // Radix passes over the pairs for key bits [first_bit, end_bit), an even
  // number of passes so that the pairs end where they started.
  void encodeRadix(
    id<MTLComputeCommandEncoder> encoder, const Order & order, std::uint32_t first_bit,
    std::uint32_t end_bit);
  // finish_order, then the order under way becomes the one drawn from.
  void completeBuild(id<MTLComputeCommandEncoder> encoder);

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
  id<MTLComputePipelineState> finish_pipeline_ = nil;
  id<MTLComputePipelineState> recolour_pipeline_ = nil;

  std::uint32_t count_ = 0;
  std::uint32_t tile_count_ = 0;
  std::uint32_t index_count_ = 0;
  std::uint32_t gather_width_ = 0;

  id<MTLBuffer> splats_ = nil;
  id<MTLBuffer> sh_dc_ = nil;
  id<MTLBuffer> sh_rest_ = nil;
  // Each splat's key in index order for compaction, then the radix's other
  // buffer; free between build steps.
  id<MTLBuffer> staged_ = nil;
  // The order under way's compacted key/slot pairs, sorted in place.
  id<MTLBuffer> pairs_ = nil;
  // The order under way's survivors by compacted slot, and the newest complete
  // order's in draw order, as OrderedSplat.
  id<MTLBuffer> slots_ = nil;
  id<MTLBuffer> drawn_ = nil;
  // For each slot of the newest complete order, its splat and draw position,
  // by which recolour_order refreshes the colours in drawn_.
  id<MTLBuffer> slot_draws_ = nil;
  std::uint32_t colour_parts_ = 0;
  std::uint32_t colour_frame_ = 0;
  // Whether the newest complete order kept what recolour_order needs.
  bool refreshable_ = false;
  id<MTLBuffer> tile_counts_ = nil;
  id<MTLBuffer> tile_offsets_ = nil;
  id<MTLBuffer> histograms_ = nil;
  id<MTLBuffer> offsets_ = nil;
  id<MTLBuffer> totals_ = nil;
  id<MTLBuffer> digit_bases_ = nil;

  Order orders_[2];
  int current_ = -1;
  int building_ = -1;
  int build_step_ = 0;
  // The view the order under way is built for.
  ViewParameters build_view_{};
  std::string error_;
};

}  // namespace gaussian_splatting_rviz_plugins
