#pragma once

// GLEW before any other OpenGL header. In the plug-in its entry points resolve
// to the GLEW inside Ogre's GL render system; the offline verifier links GLEW.
#include <GL/glew.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "gaussian_splatting_rviz_plugins/splat_view.hpp"

namespace gaussian_splatting_rviz_plugins
{

// GPU view preparation in OpenGL 4.3 compute, with no Ogre in it. The display's
// Ogre adapter and the offline verifier both drive this class, so what the
// verifier checks is the dispatch sequence the display runs. Every call needs
// the same OpenGL 4.3 context current, construction and destruction included.
//
// Preparation builds a draw order and draws from it, as MetalViewPreparationCore
// does. Building an order for a view: MAKE_DEPTH_KEYS writes a key, or a
// rejection mark, per splat; COMPACT_COUNT, COMPACT_SCAN and COMPACT_SCATTER
// pack the survivors in index order and count them; SHADE_SLOTS colours them
// and copies their records while they are still in index order, and makes each
// one's compacted slot the sort's value; the radix sorts only those survivors;
// FINISH_ORDER lays the survivors and their colours out in draw order.
// Drawing: PUBLISH writes the draw arguments, RECOLOUR_ORDER can evaluate the
// colours again for the frame's camera, REPROJECT_GATHER projects the
// survivors in sequence for the frame's camera, and CLEAR_TAIL
// empties stale instance slots for a renderer that draws every instance.
// Nothing is read back on the CPU but the visible count, and that without
// waiting.
//
// The next order can be built while the last is drawn from: encode() builds
// one for its view and draws from it; encodeIncremental() spreads a build over
// five calls - culling, two halves of the colour, and two halves of the sort -
// drawing every call from the newest complete order. Only the orders' counts
// are kept twice.
class OpenGlViewPreparationCore
{
public:
  // Mirrors PreparationState in gsplat_prepare.comp: an order's survivor count
  // and the dispatch sizes it implies.
  struct State
  {
    std::uint32_t visible;
    std::uint32_t visible_tiles;
    std::uint32_t gather_groups;
    std::uint32_t reserved;
  };

  // GPU time between the stage boundaries of the latest preparation, and
  // within culling and the sort; the radix steps are summed over the passes.
  // An incremental preparation times only the steps it ran.
  struct Timings
  {
    double cull_ms = 0.0;
    double shade_ms = 0.0;
    double sort_ms = 0.0;
    double gather_ms = 0.0;

    double keys_ms = 0.0;
    double compact_count_ms = 0.0;
    double compact_scan_ms = 0.0;
    double compact_scatter_ms = 0.0;
    double radix_histogram_ms = 0.0;
    double radix_scan_ms = 0.0;
    double radix_scatter_ms = 0.0;
  };

  // With `allow_subgroups`, the scatters rank lanes with subgroup ballots where
  // the driver supports them; see subgroupBallot().
  explicit OpenGlViewPreparationCore(
    const std::string & shader_source, bool allow_subgroups = true);
  ~OpenGlViewPreparationCore();
  OpenGlViewPreparationCore(const OpenGlViewPreparationCore &) = delete;
  OpenGlViewPreparationCore & operator=(const OpenGlViewPreparationCore &) = delete;

  bool ready() const;
  const std::string & error() const;
  // Whether the scatter stages were compiled to rank lanes with subgroup
  // ballots rather than through shared memory.
  bool subgroupBallot() const {return subgroup_ballot_;}

  // Sizes the scratch buffers for `count` splats, drawn with `index_count`
  // indices each. Drops any scene data and both orders.
  bool configure(std::uint32_t count, std::uint32_t index_count);
  void clear();

  // Copies scene data in. Both orders are dropped, since their colours came
  // from the old data.
  bool uploadStaticData(
    const SplatRecord * records, std::size_t record_count,
    const float * sh_dc, std::size_t sh_dc_floats,
    const float * sh_rest, std::size_t sh_rest_floats);

  // One preparation with an order built for `parameters` now: the same draw
  // stream as the CPU path gives for that view. The draw stream goes to
  // `instances`, a buffer of at least the configured count of
  // ProjectedInstance; the indexed indirect draw arguments to
  // `draw_arguments`, or to drawArgumentsBuffer() when that is 0. With
  // `clear_tail`, instance slots past the visible count that an earlier
  // preparation filled are emptied, for a renderer that draws every instance.
  bool encode(
    const ViewParameters & parameters, const ProjectionParameters & projection,
    GLuint instances, GLuint draw_arguments, bool clear_tail);
  // One of five steps of building an order - started for `parameters` if
  // none is under way - then the draw stream from the newest complete order,
  // projected with `projection`, and with its colours evaluated again for
  // `parameters` as setColourRefresh() asks. With no complete order yet, it is
  // encode().
  bool encodeIncremental(
    const ViewParameters & parameters, const ProjectionParameters & projection,
    GLuint instances, GLuint draw_arguments, bool clear_tail);
  // How incremental preparation refreshes colours: 0 not at all, so they lag
  // with the order; 1 every colour every frame; 2 half of them a frame in
  // turn. Orders completed while it is 0 cannot be refreshed, which saves
  // their completion some writes.
  void setColourRefresh(std::uint32_t parts) {colour_parts_ = parts;}
  bool hasOrder() const {return current_ >= 0;}
  bool building() const {return building_ >= 0;}

  // Times every preparation's stages with timestamp queries. Reading them waits
  // for the GPU, so this serialises each preparation with the CPU.
  void setProfiling(bool enabled);
  bool profiling() const {return profiling_;}
  const Timings & lastTimings() const {return timings_;}

  // The visible count of the latest preparation known to have finished; -1
  // until one has. Polled at each encode without waiting.
  std::int64_t lastVisibleCount() const {return last_visible_;}

  std::uint32_t count() const {return count_;}
  // The newest complete order's state, and its survivors in draw order as
  // DrawnSplat.
  GLuint stateBuffer() const;
  GLuint drawnBuffer() const {return drawn_;}
  GLuint drawArgumentsBuffer() const {return draw_arguments_;}
  std::size_t scratchBytes() const;

private:
  struct Readback
  {
    GLuint buffer = 0u;
    GLsync fence = nullptr;
  };

  // What is kept per order: its survivor count and the dispatch sizes it
  // implies.
  struct Order
  {
    GLuint state = 0u;
    GLuint tile_dispatch = 0u;
    GLuint gather_dispatch = 0u;
  };

  // Which parts of a build one preparation runs: culling, part `shade_part`
  // of `shade_parts` of the colour (none when 0), and the sort's key bits
  // [first_bit, end_bit).
  struct Work
  {
    bool cull = false;
    std::uint32_t shade_part = 0u;
    std::uint32_t shade_parts = 0u;
    std::uint32_t first_bit = 0u;
    std::uint32_t end_bit = 0u;
  };

  bool prepare(
    const ViewParameters & parameters, const ProjectionParameters & projection,
    GLuint instances, GLuint draw_arguments, bool clear_tail, bool incremental);
  bool checkScene(const ViewParameters & parameters);
  GLuint compile(const char * stage);
  GLuint newBuffer(std::size_t bytes, GLenum usage);
  bool uploadBuffer(GLuint & buffer, const void * data, std::size_t bytes, GLenum usage);
  bool fitsStorageBlock(std::size_t bytes) const;
  void use(GLuint program);
  void uniform(const char * name, std::uint32_t value);
  void mark(std::size_t boundary);
  void readTimings();
  void pollReadbacks();
  void queueReadback();

  std::string shader_source_;
  std::string error_;
  // keys, compact count, compact scan, compact scatter, radix histogram,
  // offsets, bases, scatter, shade, finish, publish, reproject, clear tail,
  // recolour
  std::array<GLuint, 14> programs_{};
  bool profiling_ = false;
  bool subgroup_ballot_ = false;
  // Stage boundaries 0-4, then three within culling and four per radix pass.
  std::array<GLuint, 24> queries_{};
  Timings timings_;
  GLuint current_program_ = 0u;
  GLint64 max_storage_block_bytes_ = 0;

  std::uint32_t count_ = 0u;
  std::uint32_t tile_count_ = 0u;
  std::uint32_t index_count_ = 0u;
  std::size_t scratch_bytes_ = 0u;

  GLuint splats_ = 0u;
  GLuint sh_dc_ = 0u;
  GLuint sh_rest_ = 0u;
  // Each splat's key in index order for compaction, then the radix's other
  // buffer; free between build steps.
  GLuint staged_ = 0u;
  GLuint tile_counts_ = 0u;
  GLuint tile_offsets_ = 0u;
  GLuint histograms_ = 0u;
  GLuint offsets_ = 0u;
  GLuint totals_ = 0u;
  GLuint digit_bases_ = 0u;
  // The order under way's compacted key/slot pairs, sorted in place, and its
  // survivors by slot; the newest complete order's survivors in draw order.
  GLuint pairs_ = 0u;
  GLuint slots_ = 0u;
  GLuint drawn_ = 0u;
  // The newest complete order's colours in draw order, and for each slot its
  // splat and draw position, by which RECOLOUR_ORDER refreshes them.
  GLuint colours_ = 0u;
  GLuint slot_draws_ = 0u;
  std::uint32_t colour_parts_ = 0u;
  std::uint32_t colour_frame_ = 0u;
  // Whether the newest complete order kept what RECOLOUR_ORDER needs.
  bool refreshable_ = false;
  // The instance buffer's visible count, filled high-water mark and the end of
  // the range CLEAR_TAIL empties; written by PUBLISH.
  GLuint instance_state_ = 0u;
  GLuint clear_dispatch_ = 0u;
  // Draw arguments for a caller that has no buffer of its own for them.
  GLuint draw_arguments_ = 0u;
  // The view the order under way is built for, the frame's view, and the
  // frame's projection.
  GLuint view_parameters_ = 0u;
  GLuint frame_view_parameters_ = 0u;
  GLuint projection_parameters_ = 0u;

  std::array<Order, 2> orders_{};
  int current_ = -1;
  int building_ = -1;
  int build_step_ = 0;
  std::array<Readback, 3> readbacks_{};
  std::size_t readback_cursor_ = 0u;
  std::int64_t last_visible_ = -1;
};

}  // namespace gaussian_splatting_rviz_plugins
