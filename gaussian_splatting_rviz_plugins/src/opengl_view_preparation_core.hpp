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
// One preparation: MAKE_DEPTH_KEYS writes a key, or a rejection mark, per
// splat; COMPACT_COUNT, COMPACT_SCAN and COMPACT_SCATTER pack the survivors in
// index order and write the visible count, the indirect dispatch sizes and the
// draw arguments; SHADE_PROJECT colours and projects them while they are still
// in index order, and makes each one's compacted slot the sort's value; the
// radix sorts only those survivors; GATHER permutes the projected survivors
// into depth order. Nothing is read back on the CPU but the visible count, and
// that without waiting.
class OpenGlViewPreparationCore
{
public:
  // Mirrors PreparationState in gsplat_prepare.comp.
  struct State
  {
    std::uint32_t visible;
    std::uint32_t visible_tiles;
    std::uint32_t gather_groups;
    std::uint32_t written;
    std::uint32_t clear_end;
  };

  // GPU time between the stage boundaries of the latest preparation, and
  // within culling and the sort; the radix steps are summed over the passes.
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
  // indices each. Drops any scene data.
  bool configure(std::uint32_t count, std::uint32_t index_count);
  void clear();

  bool uploadStaticData(
    const SplatRecord * records, std::size_t record_count,
    const float * sh_dc, std::size_t sh_dc_floats,
    const float * sh_rest, std::size_t sh_rest_floats);

  // One preparation. The draw stream goes to `instances`, a buffer of at least
  // the configured count of ProjectedInstance; the indexed indirect draw
  // arguments to `draw_arguments`, or to drawArgumentsBuffer() when that is 0.
  // With `clear_tail`, instance slots past the visible count that an earlier
  // preparation filled are emptied, for a renderer that draws every instance.
  bool encode(
    const ViewParameters & parameters, const ProjectionParameters & projection,
    GLuint instances, GLuint draw_arguments, bool clear_tail);

  // Times every preparation's stages with timestamp queries. Reading them waits
  // for the GPU, so this serialises each preparation with the CPU.
  void setProfiling(bool enabled);
  bool profiling() const {return profiling_;}
  const Timings & lastTimings() const {return timings_;}

  // The visible count of the latest preparation known to have finished; -1
  // until one has. Polled at each encode() without waiting.
  std::int64_t lastVisibleCount() const {return last_visible_;}

  std::uint32_t count() const {return count_;}
  GLuint stateBuffer() const {return state_;}
  // The key/slot pairs the latest preparation left sorted.
  GLuint sortedBuffer() const {return keys_a_;}
  GLuint drawArgumentsBuffer() const {return draw_arguments_;}
  std::size_t scratchBytes() const;

private:
  struct Readback
  {
    GLuint buffer = 0u;
    GLsync fence = nullptr;
  };

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
  std::array<GLuint, 11> programs_{};
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
  GLuint keys_a_ = 0u;
  GLuint keys_b_ = 0u;
  GLuint tile_counts_ = 0u;
  GLuint tile_offsets_ = 0u;
  GLuint histograms_ = 0u;
  GLuint offsets_ = 0u;
  GLuint totals_ = 0u;
  GLuint digit_bases_ = 0u;
  GLuint state_ = 0u;
  GLuint tile_dispatch_ = 0u;
  GLuint gather_dispatch_ = 0u;
  GLuint clear_dispatch_ = 0u;
  // Each survivor projected, in compacted slot order, for the gather to permute.
  GLuint projected_ = 0u;
  // Draw arguments for a caller that has no buffer of its own for them.
  GLuint draw_arguments_ = 0u;
  GLuint view_parameters_ = 0u;
  GLuint projection_parameters_ = 0u;
  std::array<Readback, 3> readbacks_{};
  std::size_t readback_cursor_ = 0u;
  std::int64_t last_visible_ = -1;
};

}  // namespace gaussian_splatting_rviz_plugins
