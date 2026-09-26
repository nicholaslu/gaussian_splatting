#include "opengl_view_preparation_core.hpp"

#include <algorithm>
#include <sstream>
#include <utility>

#include <dlfcn.h>

namespace gaussian_splatting_rviz_plugins
{
namespace
{

constexpr std::uint32_t kElementsPerTile = 1024u;
constexpr std::uint32_t kGatherWidth = 256u;
constexpr std::uint32_t kRadix = 256u;

// Timestamp slots: the stage boundaries, then the steps inside culling and
// inside each of the four radix passes.
constexpr std::size_t kMarkAfterKeys = 5u;
constexpr std::size_t kMarkAfterCompactCount = 6u;
constexpr std::size_t kMarkAfterCompactScan = 7u;
// + 4 * pass + {histogram, offsets, bases, scatter}, each marking its end.
constexpr std::size_t kMarkRadix = 8u;

// Ogre's GL render system carries its own GLEW and exports its function
// pointers, but that GLEW stops short of OpenGL 4.3: nothing provides
// __glewDispatchCompute, __glewDispatchComputeIndirect or __GLEW_VERSION_4_3,
// so a plug-in reaching them through GLEW's macros fails to load. Linking a
// second GLEW would not help, since its glewInit would resolve to Ogre's and
// leave those pointers null. The two entry points come from the GL library
// instead, which dispatches to the current context just as GLEW does.
struct ComputeFunctions
{
  void (*dispatch)(GLuint, GLuint, GLuint) = nullptr;
  void (*dispatch_indirect)(GLintptr) = nullptr;
};

const ComputeFunctions & computeFunctions()
{
  static const ComputeFunctions functions = [] {
      ComputeFunctions resolved;
      for (const char * name : {"libGL.so.1", "libOpenGL.so.0"}) {
        // Already loaded by Ogre; this only takes a reference to look into.
        void * library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!library) {
          continue;
        }
        resolved.dispatch = reinterpret_cast<void (*)(GLuint, GLuint, GLuint)>(
          dlsym(library, "glDispatchCompute"));
        resolved.dispatch_indirect = reinterpret_cast<void (*)(GLintptr)>(
          dlsym(library, "glDispatchComputeIndirect"));
        if (resolved.dispatch && resolved.dispatch_indirect) {
          break;
        }
        resolved = ComputeFunctions{};
      }
      return resolved;
    }();
  return functions;
}

// Profiling times each stage with timestamp queries, taken from the GL library
// for the same reason as the dispatch entry points above.
struct TimerFunctions
{
  void (*gen_queries)(GLsizei, GLuint *) = nullptr;
  void (*delete_queries)(GLsizei, const GLuint *) = nullptr;
  void (*query_counter)(GLuint, GLenum) = nullptr;
  void (*get_query_result)(GLuint, GLenum, GLuint64 *) = nullptr;

  bool resolved() const
  {
    return gen_queries && delete_queries && query_counter && get_query_result;
  }
};

const TimerFunctions & timerFunctions()
{
  static const TimerFunctions functions = [] {
      TimerFunctions found;
      for (const char * name : {"libGL.so.1", "libOpenGL.so.0"}) {
        void * library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!library) {
          continue;
        }
        found.gen_queries = reinterpret_cast<void (*)(GLsizei, GLuint *)>(
          dlsym(library, "glGenQueries"));
        found.delete_queries = reinterpret_cast<void (*)(GLsizei, const GLuint *)>(
          dlsym(library, "glDeleteQueries"));
        found.query_counter = reinterpret_cast<void (*)(GLuint, GLenum)>(
          dlsym(library, "glQueryCounter"));
        found.get_query_result = reinterpret_cast<void (*)(GLuint, GLenum, GLuint64 *)>(
          dlsym(library, "glGetQueryObjectui64v"));
        if (found.resolved()) {
          break;
        }
        found = TimerFunctions{};
      }
      return found;
    }();
  return functions;
}

#undef glDispatchCompute
#define glDispatchCompute computeFunctions().dispatch
#undef glDispatchComputeIndirect
#define glDispatchComputeIndirect computeFunctions().dispatch_indirect

struct DrawIndexedArguments
{
  std::uint32_t index_count;
  std::uint32_t instance_count;
  std::uint32_t index_start;
  std::uint32_t base_vertex;
  std::uint32_t base_instance;
};
static_assert(sizeof(DrawIndexedArguments) == 20, "OpenGL indexed indirect argument layout");

struct DispatchArguments
{
  std::uint32_t groups_x;
  std::uint32_t groups_y;
  std::uint32_t groups_z;
};
static_assert(sizeof(DispatchArguments) == 12, "OpenGL dispatch indirect argument layout");

static_assert(
  sizeof(OpenGlViewPreparationCore::State) == 20, "OpenGL preparation state layout");

std::string shaderLog(GLuint object, bool program)
{
  GLint length = 0;
  if (program) {
    glGetProgramiv(object, GL_INFO_LOG_LENGTH, &length);
  } else {
    glGetShaderiv(object, GL_INFO_LOG_LENGTH, &length);
  }
  if (length <= 1) {
    return {};
  }
  std::string log(static_cast<std::size_t>(length), '\0');
  if (program) {
    glGetProgramInfoLog(object, length, nullptr, log.data());
  } else {
    glGetShaderInfoLog(object, length, nullptr, log.data());
  }
  while (!log.empty() && log.back() == '\0') {
    log.pop_back();
  }
  return log;
}

// GL_KHR_shader_subgroup, which older GLEW headers lack.
constexpr GLenum kSubgroupSize = 0x9532;
constexpr GLenum kSubgroupSupportedStages = 0x9533;
constexpr GLenum kSubgroupSupportedFeatures = 0x9534;
constexpr GLint kSubgroupFeatureBallot = 0x8;
constexpr GLint kComputeShaderStage = 0x20;

bool hasExtension(const char * name)
{
  GLint count = 0;
  glGetIntegerv(GL_NUM_EXTENSIONS, &count);
  for (GLint i = 0; i < count; ++i) {
    const GLubyte * extension = glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i));
    if (extension && std::string(reinterpret_cast<const char *>(extension)) == name) {
      return true;
    }
  }
  return false;
}

// Subgroup ballots can stand in for the shared-memory lane ranking when the
// driver offers them in compute shaders with 32-lane subgroups, the width of a
// scatter workgroup. The shader still checks the subgroup it actually gets.
bool subgroupBallotUsable()
{
  if (!hasExtension("GL_KHR_shader_subgroup")) {
    return false;
  }
  GLint size = 0;
  GLint stages = 0;
  GLint features = 0;
  glGetIntegerv(kSubgroupSize, &size);
  glGetIntegerv(kSubgroupSupportedStages, &stages);
  glGetIntegerv(kSubgroupSupportedFeatures, &features);
  glGetError();
  return size == 32 && (stages & kComputeShaderStage) != 0 &&
         (features & kSubgroupFeatureBallot) != 0;
}

GLuint groups(std::uint32_t count, std::uint32_t width)
{
  return (count + width - 1u) / width;
}

void bind(GLuint binding, GLuint buffer)
{
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffer);
}

void updateBuffer(GLuint buffer, const void * data, std::size_t bytes)
{
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
  glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(bytes), data);
}

void storageBarrier()
{
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void dispatchIndirect(GLuint buffer)
{
  glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buffer);
  glDispatchComputeIndirect(0);
}

}  // namespace

OpenGlViewPreparationCore::OpenGlViewPreparationCore(
  const std::string & shader_source, bool allow_subgroups)
: shader_source_(shader_source)
{
  GLint major = 0;
  GLint minor = 0;
  glGetIntegerv(GL_MAJOR_VERSION, &major);
  glGetIntegerv(GL_MINOR_VERSION, &minor);
  if (major < 4 || (major == 4 && minor < 3)) {
    const GLubyte * version = glGetString(GL_VERSION);
    error_ = "OpenGL 4.3 compute is required; active context is " +
      std::string(version ? reinterpret_cast<const char *>(version) : "unknown");
    return;
  }
  if (!computeFunctions().dispatch || !computeFunctions().dispatch_indirect) {
    error_ = "could not resolve glDispatchCompute from libGL.so.1 or libOpenGL.so.0";
    return;
  }
  GLint local_size = 0;
  GLint storage_blocks = 0;
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &local_size);
  glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &storage_blocks);
  glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &max_storage_block_bytes_);
  if (local_size < static_cast<GLint>(kGatherWidth) || storage_blocks < 8) {
    error_ = "OpenGL 4.3 context lacks the compute workgroup or SSBO limits required "
      "by Gaussian splat preparation";
    return;
  }
  if (shader_source_.empty()) {
    error_ = "no OpenGL preparation shader source";
    return;
  }

  subgroup_ballot_ = allow_subgroups && subgroupBallotUsable();
  programs_[0] = compile("GSPLAT_STAGE_MAKE_DEPTH_KEYS");
  programs_[1] = compile("GSPLAT_STAGE_COMPACT_COUNT");
  programs_[2] = compile("GSPLAT_STAGE_COMPACT_SCAN");
  programs_[3] = compile("GSPLAT_STAGE_COMPACT_SCATTER");
  programs_[4] = compile("GSPLAT_STAGE_RADIX_HISTOGRAM");
  programs_[5] = compile("GSPLAT_STAGE_RADIX_SCAN_OFFSETS");
  programs_[6] = compile("GSPLAT_STAGE_RADIX_SCAN_BASES");
  programs_[7] = compile("GSPLAT_STAGE_RADIX_SCATTER");
  programs_[8] = compile("GSPLAT_STAGE_GATHER");
  programs_[9] = compile("GSPLAT_STAGE_CLEAR_TAIL");
  programs_[10] = compile("GSPLAT_STAGE_SHADE_PROJECT");

  // Optional: a failure here leaves preparation working and rasterReady() false.
  if (ready()) {
    const char * raster_stages[6] = {
      "GSPLAT_STAGE_RASTER_BIN_COUNT", "GSPLAT_STAGE_RASTER_BIN_SCAN",
      "GSPLAT_STAGE_RASTER_BIN_EMIT", "GSPLAT_STAGE_RASTER_CLEAR_RANGES",
      "GSPLAT_STAGE_RASTER_TILE_RANGES", "GSPLAT_STAGE_RASTER_TILES"};
    for (std::size_t i = 0; i < raster_programs_.size(); ++i) {
      raster_programs_[i] = compile(raster_stages[i]);
    }
    raster_error_ = error_;
    error_.clear();
  }
}

OpenGlViewPreparationCore::~OpenGlViewPreparationCore()
{
  clear();
  setProfiling(false);
  for (GLuint program : programs_) {
    if (program != 0u) {
      glDeleteProgram(program);
    }
  }
  for (GLuint program : raster_programs_) {
    if (program != 0u) {
      glDeleteProgram(program);
    }
  }
}

bool OpenGlViewPreparationCore::ready() const
{
  return std::all_of(programs_.begin(), programs_.end(), [](GLuint program) {
             return program != 0u;
           });
}

const std::string & OpenGlViewPreparationCore::error() const
{
  return error_;
}

void OpenGlViewPreparationCore::setProfiling(bool enabled)
{
  enabled = enabled && timerFunctions().resolved();
  if (enabled == profiling_) {
    return;
  }
  if (enabled) {
    timerFunctions().gen_queries(static_cast<GLsizei>(queries_.size()), queries_.data());
  } else {
    timerFunctions().delete_queries(static_cast<GLsizei>(queries_.size()), queries_.data());
    queries_ = {};
  }
  profiling_ = enabled;
  timings_ = Timings{};
}

bool OpenGlViewPreparationCore::configure(std::uint32_t count, std::uint32_t index_count)
{
  clear();
  if (!ready()) {
    return false;
  }
  if (count == 0) {
    error_ = "OpenGL preparation requires a non-zero splat count";
    return false;
  }

  count_ = count;
  index_count_ = index_count;
  tile_count_ = (count_ + kElementsPerTile - 1u) / kElementsPerTile;
  const std::size_t pair_bytes = std::size_t(count) * sizeof(std::uint32_t) * 2u;
  const std::size_t tile_bytes = static_cast<std::size_t>(tile_count_) * sizeof(std::uint32_t);
  const std::size_t table_bytes = tile_bytes * kRadix;
  const std::size_t projected_bytes = std::size_t(count) * sizeof(ProjectedInstance);
  const std::array<std::size_t, 8> storage_sizes = {
    pair_bytes, pair_bytes, tile_bytes, tile_bytes,
    table_bytes, table_bytes, kRadix * sizeof(std::uint32_t), projected_bytes};
  for (std::size_t bytes : storage_sizes) {
    if (!fitsStorageBlock(bytes)) {
      error_ = "scene requires an SSBO larger than GL_MAX_SHADER_STORAGE_BLOCK_SIZE";
      clear();
      return false;
    }
  }

  keys_a_ = newBuffer(pair_bytes, GL_DYNAMIC_COPY);
  keys_b_ = newBuffer(pair_bytes, GL_DYNAMIC_COPY);
  tile_counts_ = newBuffer(tile_bytes, GL_DYNAMIC_COPY);
  tile_offsets_ = newBuffer(tile_bytes, GL_DYNAMIC_COPY);
  histograms_ = newBuffer(table_bytes, GL_DYNAMIC_COPY);
  offsets_ = newBuffer(table_bytes, GL_DYNAMIC_COPY);
  totals_ = newBuffer(kRadix * sizeof(std::uint32_t), GL_DYNAMIC_COPY);
  digit_bases_ = newBuffer(kRadix * sizeof(std::uint32_t), GL_DYNAMIC_COPY);
  state_ = newBuffer(sizeof(State), GL_DYNAMIC_COPY);
  tile_dispatch_ = newBuffer(sizeof(DispatchArguments), GL_DYNAMIC_COPY);
  gather_dispatch_ = newBuffer(sizeof(DispatchArguments), GL_DYNAMIC_COPY);
  clear_dispatch_ = newBuffer(sizeof(DispatchArguments), GL_DYNAMIC_COPY);
  projected_ = newBuffer(projected_bytes, GL_DYNAMIC_COPY);
  draw_arguments_ = newBuffer(sizeof(DrawIndexedArguments), GL_DYNAMIC_COPY);
  view_parameters_ = newBuffer(sizeof(ViewParameters), GL_STREAM_DRAW);
  projection_parameters_ = newBuffer(sizeof(ProjectionParameters), GL_STREAM_DRAW);
  for (Readback & readback : readbacks_) {
    readback.buffer = newBuffer(sizeof(std::uint32_t), GL_STREAM_READ);
  }
  if (!keys_a_ || !keys_b_ || !tile_counts_ || !tile_offsets_ || !histograms_ ||
    !offsets_ || !totals_ || !digit_bases_ || !state_ || !tile_dispatch_ ||
    !gather_dispatch_ || !clear_dispatch_ || !projected_ || !draw_arguments_ ||
    !view_parameters_ || !projection_parameters_ || !readbacks_[0].buffer ||
    !readbacks_[1].buffer || !readbacks_[2].buffer)
  {
    const std::string failure = error_.empty() ? "allocating OpenGL preparation buffers" : error_;
    clear();
    error_ = failure;
    return false;
  }
  scratch_bytes_ = 2u * pair_bytes + 2u * tile_bytes + 2u * table_bytes +
    2u * kRadix * sizeof(std::uint32_t) + projected_bytes;

  // Every slot counts as written, so the first preparation that clears the
  // tail empties whatever an instance buffer was allocated holding.
  const State initial{0u, 0u, 0u, count_, 0u};
  updateBuffer(state_, &initial, sizeof(initial));
  const DrawIndexedArguments empty{index_count_, 0u, 0u, 0u, 0u};
  updateBuffer(draw_arguments_, &empty, sizeof(empty));
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  last_visible_ = -1;
  error_.clear();
  return true;
}

void OpenGlViewPreparationCore::clear()
{
  for (Readback & readback : readbacks_) {
    if (readback.fence) {
      glDeleteSync(readback.fence);
      readback.fence = nullptr;
    }
    if (readback.buffer != 0u) {
      glDeleteBuffers(1, &readback.buffer);
      readback.buffer = 0u;
    }
  }
  for (GLuint * buffer : {&splats_, &sh_dc_, &sh_rest_, &keys_a_, &keys_b_, &tile_counts_,
      &tile_offsets_, &histograms_, &offsets_, &totals_, &digit_bases_, &state_, &tile_dispatch_,
      &gather_dispatch_, &clear_dispatch_, &projected_, &draw_arguments_, &view_parameters_,
      &projection_parameters_, &raster_parameters_, &block_sums_, &block_offsets_,
      &raster_state_, &raster_sort_dispatch_, &raster_pair_dispatch_, &pairs_a_, &pairs_b_,
      &raster_histograms_, &raster_offsets_, &ranges_})
  {
    if (*buffer != 0u) {
      glDeleteBuffers(1, buffer);
      *buffer = 0u;
    }
  }
  count_ = 0u;
  tile_count_ = 0u;
  index_count_ = 0u;
  scratch_bytes_ = 0u;
  raster_ = RasterParameters{};
  raster_key_bits_ = 0u;
  readback_cursor_ = 0u;
  last_visible_ = -1;
}

bool OpenGlViewPreparationCore::uploadStaticData(
  const SplatRecord * records, std::size_t record_count,
  const float * sh_dc, std::size_t sh_dc_floats,
  const float * sh_rest, std::size_t sh_rest_floats)
{
  if (count_ == 0 || !records || record_count != count_ || !sh_dc ||
    sh_dc_floats != static_cast<std::size_t>(count_) * 3u)
  {
    error_ = "OpenGL static upload does not match the configured splat count";
    return false;
  }
  const std::size_t record_bytes = record_count * sizeof(SplatRecord);
  const std::size_t dc_bytes = sh_dc_floats * sizeof(float);
  const std::size_t rest_bytes = sh_rest_floats * sizeof(float);
  if (!fitsStorageBlock(record_bytes) || !fitsStorageBlock(dc_bytes) ||
    !fitsStorageBlock(std::max<std::size_t>(rest_bytes, sizeof(float))))
  {
    error_ = "splat or SH data exceeds GL_MAX_SHADER_STORAGE_BLOCK_SIZE";
    return false;
  }
  if (!uploadBuffer(splats_, records, record_bytes, GL_STATIC_DRAW) ||
    !uploadBuffer(sh_dc_, sh_dc, dc_bytes, GL_STATIC_DRAW))
  {
    return false;
  }
  const float zero = 0.0f;
  if (!uploadBuffer(
      sh_rest_, rest_bytes > 0 ? static_cast<const void *>(sh_rest) : &zero,
      std::max<std::size_t>(rest_bytes, sizeof(float)), GL_STATIC_DRAW))
  {
    return false;
  }
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  error_.clear();
  return true;
}

bool OpenGlViewPreparationCore::encode(
  const ViewParameters & parameters, const ProjectionParameters & projection,
  GLuint instances, GLuint draw_arguments, bool clear_tail)
{
  if (!ready() || count_ == 0 || !splats_ || !sh_dc_ || !sh_rest_ || !instances ||
    parameters.count != count_)
  {
    error_ = "OpenGL view preparation is not fully configured";
    return false;
  }
  if (parameters.sh_coefficients > 15u) {
    error_ = "OpenGL preparation supports spherical harmonics through degree 3";
    return false;
  }
  const GLuint draw = draw_arguments ? draw_arguments : draw_arguments_;

  pollReadbacks();
  GLint previous_program = 0;
  glGetIntegerv(GL_CURRENT_PROGRAM, &previous_program);
  updateBuffer(view_parameters_, &parameters, sizeof(parameters));
  updateBuffer(projection_parameters_, &projection, sizeof(projection));

  mark(0);
  use(programs_[0]);
  bind(0, splats_); bind(1, keys_b_); bind(2, view_parameters_);
  glDispatchCompute(groups(count_, 256u), 1, 1);
  storageBarrier();
  mark(kMarkAfterKeys);

  use(programs_[1]);
  bind(0, keys_b_); bind(1, tile_counts_);
  uniform("count", count_); uniform("tile_count", tile_count_);
  uniform("elements_per_tile", kElementsPerTile);
  glDispatchCompute(groups(tile_count_, 256u), 1, 1);
  storageBarrier();
  mark(kMarkAfterCompactCount);

  use(programs_[2]);
  bind(0, tile_counts_); bind(1, tile_offsets_); bind(2, state_);
  bind(3, tile_dispatch_); bind(4, gather_dispatch_); bind(5, draw);
  bind(6, clear_dispatch_);
  uniform("tile_count", tile_count_); uniform("elements_per_tile", kElementsPerTile);
  uniform("gather_width", kGatherWidth); uniform("index_count", index_count_);
  glDispatchCompute(1, 1, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
  mark(kMarkAfterCompactScan);

  use(programs_[3]);
  bind(0, keys_b_); bind(1, keys_a_); bind(2, tile_offsets_);
  uniform("count", count_); uniform("tile_count", tile_count_);
  uniform("elements_per_tile", kElementsPerTile);
  // Compaction must visit every source tile. The GPU-generated indirect
  // count describes compacted tiles and is used only by the radix stages.
  glDispatchCompute(tile_count_, 1, 1);
  storageBarrier();
  mark(1);

  // Colour and project before the sort, while the survivors are still in
  // index order; see GSPLAT_STAGE_SHADE_PROJECT. Rewrites keys_a_, the sort's
  // input, to carry compacted slots.
  use(programs_[10]);
  bind(0, keys_a_); bind(1, splats_); bind(2, sh_dc_); bind(3, sh_rest_);
  bind(4, state_); bind(5, projected_); bind(6, view_parameters_);
  bind(7, projection_parameters_);
  dispatchIndirect(gather_dispatch_);
  storageBarrier();
  mark(2);

  // Four passes, so the sorted pairs end in keys_a_ again; sortedBuffer().
  const GLuint input = radix(
    keys_a_, keys_b_, state_, tile_dispatch_, histograms_, offsets_, tile_count_, 32u,
    kMarkRadix);
  mark(3);

  use(programs_[8]);
  bind(0, input); bind(1, projected_); bind(4, state_); bind(5, instances);
  dispatchIndirect(gather_dispatch_);
  if (clear_tail) {
    storageBarrier();
    use(programs_[9]);
    bind(0, state_); bind(1, instances);
    dispatchIndirect(clear_dispatch_);
  }
  glMemoryBarrier(
    GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT |
    GL_COMMAND_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
  mark(4);
  queueReadback();
  readTimings();

  for (GLuint binding = 0; binding < 8; ++binding) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, 0);
  }
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
  glUseProgram(static_cast<GLuint>(previous_program));
  const GLenum gl_error = glGetError();
  if (gl_error != GL_NO_ERROR) {
    std::ostringstream message;
    message << "OpenGL compute preparation failed with error 0x" << std::hex << gl_error;
    error_ = message.str();
    return false;
  }
  error_.clear();
  return true;
}

GLuint OpenGlViewPreparationCore::radix(
  GLuint input, GLuint output, GLuint state, GLuint dispatch, GLuint histograms,
  GLuint offsets, std::uint32_t tile_count, std::uint32_t key_bits, std::size_t first_mark)
{
  for (std::uint32_t shift = 0; shift < key_bits; shift += 8u) {
    const std::size_t pass_mark = first_mark + 4u * (shift / 8u);
    const auto step_mark = [&](std::size_t step) {
        if (first_mark != 0u) {
          mark(pass_mark + step);
        }
      };
    use(programs_[4]);
    bind(0, input); bind(1, histograms); bind(2, state);
    uniform("tile_count", tile_count); uniform("elements_per_tile", kElementsPerTile);
    uniform("shift", shift);
    dispatchIndirect(dispatch);
    storageBarrier();
    step_mark(0u);

    use(programs_[5]);
    bind(0, histograms); bind(1, offsets); bind(2, totals_); bind(3, state);
    uniform("tile_count", tile_count);
    glDispatchCompute(kRadix, 1, 1);
    storageBarrier();
    step_mark(1u);

    use(programs_[6]);
    bind(0, totals_); bind(1, digit_bases_);
    glDispatchCompute(1, 1, 1);
    storageBarrier();
    step_mark(2u);

    use(programs_[7]);
    bind(0, input); bind(1, output); bind(2, offsets); bind(3, digit_bases_); bind(4, state);
    uniform("tile_count", tile_count); uniform("elements_per_tile", kElementsPerTile);
    uniform("shift", shift);
    dispatchIndirect(dispatch);
    storageBarrier();
    step_mark(3u);
    std::swap(input, output);
  }
  return input;
}

bool OpenGlViewPreparationCore::rasterReady() const
{
  return std::all_of(raster_programs_.begin(), raster_programs_.end(), [](GLuint program) {
             return program != 0u;
           });
}

bool OpenGlViewPreparationCore::configureRaster(
  std::uint32_t width, std::uint32_t height, std::uint32_t pair_capacity)
{
  if (!ready() || count_ == 0u) {
    error_ = "configure the preparation before the rasteriser";
    return false;
  }
  if (!rasterReady()) {
    error_ = raster_error_.empty() ? "the tile rasteriser did not compile" : raster_error_;
    return false;
  }
  if (width == 0u || height == 0u || pair_capacity == 0u) {
    error_ = "the tile rasteriser needs a non-empty target";
    return false;
  }
  for (GLuint * buffer : {&raster_parameters_, &block_sums_, &block_offsets_, &raster_state_,
      &raster_sort_dispatch_, &raster_pair_dispatch_, &pairs_a_, &pairs_b_,
      &raster_histograms_, &raster_offsets_, &ranges_})
  {
    if (*buffer != 0u) {
      glDeleteBuffers(1, buffer);
      *buffer = 0u;
    }
  }
  const std::uint32_t tiles_x = (width + 15u) / 16u;
  const std::uint32_t tiles_y = (height + 15u) / 16u;
  const std::uint32_t pair_tiles = (pair_capacity + kElementsPerTile - 1u) / kElementsPerTile;
  raster_ = RasterParameters{width, height, tiles_x, tiles_y, pair_capacity, pair_tiles, 0u,
    -1.0f};
  raster_key_bits_ = 8u;
  while (raster_key_bits_ < 32u &&
    (std::uint64_t(1) << raster_key_bits_) < std::uint64_t(tiles_x) * tiles_y)
  {
    raster_key_bits_ += 8u;
  }
  const std::size_t blocks = groups(count_, kGatherWidth);
  const std::size_t pair_bytes = std::size_t(pair_capacity) * sizeof(std::uint32_t) * 2u;
  const std::size_t table_bytes = std::size_t(pair_tiles) * kRadix * sizeof(std::uint32_t);
  if (!fitsStorageBlock(pair_bytes) || !fitsStorageBlock(table_bytes)) {
    error_ = "the tile pairs exceed GL_MAX_SHADER_STORAGE_BLOCK_SIZE";
    return false;
  }
  raster_parameters_ = newBuffer(sizeof(RasterParameters), GL_STATIC_DRAW);
  block_sums_ = newBuffer(blocks * sizeof(std::uint32_t), GL_DYNAMIC_COPY);
  block_offsets_ = newBuffer(blocks * sizeof(std::uint32_t), GL_DYNAMIC_COPY);
  raster_state_ = newBuffer(sizeof(State), GL_DYNAMIC_COPY);
  raster_sort_dispatch_ = newBuffer(sizeof(DispatchArguments), GL_DYNAMIC_COPY);
  raster_pair_dispatch_ = newBuffer(sizeof(DispatchArguments), GL_DYNAMIC_COPY);
  pairs_a_ = newBuffer(pair_bytes, GL_DYNAMIC_COPY);
  pairs_b_ = newBuffer(pair_bytes, GL_DYNAMIC_COPY);
  raster_histograms_ = newBuffer(table_bytes, GL_DYNAMIC_COPY);
  raster_offsets_ = newBuffer(table_bytes, GL_DYNAMIC_COPY);
  ranges_ = newBuffer(std::size_t(tiles_x) * tiles_y * sizeof(std::uint32_t) * 2u,
      GL_DYNAMIC_COPY);
  if (!raster_parameters_ || !block_sums_ || !block_offsets_ || !raster_state_ ||
    !raster_sort_dispatch_ || !raster_pair_dispatch_ || !pairs_a_ || !pairs_b_ ||
    !raster_histograms_ || !raster_offsets_ || !ranges_)
  {
    return false;
  }
  updateBuffer(raster_parameters_, &raster_, sizeof(raster_));
  const State empty{};
  updateBuffer(raster_state_, &empty, sizeof(empty));
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  error_.clear();
  return true;
}

void OpenGlViewPreparationCore::encodeRasterBin(GLuint instances)
{
  use(raster_programs_[0]);
  bind(0, instances); bind(1, state_); bind(2, block_sums_); bind(3, raster_parameters_);
  dispatchIndirect(gather_dispatch_);
  storageBarrier();

  use(raster_programs_[1]);
  bind(0, block_sums_); bind(1, block_offsets_); bind(2, state_); bind(3, raster_state_);
  bind(4, raster_sort_dispatch_); bind(5, raster_pair_dispatch_); bind(6, raster_parameters_);
  glDispatchCompute(1, 1, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

  use(raster_programs_[2]);
  bind(0, instances); bind(1, state_); bind(2, block_offsets_); bind(3, raster_parameters_);
  bind(4, pairs_a_);
  dispatchIndirect(gather_dispatch_);
  storageBarrier();
}

GLuint OpenGlViewPreparationCore::encodeRasterSort()
{
  return radix(
    pairs_a_, pairs_b_, raster_state_, raster_sort_dispatch_, raster_histograms_,
    raster_offsets_, raster_.pair_tile_capacity, raster_key_bits_, 0u);
}

void OpenGlViewPreparationCore::encodeRasterTiles(
  GLuint sorted_pairs, GLuint instances, GLuint texture)
{
  use(raster_programs_[3]);
  bind(0, ranges_); bind(1, raster_parameters_);
  glDispatchCompute(groups(2u * raster_.tiles_x * raster_.tiles_y, 256u), 1, 1);
  storageBarrier();

  use(raster_programs_[4]);
  bind(0, sorted_pairs); bind(1, raster_state_); bind(2, ranges_);
  dispatchIndirect(raster_pair_dispatch_);
  storageBarrier();

  use(raster_programs_[5]);
  bind(0, sorted_pairs); bind(1, ranges_); bind(2, instances); bind(3, raster_parameters_);
  glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
  glDispatchCompute(raster_.tiles_x, raster_.tiles_y, 1);
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
    GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
  glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
}

bool OpenGlViewPreparationCore::encodeRaster(GLuint instances, GLuint texture)
{
  if (!rasterReady() || !raster_state_ || !instances || !texture) {
    error_ = "the tile rasteriser is not configured";
    return false;
  }
  GLint previous_program = 0;
  glGetIntegerv(GL_CURRENT_PROGRAM, &previous_program);
  encodeRasterBin(instances);
  encodeRasterTiles(encodeRasterSort(), instances, texture);
  for (GLuint binding = 0; binding < 8; ++binding) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, 0);
  }
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
  glUseProgram(static_cast<GLuint>(previous_program));
  const GLenum gl_error = glGetError();
  if (gl_error != GL_NO_ERROR) {
    std::ostringstream message;
    message << "OpenGL tile rasterisation failed with error 0x" << std::hex << gl_error;
    error_ = message.str();
    return false;
  }
  error_.clear();
  return true;
}

std::size_t OpenGlViewPreparationCore::scratchBytes() const
{
  return scratch_bytes_;
}

GLuint OpenGlViewPreparationCore::compile(const char * stage)
{
  if (!error_.empty()) {
    return 0u;
  }
  const std::string stage_name(stage);
  const bool ballots = subgroup_ballot_ &&
    (stage_name == "GSPLAT_STAGE_COMPACT_SCATTER" || stage_name == "GSPLAT_STAGE_RADIX_SCATTER" ||
    stage_name == "GSPLAT_STAGE_RASTER_TILES");
  const std::string source = std::string("#version 430\n") +
    (ballots ?
    "#extension GL_KHR_shader_subgroup_basic : require\n"
    "#extension GL_KHR_shader_subgroup_ballot : require\n"
    "#define GSPLAT_SUBGROUP_BALLOT 1\n" : "") +
    "#define " + stage_name + " 1\n" + shader_source_;
  const char * data = source.c_str();
  const GLint length = static_cast<GLint>(source.size());
  GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(shader, 1, &data, &length);
  glCompileShader(shader);
  GLint compiled = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (compiled != GL_TRUE) {
    error_ = std::string("compiling OpenGL stage ") + stage + ": " + shaderLog(shader, false);
    glDeleteShader(shader);
    return 0u;
  }
  GLuint program = glCreateProgram();
  glAttachShader(program, shader);
  glLinkProgram(program);
  glDeleteShader(shader);
  GLint linked = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &linked);
  if (linked != GL_TRUE) {
    error_ = std::string("linking OpenGL stage ") + stage + ": " + shaderLog(program, true);
    glDeleteProgram(program);
    return 0u;
  }
  return program;
}

GLuint OpenGlViewPreparationCore::newBuffer(std::size_t bytes, GLenum usage)
{
  GLuint buffer = 0u;
  glGenBuffers(1, &buffer);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, usage);
  if (glGetError() != GL_NO_ERROR) {
    if (buffer) {
      glDeleteBuffers(1, &buffer);
    }
    error_ = "OpenGL buffer allocation failed";
    return 0u;
  }
  return buffer;
}

bool OpenGlViewPreparationCore::uploadBuffer(
  GLuint & buffer, const void * data, std::size_t bytes, GLenum usage)
{
  if (buffer == 0u) {
    glGenBuffers(1, &buffer);
  }
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
  glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes), data, usage);
  if (glGetError() != GL_NO_ERROR) {
    error_ = "uploading an OpenGL preparation buffer failed";
    return false;
  }
  return true;
}

bool OpenGlViewPreparationCore::fitsStorageBlock(std::size_t bytes) const
{
  return bytes <= static_cast<std::size_t>(max_storage_block_bytes_);
}

void OpenGlViewPreparationCore::use(GLuint program)
{
  current_program_ = program;
  glUseProgram(program);
}

void OpenGlViewPreparationCore::uniform(const char * name, std::uint32_t value)
{
  const GLint location = glGetUniformLocation(current_program_, name);
  if (location >= 0) {
    glUniform1ui(location, value);
  }
}

void OpenGlViewPreparationCore::mark(std::size_t boundary)
{
  if (profiling_) {
    timerFunctions().query_counter(queries_[boundary], GL_TIMESTAMP);
  }
}

void OpenGlViewPreparationCore::readTimings()
{
  if (!profiling_) {
    return;
  }
  std::array<GLuint64, 24> ticks{};
  for (std::size_t i = 0; i < ticks.size(); ++i) {
    timerFunctions().get_query_result(queries_[i], GL_QUERY_RESULT, &ticks[i]);
  }
  const auto ms = [&ticks](std::size_t from, std::size_t to) {
      return static_cast<double>(ticks[to] - ticks[from]) / 1.0e6;
    };
  timings_.cull_ms = ms(0, 1);
  timings_.shade_ms = ms(1, 2);
  timings_.sort_ms = ms(2, 3);
  timings_.gather_ms = ms(3, 4);

  timings_.keys_ms = ms(0, kMarkAfterKeys);
  timings_.compact_count_ms = ms(kMarkAfterKeys, kMarkAfterCompactCount);
  timings_.compact_scan_ms = ms(kMarkAfterCompactCount, kMarkAfterCompactScan);
  timings_.compact_scatter_ms = ms(kMarkAfterCompactScan, 1);
  timings_.radix_histogram_ms = 0.0;
  timings_.radix_scan_ms = 0.0;
  timings_.radix_scatter_ms = 0.0;
  for (std::size_t pass = 0; pass < 4u; ++pass) {
    const std::size_t base = kMarkRadix + 4u * pass;
    const std::size_t start = pass == 0 ? 2u : base - 1u;  // the previous scatter's end
    timings_.radix_histogram_ms += ms(start, base);
    timings_.radix_scan_ms += ms(base, base + 2u);
    timings_.radix_scatter_ms += ms(base + 2u, base + 3u);
  }
}

void OpenGlViewPreparationCore::pollReadbacks()
{
  for (Readback & readback : readbacks_) {
    if (!readback.fence) {
      continue;
    }
    const GLenum status = glClientWaitSync(readback.fence, 0, 0);
    if (status != GL_ALREADY_SIGNALED && status != GL_CONDITION_SATISFIED) {
      continue;
    }
    std::uint32_t visible = 0u;
    glBindBuffer(GL_COPY_READ_BUFFER, readback.buffer);
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(visible), &visible);
    last_visible_ = visible;
    glDeleteSync(readback.fence);
    readback.fence = nullptr;
  }
  glBindBuffer(GL_COPY_READ_BUFFER, 0);
}

void OpenGlViewPreparationCore::queueReadback()
{
  Readback & readback = readbacks_[readback_cursor_];
  readback_cursor_ = (readback_cursor_ + 1u) % readbacks_.size();
  if (readback.fence) {
    return;
  }
  glBindBuffer(GL_COPY_READ_BUFFER, state_);
  glBindBuffer(GL_COPY_WRITE_BUFFER, readback.buffer);
  glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(std::uint32_t));
  readback.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  glBindBuffer(GL_COPY_READ_BUFFER, 0);
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

}  // namespace gaussian_splatting_rviz_plugins
