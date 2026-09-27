#include "opengl_view_preparation_core.hpp"

#include <algorithm>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <dlfcn.h>

namespace gaussian_splatting_rviz_plugins
{
namespace
{

constexpr std::uint32_t kElementsPerTile = 1024u;
constexpr std::uint32_t kGatherWidth = 256u;
constexpr std::uint32_t kRadix = 256u;
// Slots FINISH_ORDER gathers from per pass: 64 MiB of DrawnSplat.
constexpr std::uint32_t kFinishSlotsPerPass = 1u << 20;

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
  sizeof(OpenGlViewPreparationCore::State) == 16, "OpenGL preparation state layout");

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
  const char * stages[] = {
    "GSPLAT_STAGE_MAKE_DEPTH_KEYS", "GSPLAT_STAGE_COMPACT_COUNT", "GSPLAT_STAGE_COMPACT_SCAN",
    "GSPLAT_STAGE_COMPACT_SCATTER", "GSPLAT_STAGE_RADIX_HISTOGRAM",
    "GSPLAT_STAGE_RADIX_SCAN_OFFSETS", "GSPLAT_STAGE_RADIX_SCAN_BASES",
    "GSPLAT_STAGE_RADIX_SCATTER", "GSPLAT_STAGE_SHADE_SLOTS", "GSPLAT_STAGE_FINISH_ORDER",
    "GSPLAT_STAGE_PUBLISH", "GSPLAT_STAGE_REPROJECT_GATHER", "GSPLAT_STAGE_CLEAR_TAIL"};
  static_assert(sizeof(stages) / sizeof(stages[0]) == std::tuple_size<decltype(programs_)>::value,
    "a program per stage");
  for (std::size_t i = 0; i < programs_.size(); ++i) {
    programs_[i] = compile(stages[i]);
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
  const std::size_t survivor_bytes = std::size_t(count) * sizeof(DrawnSplat);
  const std::size_t tile_bytes = static_cast<std::size_t>(tile_count_) * sizeof(std::uint32_t);
  const std::size_t table_bytes = tile_bytes * kRadix;
  for (std::size_t bytes : {pair_bytes, survivor_bytes, table_bytes}) {
    if (!fitsStorageBlock(bytes)) {
      error_ = "scene requires an SSBO larger than GL_MAX_SHADER_STORAGE_BLOCK_SIZE";
      clear();
      return false;
    }
  }

  staged_ = newBuffer(pair_bytes, GL_DYNAMIC_COPY);
  tile_counts_ = newBuffer(tile_bytes, GL_DYNAMIC_COPY);
  tile_offsets_ = newBuffer(tile_bytes, GL_DYNAMIC_COPY);
  histograms_ = newBuffer(table_bytes, GL_DYNAMIC_COPY);
  offsets_ = newBuffer(table_bytes, GL_DYNAMIC_COPY);
  totals_ = newBuffer(kRadix * sizeof(std::uint32_t), GL_DYNAMIC_COPY);
  digit_bases_ = newBuffer(kRadix * sizeof(std::uint32_t), GL_DYNAMIC_COPY);
  pairs_ = newBuffer(pair_bytes, GL_DYNAMIC_COPY);
  slots_ = newBuffer(survivor_bytes, GL_DYNAMIC_COPY);
  drawn_ = newBuffer(survivor_bytes, GL_DYNAMIC_COPY);
  instance_state_ = newBuffer(sizeof(std::uint32_t) * 3u, GL_DYNAMIC_COPY);
  clear_dispatch_ = newBuffer(sizeof(DispatchArguments), GL_DYNAMIC_COPY);
  draw_arguments_ = newBuffer(sizeof(DrawIndexedArguments), GL_DYNAMIC_COPY);
  view_parameters_ = newBuffer(sizeof(ViewParameters), GL_STREAM_DRAW);
  projection_parameters_ = newBuffer(sizeof(ProjectionParameters), GL_STREAM_DRAW);
  bool allocated = staged_ && tile_counts_ && tile_offsets_ && histograms_ && offsets_ &&
    totals_ && digit_bases_ && pairs_ && slots_ && drawn_ && instance_state_ &&
    clear_dispatch_ && draw_arguments_ && view_parameters_ && projection_parameters_;
  const State empty_state{0u, 0u, 0u, 0u};
  for (Order & order : orders_) {
    order.state = newBuffer(sizeof(State), GL_DYNAMIC_COPY);
    order.tile_dispatch = newBuffer(sizeof(DispatchArguments), GL_DYNAMIC_COPY);
    order.gather_dispatch = newBuffer(sizeof(DispatchArguments), GL_DYNAMIC_COPY);
    allocated = allocated && order.state && order.tile_dispatch && order.gather_dispatch;
    if (order.state) {
      updateBuffer(order.state, &empty_state, sizeof(empty_state));
    }
  }
  for (Readback & readback : readbacks_) {
    readback.buffer = newBuffer(sizeof(std::uint32_t), GL_STREAM_READ);
    allocated = allocated && readback.buffer;
  }
  if (!allocated) {
    const std::string failure = error_.empty() ? "allocating OpenGL preparation buffers" : error_;
    clear();
    error_ = failure;
    return false;
  }
  scratch_bytes_ = 2u * pair_bytes + 2u * survivor_bytes + 2u * tile_bytes +
    2u * table_bytes + 2u * kRadix * sizeof(std::uint32_t);

  // Every slot counts as written, so the first preparation that clears the
  // tail empties whatever an instance buffer was allocated holding.
  const std::uint32_t instances[3] = {0u, count_, 0u};
  updateBuffer(instance_state_, instances, sizeof(instances));
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
  std::vector<GLuint *> buffers = {&splats_, &sh_dc_, &sh_rest_, &staged_, &tile_counts_,
    &tile_offsets_, &histograms_, &offsets_, &totals_, &digit_bases_, &pairs_, &slots_,
    &drawn_, &instance_state_, &clear_dispatch_, &draw_arguments_, &view_parameters_,
    &projection_parameters_};
  for (Order & order : orders_) {
    for (GLuint * buffer : {&order.state, &order.tile_dispatch, &order.gather_dispatch})
    {
      buffers.push_back(buffer);
    }
  }
  for (GLuint * buffer : buffers) {
    if (*buffer != 0u) {
      glDeleteBuffers(1, buffer);
      *buffer = 0u;
    }
  }
  count_ = 0u;
  tile_count_ = 0u;
  index_count_ = 0u;
  scratch_bytes_ = 0u;
  readback_cursor_ = 0u;
  last_visible_ = -1;
  current_ = -1;
  building_ = -1;
  build_step_ = 0;
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
  // The orders' colours and survivors belong to the old data.
  current_ = -1;
  building_ = -1;
  build_step_ = 0;
  error_.clear();
  return true;
}

bool OpenGlViewPreparationCore::encode(
  const ViewParameters & parameters, const ProjectionParameters & projection,
  GLuint instances, GLuint draw_arguments, bool clear_tail)
{
  return prepare(parameters, projection, instances, draw_arguments, clear_tail, false);
}

bool OpenGlViewPreparationCore::encodeIncremental(
  const ViewParameters & parameters, const ProjectionParameters & projection,
  GLuint instances, GLuint draw_arguments, bool clear_tail)
{
  return prepare(parameters, projection, instances, draw_arguments, clear_tail, current_ >= 0);
}

bool OpenGlViewPreparationCore::checkScene(const ViewParameters & parameters)
{
  if (!ready() || count_ == 0 || !splats_ || !sh_dc_ || !sh_rest_ ||
    parameters.count != count_)
  {
    error_ = "OpenGL view preparation is not fully configured";
    return false;
  }
  if (parameters.sh_coefficients > 15u) {
    error_ = "OpenGL preparation supports spherical harmonics through degree 3";
    return false;
  }
  return true;
}

bool OpenGlViewPreparationCore::prepare(
  const ViewParameters & parameters, const ProjectionParameters & projection,
  GLuint instances, GLuint draw_arguments, bool clear_tail, bool incremental)
{
  if (!checkScene(parameters) || !instances) {
    if (!instances) {
      error_ = "OpenGL view preparation has no instance buffer";
    }
    return false;
  }
  const GLuint draw = draw_arguments ? draw_arguments : draw_arguments_;

  // What this call builds: a whole order, or the next of four steps of one.
  // Culling and colour run with the view the build started with, which the
  // view block keeps until the next build starts.
  Work work;
  if (!incremental || building_ < 0) {
    building_ = current_ == 0 ? 1 : 0;
    build_step_ = 0;
    work.cull = true;
    updateBuffer(view_parameters_, &parameters, sizeof(parameters));
  }
  // Five steps of comparable cost on the full Garden scene: culling and
  // compaction, each half of the colour, and each half of the sort.
  if (!incremental) {
    work.shade_parts = 1u;
    work.first_bit = 0u;
    work.end_bit = 32u;
  } else if (build_step_ == 1 || build_step_ == 2) {
    work.shade_parts = 2u;
    work.shade_part = static_cast<std::uint32_t>(build_step_ - 1);
  } else if (build_step_ == 3) {
    work.end_bit = 16u;
  } else if (build_step_ == 4) {
    work.first_bit = 16u;
    work.end_bit = 32u;
  }
  const Order & order = orders_[building_];

  pollReadbacks();
  GLint previous_program = 0;
  glGetIntegerv(GL_CURRENT_PROGRAM, &previous_program);
  updateBuffer(projection_parameters_, &projection, sizeof(projection));

  // Every boundary is marked whether or not its stage ran, so that a stage an
  // incremental call skipped times as zero.
  mark(0);
  if (work.cull) {
    use(programs_[0]);
    bind(0, splats_); bind(1, staged_); bind(2, view_parameters_);
    glDispatchCompute(groups(count_, 256u), 1, 1);
    storageBarrier();
  }
  mark(kMarkAfterKeys);
  if (work.cull) {
    use(programs_[1]);
    bind(0, staged_); bind(1, tile_counts_);
    uniform("count", count_); uniform("tile_count", tile_count_);
    uniform("elements_per_tile", kElementsPerTile);
    glDispatchCompute(groups(tile_count_, 256u), 1, 1);
    storageBarrier();
  }
  mark(kMarkAfterCompactCount);
  if (work.cull) {
    use(programs_[2]);
    bind(0, tile_counts_); bind(1, tile_offsets_); bind(2, order.state);
    bind(3, order.tile_dispatch); bind(4, order.gather_dispatch);
    uniform("tile_count", tile_count_); uniform("elements_per_tile", kElementsPerTile);
    uniform("gather_width", kGatherWidth);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
  }
  mark(kMarkAfterCompactScan);
  if (work.cull) {
    use(programs_[3]);
    bind(0, staged_); bind(1, pairs_); bind(2, tile_offsets_);
    uniform("count", count_); uniform("tile_count", tile_count_);
    uniform("elements_per_tile", kElementsPerTile);
    // Compaction must visit every source tile. The GPU-generated indirect
    // count describes compacted tiles and is used only by the radix stages.
    glDispatchCompute(tile_count_, 1, 1);
    storageBarrier();
    build_step_ = 1;
  }
  mark(1);

  // Colour and copy before the sort, while the survivors are still in index
  // order; see GSPLAT_STAGE_SHADE_SLOTS. Rewrites the pairs, the sort's input,
  // to carry compacted slots.
  if (work.shade_parts > 0u) {
    use(programs_[8]);
    bind(0, pairs_); bind(1, splats_); bind(2, sh_dc_); bind(3, sh_rest_);
    bind(4, order.state); bind(5, slots_); bind(6, view_parameters_);
    uniform("part_index", work.shade_part); uniform("part_count", work.shade_parts);
    // Sized for every survivor; invocations past the part return at once.
    dispatchIndirect(order.gather_dispatch);
    storageBarrier();
    build_step_ = work.shade_part + 1u == work.shade_parts ? 3 : 2;
  }
  mark(2);

  // Pairs of passes, so the sorted pairs end back in pairs_.
  GLuint input = pairs_;
  GLuint output = staged_;
  for (std::uint32_t shift = 0; shift < 32u; shift += 8u) {
    const std::size_t pass_mark = kMarkRadix + 4u * (shift / 8u);
    const bool run = shift >= work.first_bit && shift < work.end_bit;
    if (run) {
      use(programs_[4]);
      bind(0, input); bind(1, histograms_); bind(2, order.state);
      uniform("tile_count", tile_count_); uniform("elements_per_tile", kElementsPerTile);
      uniform("shift", shift);
      dispatchIndirect(order.tile_dispatch);
      storageBarrier();
    }
    mark(pass_mark);
    if (run) {
      use(programs_[5]);
      bind(0, histograms_); bind(1, offsets_); bind(2, totals_); bind(3, order.state);
      uniform("tile_count", tile_count_);
      glDispatchCompute(kRadix, 1, 1);
      storageBarrier();
    }
    mark(pass_mark + 1u);
    if (run) {
      use(programs_[6]);
      bind(0, totals_); bind(1, digit_bases_);
      glDispatchCompute(1, 1, 1);
      storageBarrier();
    }
    mark(pass_mark + 2u);
    if (run) {
      use(programs_[7]);
      bind(0, input); bind(1, output); bind(2, offsets_); bind(3, digit_bases_);
      bind(4, order.state);
      uniform("tile_count", tile_count_); uniform("elements_per_tile", kElementsPerTile);
      uniform("shift", shift);
      dispatchIndirect(order.tile_dispatch);
      storageBarrier();
      std::swap(input, output);
    }
    mark(pass_mark + 3u);
  }
  if (work.end_bit == 16u) {
    build_step_ = 4;
  }
  if (work.end_bit == 32u) {
    use(programs_[9]);
    bind(0, pairs_); bind(1, slots_); bind(4, order.state); bind(5, drawn_);
    // The passes write disjoint positions, so they need no barrier between them.
    for (std::uint32_t first = 0; first < count_; first += kFinishSlotsPerPass) {
      uniform("first_slot", first);
      uniform("end_slot", first + std::min(kFinishSlotsPerPass, count_ - first));
      dispatchIndirect(order.gather_dispatch);
    }
    storageBarrier();
    current_ = building_;
    building_ = -1;
    build_step_ = 0;
  }
  mark(3);

  const Order & drawn = orders_[current_];
  use(programs_[10]);
  bind(0, drawn.state); bind(1, instance_state_); bind(2, draw); bind(3, clear_dispatch_);
  uniform("index_count", index_count_); uniform("gather_width", kGatherWidth);
  glDispatchCompute(1, 1, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

  use(programs_[11]);
  bind(0, drawn_); bind(4, drawn.state); bind(5, instances); bind(6, projection_parameters_);
  dispatchIndirect(drawn.gather_dispatch);
  if (clear_tail) {
    storageBarrier();
    use(programs_[12]);
    bind(0, instance_state_); bind(1, instances);
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

GLuint OpenGlViewPreparationCore::stateBuffer() const
{
  return current_ >= 0 ? orders_[current_].state : 0u;
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
    (stage_name == "GSPLAT_STAGE_COMPACT_SCATTER" || stage_name == "GSPLAT_STAGE_RADIX_SCATTER");
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
  glBindBuffer(GL_COPY_READ_BUFFER, stateBuffer());
  glBindBuffer(GL_COPY_WRITE_BUFFER, readback.buffer);
  glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(std::uint32_t));
  readback.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  glBindBuffer(GL_COPY_READ_BUFFER, 0);
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

}  // namespace gaussian_splatting_rviz_plugins
