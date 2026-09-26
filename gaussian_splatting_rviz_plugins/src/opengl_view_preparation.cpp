#include "gaussian_splatting_rviz_plugins/opengl_view_preparation.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <dlfcn.h>

#include <RenderSystems/GL/OgreGLHardwareVertexBuffer.h>
#include <RenderSystems/GL/OgreGLRenderSystem.h>
#include <OgreHardwareBufferManager.h>
#include <OgreRenderable.h>

namespace gaussian_splatting_rviz_plugins
{
namespace
{

constexpr std::uint32_t kThreadsPerTile = 32u;
constexpr std::uint32_t kElementsPerTile = 1024u;
constexpr std::uint32_t kGatherWidth = 256u;
constexpr std::uint32_t kRadix = 256u;

#ifdef GSPLAT_OGRE_GL_INDIRECT_DRAW
constexpr bool kOgreIndirectDraw = true;
#else
// Upstream Ogre draws a renderable with the instance count the CPU gives it, so
// every instance is drawn and those past the visible count are left empty.
constexpr bool kOgreIndirectDraw = false;
#endif

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

// GSPLAT_PROFILE_GPU times each stage with timestamp queries, taken from the GL
// library for the same reason as the dispatch entry points above.
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

// Set and not "0", as for the Metal backend.
bool environmentFlag(const char * name)
{
  const char * value = std::getenv(name);
  return value && *value && std::strcmp(value, "0") != 0;
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

struct PreparationState
{
  std::uint32_t visible;
  std::uint32_t visible_tiles;
  std::uint32_t gather_groups;
  std::uint32_t written;
  std::uint32_t clear_end;
};
static_assert(sizeof(PreparationState) == 20, "OpenGL preparation state layout");

std::string readFile(const std::string & path)
{
  std::ifstream stream(path);
  if (!stream) {
    return {};
  }
  return std::string(
    std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

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

class OpenGlViewPreparation final : public GpuViewPreparation
{
public:
  OpenGlViewPreparation(Ogre::GLRenderSystem * render_system, const std::string & shader_path)
  : render_system_(render_system), shader_source_(readFile(shader_path))
  {
    if (!render_system_) {
      error_ = "Ogre OpenGL render system is unavailable";
      return;
    }
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
      error_ = "could not read OpenGL preparation shader " + shader_path;
      return;
    }

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
    programs_[10] = compile("GSPLAT_STAGE_SHADE_VISIBLE");

    profiling_ = environmentFlag("GSPLAT_PROFILE_GPU") && timerFunctions().resolved();
    if (profiling_) {
      timerFunctions().gen_queries(static_cast<GLsizei>(queries_.size()), queries_.data());
    }
  }

  ~OpenGlViewPreparation() override
  {
    clear();
    if (profiling_) {
      timerFunctions().delete_queries(static_cast<GLsizei>(queries_.size()), queries_.data());
    }
    for (GLuint program : programs_) {
      if (program != 0u) {
        glDeleteProgram(program);
      }
    }
  }

  const char * backendName() const override
  {
    return kOgreIndirectDraw ? "OpenGL 4.3" : "OpenGL 4.3 (every instance drawn)";
  }

  bool configure(
    std::size_t count, std::uint32_t index_count,
    const Ogre::HardwareVertexBufferSharedPtr & instance_buffer,
    const Ogre::Renderable * renderable) override
  {
    clear();
    if (!ready()) {
      return false;
    }
    if (count == 0 || count > std::numeric_limits<std::uint32_t>::max() ||
      !instance_buffer || !renderable)
    {
      error_ = "OpenGL preparation needs a non-zero uint32 splat count, an instance "
        "buffer and a renderable";
      return false;
    }

    count_ = static_cast<std::uint32_t>(count);
    index_count_ = index_count;
    tile_count_ = (count_ + kElementsPerTile - 1u) / kElementsPerTile;
    const std::size_t pair_bytes = count * sizeof(std::uint32_t) * 2u;
    const std::size_t tile_bytes = static_cast<std::size_t>(tile_count_) * sizeof(std::uint32_t);
    const std::size_t table_bytes = tile_bytes * kRadix;
    const std::size_t colour_bytes = count * sizeof(float) * 4u;
    const std::array<std::size_t, 9> storage_sizes = {
      pair_bytes, pair_bytes, tile_bytes, tile_bytes,
      table_bytes, table_bytes, kRadix * sizeof(std::uint32_t),
      count * sizeof(ProjectedInstance), colour_bytes};
    for (std::size_t bytes : storage_sizes) {
      if (bytes > static_cast<std::size_t>(max_storage_block_bytes_)) {
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
    state_ = newBuffer(sizeof(PreparationState), GL_DYNAMIC_COPY);
    tile_dispatch_ = newBuffer(sizeof(DispatchArguments), GL_DYNAMIC_COPY);
    gather_dispatch_ = newBuffer(sizeof(DispatchArguments), GL_DYNAMIC_COPY);
    clear_dispatch_ = newBuffer(sizeof(DispatchArguments), GL_DYNAMIC_COPY);
    colours_ = newBuffer(colour_bytes, GL_DYNAMIC_COPY);
    view_parameters_ = newBuffer(sizeof(ViewParameters), GL_STREAM_DRAW);
    projection_parameters_ = newBuffer(sizeof(ProjectionParameters), GL_STREAM_DRAW);
    for (Readback & readback : readbacks_) {
      readback.buffer = newBuffer(sizeof(std::uint32_t), GL_STREAM_READ);
    }
    if (!keys_a_ || !keys_b_ || !tile_counts_ || !tile_offsets_ || !histograms_ ||
      !offsets_ || !totals_ || !digit_bases_ || !state_ || !tile_dispatch_ ||
      !gather_dispatch_ || !clear_dispatch_ || !colours_ || !view_parameters_ ||
      !projection_parameters_ || !readbacks_[0].buffer || !readbacks_[1].buffer ||
      !readbacks_[2].buffer)
    {
      const std::string failure = error_.empty() ? "allocating OpenGL preparation buffers" : error_;
      clear();
      error_ = failure;
      return false;
    }

    // Every slot counts as written, so the first preparation empties whatever
    // the new instance buffer was allocated holding before anything draws it.
    const PreparationState initial{0u, 0u, 0u, count_, 0u};
    updateBuffer(state_, &initial, sizeof(initial));

    const DrawIndexedArguments empty{index_count_, 0u, 0u, 0u, 0u};
    instance_buffer_ = instance_buffer;
    instance_gl_ =
      static_cast<Ogre::GLHardwareVertexBuffer *>(instance_buffer_.get())->getGLBufferId();
    renderable_ = renderable;
#ifdef GSPLAT_OGRE_GL_INDIRECT_DRAW
    auto & manager = Ogre::HardwareBufferManager::getSingleton();
    indirect_buffer_ = manager.createVertexBuffer(
      sizeof(DrawIndexedArguments), 1u, Ogre::HardwareBuffer::HBU_DYNAMIC_WRITE_ONLY);
    indirect_buffer_->writeData(0, sizeof(empty), &empty, true);
    indirect_gl_ =
      static_cast<Ogre::GLHardwareVertexBuffer *>(indirect_buffer_.get())->getGLBufferId();
    render_system_->setIndirectDrawBuffer(renderable_, indirect_buffer_);
#else
    // compact_scan still writes draw arguments; nothing reads them.
    draw_arguments_ = newBuffer(sizeof(DrawIndexedArguments), GL_DYNAMIC_COPY);
    if (!draw_arguments_) {
      const std::string failure = error_;
      clear();
      error_ = failure;
      return false;
    }
    updateBuffer(draw_arguments_, &empty, sizeof(empty));
    indirect_gl_ = draw_arguments_;
#endif
    last_visible_ = -1;
    error_.clear();
    return true;
  }

  bool uploadStaticData(
    const SplatRecord * records, std::size_t record_count,
    const float * sh_dc, std::size_t sh_dc_floats,
    const float * sh_rest, std::size_t sh_rest_floats) override
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
    error_.clear();
    return true;
  }

  bool encode(
    const ViewParameters & parameters, const ProjectionParameters & projection) override
  {
    if (!ready() || count_ == 0 || !splats_ || !sh_dc_ || !sh_rest_ ||
      !instance_gl_ || !indirect_gl_ || parameters.count != count_)
    {
      error_ = "OpenGL view preparation is not fully configured";
      return false;
    }
    if (parameters.sh_coefficients > 15u) {
      error_ = "OpenGL preparation supports spherical harmonics through degree 3";
      return false;
    }

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

    use(programs_[1]);
    bind(0, keys_b_); bind(1, tile_counts_);
    uniform("count", count_); uniform("tile_count", tile_count_);
    uniform("elements_per_tile", kElementsPerTile);
    glDispatchCompute(groups(tile_count_, 256u), 1, 1);
    storageBarrier();

    use(programs_[2]);
    bind(0, tile_counts_); bind(1, tile_offsets_); bind(2, state_);
    bind(3, tile_dispatch_); bind(4, gather_dispatch_); bind(5, indirect_gl_);
    bind(6, clear_dispatch_);
    uniform("tile_count", tile_count_); uniform("elements_per_tile", kElementsPerTile);
    uniform("gather_width", kGatherWidth); uniform("index_count", index_count_);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    use(programs_[3]);
    bind(0, keys_b_); bind(1, keys_a_); bind(2, tile_offsets_);
    uniform("count", count_); uniform("tile_count", tile_count_);
    uniform("elements_per_tile", kElementsPerTile);
    // Compaction must visit every source tile. The GPU-generated indirect
    // count describes compacted tiles and is used only by the radix stages.
    glDispatchCompute(tile_count_, 1, 1);
    storageBarrier();
    mark(1);

    // Colour before the sort, while the survivors are still in index order;
    // see GSPLAT_STAGE_SHADE_VISIBLE. The sort's second pass overwrites keys_a_.
    use(programs_[10]);
    bind(0, keys_a_); bind(1, splats_); bind(2, sh_dc_); bind(3, sh_rest_);
    bind(4, state_); bind(5, colours_); bind(6, view_parameters_);
    dispatchIndirect(gather_dispatch_);
    storageBarrier();
    mark(2);

    GLuint input = keys_a_;
    GLuint output = keys_b_;
    for (std::uint32_t shift = 0; shift < 32u; shift += 8u) {
      use(programs_[4]);
      bind(0, input); bind(1, histograms_); bind(2, state_);
      uniform("tile_count", tile_count_); uniform("elements_per_tile", kElementsPerTile);
      uniform("shift", shift);
      dispatchIndirect(tile_dispatch_);
      storageBarrier();

      use(programs_[5]);
      bind(0, histograms_); bind(1, offsets_); bind(2, totals_); bind(3, state_);
      uniform("tile_count", tile_count_);
      glDispatchCompute(kRadix / 64u, 1, 1);
      storageBarrier();

      use(programs_[6]);
      bind(0, totals_); bind(1, digit_bases_);
      glDispatchCompute(1, 1, 1);
      storageBarrier();

      use(programs_[7]);
      bind(0, input); bind(1, output); bind(2, offsets_); bind(3, digit_bases_); bind(4, state_);
      uniform("tile_count", tile_count_); uniform("elements_per_tile", kElementsPerTile);
      uniform("shift", shift);
      dispatchIndirect(tile_dispatch_);
      storageBarrier();
      std::swap(input, output);
    }
    mark(3);

    use(programs_[8]);
    bind(0, input); bind(1, splats_); bind(2, colours_);
    bind(4, state_); bind(5, instance_gl_); bind(7, projection_parameters_);
    dispatchIndirect(gather_dispatch_);
    if (!kOgreIndirectDraw) {
      storageBarrier();
      use(programs_[9]);
      bind(0, state_); bind(1, instance_gl_);
      dispatchIndirect(clear_dispatch_);
    }
    glMemoryBarrier(
      GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT |
      GL_COMMAND_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    mark(4);
    queueReadback();
    readStageTimes();

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

  std::int64_t lastVisibleCount() const override
  {
    return last_visible_;
  }

  bool drawsEveryInstance() const override
  {
    return !kOgreIndirectDraw;
  }

  bool profiling() const override
  {
    return profiling_;
  }

  StageTimes lastStageTimes() const override
  {
    return times_;
  }

  void clear() override
  {
#ifdef GSPLAT_OGRE_GL_INDIRECT_DRAW
    if (render_system_ && renderable_) {
      Ogre::HardwareVertexBufferSharedPtr none;
      render_system_->setIndirectDrawBuffer(renderable_, none);
    }
#endif
    renderable_ = nullptr;
    instance_buffer_.reset();
    indirect_buffer_.reset();
    instance_gl_ = 0u;
    indirect_gl_ = 0u;
    for (Readback & readback : readbacks_) {
      if (readback.fence) {
        glDeleteSync(readback.fence);
        readback.fence = nullptr;
      }
    }
    std::array<GLuint *, 20> buffers = {
      &splats_, &sh_dc_, &sh_rest_, &keys_a_, &keys_b_, &tile_counts_, &tile_offsets_,
      &histograms_, &offsets_, &totals_, &digit_bases_, &state_, &tile_dispatch_,
      &gather_dispatch_, &clear_dispatch_, &colours_, &draw_arguments_, &view_parameters_,
      &projection_parameters_, &readbacks_[0].buffer};
    // The first readback is included above; delete the other two separately.
    for (GLuint * buffer : buffers) {
      if (*buffer != 0u) {
        glDeleteBuffers(1, buffer);
        *buffer = 0u;
      }
    }
    for (std::size_t i = 1; i < readbacks_.size(); ++i) {
      if (readbacks_[i].buffer != 0u) {
        glDeleteBuffers(1, &readbacks_[i].buffer);
        readbacks_[i].buffer = 0u;
      }
    }
    count_ = 0u;
    tile_count_ = 0u;
    index_count_ = 0u;
    readback_cursor_ = 0u;
    last_visible_ = -1;
  }

  const std::string & error() const override
  {
    return error_;
  }

private:
  struct Readback
  {
    GLuint buffer = 0u;
    GLsync fence = nullptr;
  };

  // Timestamps at the stage boundaries, read back at once: the read waits for
  // the GPU, so GSPLAT_PROFILE_GPU serialises every preparation with the CPU.
  void mark(std::size_t boundary)
  {
    if (profiling_) {
      timerFunctions().query_counter(queries_[boundary], GL_TIMESTAMP);
    }
  }

  void readStageTimes()
  {
    if (!profiling_) {
      return;
    }
    std::array<GLuint64, 5> ticks{};
    for (std::size_t i = 0; i < ticks.size(); ++i) {
      timerFunctions().get_query_result(queries_[i], GL_QUERY_RESULT, &ticks[i]);
    }
    const auto ms = [&ticks](std::size_t from, std::size_t to) {
        return static_cast<double>(ticks[to] - ticks[from]) / 1.0e6;
      };
    times_.cull_ms = ms(0, 1);
    times_.shade_ms = ms(1, 2);
    times_.sort_ms = ms(2, 3);
    times_.gather_ms = ms(3, 4);
  }

  bool ready() const
  {
    return std::all_of(programs_.begin(), programs_.end(), [](GLuint program) {
               return program != 0u;
                                                                                                    });
  }

  GLuint compile(const char * stage)
  {
    if (!error_.empty()) {
      return 0u;
    }
    const std::string source = std::string("#version 430\n#define ") + stage + " 1\n" +
      shader_source_;
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

  GLuint newBuffer(std::size_t bytes, GLenum usage)
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

  bool uploadBuffer(GLuint & buffer, const void * data, std::size_t bytes, GLenum usage)
  {
    if (buffer == 0u) {
      glGenBuffers(1, &buffer);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(
      GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes), data, usage);
    if (glGetError() != GL_NO_ERROR) {
      error_ = "uploading an OpenGL preparation buffer failed";
      return false;
    }
    return true;
  }

  void updateBuffer(GLuint buffer, const void * data, std::size_t bytes)
  {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(bytes), data);
  }

  bool fitsStorageBlock(std::size_t bytes) const
  {
    return bytes <= static_cast<std::size_t>(max_storage_block_bytes_);
  }

  static GLuint groups(std::uint32_t count, std::uint32_t width)
  {
    return (count + width - 1u) / width;
  }

  void use(GLuint program)
  {
    current_program_ = program;
    glUseProgram(program);
  }

  static void bind(GLuint binding, GLuint buffer)
  {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffer);
  }

  void uniform(const char * name, std::uint32_t value)
  {
    const GLint location = glGetUniformLocation(current_program_, name);
    if (location >= 0) {
      glUniform1ui(location, value);
    }
  }

  static void storageBarrier()
  {
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  }

  static void dispatchIndirect(GLuint buffer)
  {
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buffer);
    glDispatchComputeIndirect(0);
  }

  void pollReadbacks()
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
  }

  void queueReadback()
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

  Ogre::GLRenderSystem * render_system_ = nullptr;
  const Ogre::Renderable * renderable_ = nullptr;
  Ogre::HardwareVertexBufferSharedPtr instance_buffer_;
  Ogre::HardwareVertexBufferSharedPtr indirect_buffer_;
  std::string shader_source_;
  std::string error_;
  std::array<GLuint, 11> programs_{};
  bool profiling_ = false;
  std::array<GLuint, 5> queries_{};
  StageTimes times_;
  GLuint current_program_ = 0u;
  GLint64 max_storage_block_bytes_ = 0;
  std::uint32_t count_ = 0u;
  std::uint32_t tile_count_ = 0u;
  std::uint32_t index_count_ = 0u;
  GLuint instance_gl_ = 0u;
  GLuint indirect_gl_ = 0u;
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
  // One colour per splat, by splat index, from SHADE_VISIBLE for the gather.
  GLuint colours_ = 0u;
  // The draw arguments when Ogre cannot use them; otherwise Ogre owns them.
  GLuint draw_arguments_ = 0u;
  GLuint view_parameters_ = 0u;
  GLuint projection_parameters_ = 0u;
  std::array<Readback, 3> readbacks_{};
  std::size_t readback_cursor_ = 0u;
  std::int64_t last_visible_ = -1;
};

}  // namespace

std::unique_ptr<GpuViewPreparation> makeOpenGlViewPreparation(
  Ogre::RenderSystem * render_system, const std::string & shader_path)
{
  if (!render_system ||
    render_system->getName().find("OpenGL Rendering Subsystem") == std::string::npos)
  {
    return nullptr;
  }
  return std::make_unique<OpenGlViewPreparation>(
    static_cast<Ogre::GLRenderSystem *>(render_system), shader_path);
}

}  // namespace gaussian_splatting_rviz_plugins
