#include "gaussian_splatting_rviz_plugins/opengl_view_preparation.hpp"
#include "opengl_view_preparation_core.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>

#include <RenderSystems/GL/OgreGLHardwareVertexBuffer.h>
#include <RenderSystems/GL/OgreGLRenderSystem.h>
#include <OgreHardwareBufferManager.h>
#include <OgreRenderable.h>

namespace gaussian_splatting_rviz_plugins
{
namespace
{

#ifdef GSPLAT_OGRE_GL_INDIRECT_DRAW
constexpr bool kOgreIndirectDraw = true;
#else
// Upstream Ogre draws a renderable with the instance count the CPU gives it, so
// every instance is drawn and those past the visible count are left empty.
constexpr bool kOgreIndirectDraw = false;
#endif

struct DrawIndexedArguments
{
  std::uint32_t index_count;
  std::uint32_t instance_count;
  std::uint32_t index_start;
  std::uint32_t base_vertex;
  std::uint32_t base_instance;
};

std::string readFile(const std::string & path)
{
  std::ifstream stream(path);
  if (!stream) {
    return {};
  }
  return std::string(
    std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

// Set and not "0", as for the Metal backend.
bool environmentFlag(const char * name)
{
  const char * value = std::getenv(name);
  return value && *value && std::strcmp(value, "0") != 0;
}

// Takes the instance buffer, and the indirect draw buffer where Ogre has one,
// from Ogre and hands the work to OpenGlViewPreparationCore, which the offline
// verifier drives directly.
class OpenGlViewPreparation final : public GpuViewPreparation
{
public:
  OpenGlViewPreparation(Ogre::GLRenderSystem * render_system, const std::string & shader_path)
  : render_system_(render_system)
  {
    if (!render_system_) {
      error_ = "Ogre OpenGL render system is unavailable";
      return;
    }
    const std::string source = readFile(shader_path);
    if (source.empty()) {
      error_ = "could not read OpenGL preparation shader " + shader_path;
      return;
    }
    core_ = std::make_unique<OpenGlViewPreparationCore>(source);
    if (!core_->ready()) {
      error_ = core_->error();
      return;
    }
    core_->setProfiling(environmentFlag("GSPLAT_PROFILE_GPU"));
  }

  ~OpenGlViewPreparation() override
  {
    clear();
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
    if (!core_ || !core_->ready()) {
      error_ = core_ ? core_->error() : error_;
      return false;
    }
    if (count == 0 || count > std::numeric_limits<std::uint32_t>::max() ||
      !instance_buffer || !renderable)
    {
      error_ = "OpenGL preparation needs a non-zero uint32 splat count, an instance "
        "buffer and a renderable";
      return false;
    }
    if (!core_->configure(static_cast<std::uint32_t>(count), index_count)) {
      error_ = core_->error();
      return false;
    }

    instance_buffer_ = instance_buffer;
    instance_gl_ =
      static_cast<Ogre::GLHardwareVertexBuffer *>(instance_buffer_.get())->getGLBufferId();
    renderable_ = renderable;
#ifdef GSPLAT_OGRE_GL_INDIRECT_DRAW
    auto & manager = Ogre::HardwareBufferManager::getSingleton();
    indirect_buffer_ = manager.createVertexBuffer(
      sizeof(DrawIndexedArguments), 1u, Ogre::HardwareBuffer::HBU_DYNAMIC_WRITE_ONLY);
    // Drawn from before the first preparation completes, so it has to describe
    // an empty draw rather than whatever the allocation held.
    const DrawIndexedArguments empty{index_count, 0u, 0u, 0u, 0u};
    indirect_buffer_->writeData(0, sizeof(empty), &empty, true);
    indirect_gl_ =
      static_cast<Ogre::GLHardwareVertexBuffer *>(indirect_buffer_.get())->getGLBufferId();
    render_system_->setIndirectDrawBuffer(renderable_, indirect_buffer_);
#else
    // The core writes draw arguments into its own buffer; nothing reads them.
    indirect_gl_ = 0u;
#endif
    error_.clear();
    return true;
  }

  bool uploadStaticData(
    const SplatRecord * records, std::size_t record_count,
    const float * sh_dc, std::size_t sh_dc_floats,
    const float * sh_rest, std::size_t sh_rest_floats) override
  {
    if (!core_ || !core_->uploadStaticData(
        records, record_count, sh_dc, sh_dc_floats, sh_rest, sh_rest_floats))
    {
      error_ = core_ ? core_->error() : "OpenGL preparation core is unavailable";
      return false;
    }
    error_.clear();
    return true;
  }

  bool encode(
    const ViewParameters & parameters, const ProjectionParameters & projection) override
  {
    if (!core_ || !instance_gl_) {
      error_ = "OpenGL view preparation is not fully configured";
      return false;
    }
    if (!core_->encode(parameters, projection, instance_gl_, indirect_gl_, !kOgreIndirectDraw)) {
      error_ = core_->error();
      return false;
    }
    error_.clear();
    return true;
  }

  std::int64_t lastVisibleCount() const override
  {
    return core_ ? core_->lastVisibleCount() : -1;
  }

  bool drawsEveryInstance() const override
  {
    return !kOgreIndirectDraw;
  }

  bool profiling() const override
  {
    return core_ && core_->profiling();
  }

  StageTimes lastStageTimes() const override
  {
    StageTimes times;
    if (core_) {
      const OpenGlViewPreparationCore::Timings & timings = core_->lastTimings();
      times.cull_ms = timings.cull_ms;
      times.shade_ms = timings.shade_ms;
      times.sort_ms = timings.sort_ms;
      times.gather_ms = timings.gather_ms;
    }
    return times;
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
    if (core_) {
      core_->clear();
    }
  }

  const std::string & error() const override
  {
    return error_;
  }

private:
  Ogre::GLRenderSystem * render_system_ = nullptr;
  std::unique_ptr<OpenGlViewPreparationCore> core_;
  const Ogre::Renderable * renderable_ = nullptr;
  Ogre::HardwareVertexBufferSharedPtr instance_buffer_;
  Ogre::HardwareVertexBufferSharedPtr indirect_buffer_;
  GLuint instance_gl_ = 0u;
  GLuint indirect_gl_ = 0u;
  std::string error_;
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
