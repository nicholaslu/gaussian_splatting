#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <OgreHardwareVertexBuffer.h>

#include "gaussian_splatting_rviz_plugins/splat_view.hpp"

namespace Ogre
{
class Renderable;
class RenderSystem;
}

namespace gaussian_splatting_rviz_plugins
{

// Renderer-neutral contract for preparing one camera view entirely on the
// GPU. Metal, OpenGL compute and a future CUDA/OpenGL interop implementation
// all produce the same ProjectedInstance stream and indirect draw arguments.
class GpuViewPreparation
{
public:
  struct StageTimes
  {
    double cull_ms = 0.0;
    double shade_ms = 0.0;
    double sort_ms = 0.0;
    double gather_ms = 0.0;
  };

  virtual ~GpuViewPreparation() = default;

  virtual const char * backendName() const = 0;

  virtual bool configure(
    std::size_t count, std::uint32_t index_count,
    const Ogre::HardwareVertexBufferSharedPtr & instance_buffer,
    const Ogre::Renderable * renderable) = 0;

  virtual bool uploadStaticData(
    const SplatRecord * records, std::size_t record_count,
    const float * sh_dc, std::size_t sh_dc_floats,
    const float * sh_rest, std::size_t sh_rest_floats) = 0;

  virtual bool encode(
    const ViewParameters & parameters, const ProjectionParameters & projection) = 0;

  // Implementations may update this asynchronously. A negative value means
  // that no completed preparation has reported a count yet.
  virtual std::int64_t lastVisibleCount() const = 0;

  // True when the renderer cannot take the instance count from the GPU-written
  // draw arguments, so every instance is drawn and those past the visible count
  // are written as empty quads. The display then draws all of them rather than
  // the visible count, which arrives frames late.
  virtual bool drawsEveryInstance() const {return false;}

  virtual bool profiling() const = 0;
  virtual StageTimes lastStageTimes() const = 0;

  virtual void clear() = 0;
  virtual const std::string & error() const = 0;
};

// Selects a compiled backend for the active Ogre renderer. The optional
// GSPLAT_GPU_BACKEND value is auto, metal, opengl, cuda, or cpu. CUDA is a
// reserved factory slot: enabling it later does not change the display-facing
// contract or the Metal/OpenGL implementations.
std::unique_ptr<GpuViewPreparation> makeGpuViewPreparation(
  Ogre::RenderSystem * render_system, const std::string & shader_root,
  std::string & fallback_reason);

}  // namespace gaussian_splatting_rviz_plugins
