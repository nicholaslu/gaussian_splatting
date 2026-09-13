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

// The display's view of GPU preparation. Backend neutral, so the display can
// hold one in every build; only makeMetalViewPreparation() is Metal specific,
// and exists only when GSPLAT_HAS_METAL_PREPARATION is defined.
class MetalViewPreparation
{
public:
  // GPU time of each stage. Measured only in profiling mode.
  struct StageTimes
  {
    double cull_ms = 0.0;
    double sort_ms = 0.0;
    double gather_ms = 0.0;
  };

  virtual ~MetalViewPreparation() = default;

  virtual bool configure(
    std::size_t count, std::uint32_t index_count,
    const Ogre::HardwareVertexBufferSharedPtr & instance_buffer,
    const Ogre::Renderable * renderable) = 0;

  virtual bool uploadStaticData(
    const SplatRecord * records, std::size_t record_count,
    const float * sh_dc, std::size_t sh_dc_floats,
    const float * sh_rest, std::size_t sh_rest_floats) = 0;

  // Enqueues one preparation on Ogre's current command buffer, writing each
  // instance projected with `projection`; nothing is
  // committed and nothing is read back. With GSPLAT_PROFILE_GPU set, the stages
  // instead run in command buffers of their own that are waited on so each can
  // be timed, which stalls the render thread and is for measurement only.
  virtual bool encode(
    const ViewParameters & parameters, const ProjectionParameters & projection) = 0;

  // The visible count of the most recent preparation the GPU has finished.
  // It arrives from a completion handler, so it trails the frame being
  // prepared; negative until one has finished.
  virtual std::int64_t lastVisibleCount() const = 0;

  virtual bool profiling() const = 0;
  virtual StageTimes lastStageTimes() const = 0;

  virtual void clear() = 0;
  virtual const std::string & error() const = 0;
};

std::unique_ptr<MetalViewPreparation> makeMetalViewPreparation(
  Ogre::RenderSystem * render_system, const std::string & shader_path);

}  // namespace gaussian_splatting_rviz_plugins
