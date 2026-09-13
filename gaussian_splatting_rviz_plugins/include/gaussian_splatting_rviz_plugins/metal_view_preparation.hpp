#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <OgreHardwareVertexBuffer.h>

namespace Ogre
{
class Renderable;
class RenderSystem;
}

namespace gaussian_splatting_rviz_plugins
{

// Plain, explicitly aligned contract shared with gsplat_prepare.metal. Matrices
// are row-major so the shader performs the same row dot-products as Ogre.
struct alignas(16) MetalViewParameters
{
  float world_rows[16];
  float view_rows[16];
  float frustum_planes[24];
  float local_camera[4];
  float local_sort_direction[4];
  float world_scale;
  float focal_y;
  float near_clip;
  float far_clip;
  float ortho_scale;
  float sigma_radius;
  float min_screen_radius;
  float eps2d;
  std::uint32_t count;
  std::uint32_t viewport_height;
  std::uint32_t sh_coefficients;
  std::uint32_t flags;
};

static_assert(sizeof(MetalViewParameters) == 304, "Metal parameter layout changed");

class MetalViewPreparation
{
public:
  virtual ~MetalViewPreparation() = default;

  virtual bool configure(
    std::size_t count,
    const Ogre::HardwareVertexBufferSharedPtr & instance_buffer,
    const Ogre::Renderable * renderable) = 0;

  virtual bool uploadStaticData(
    const void * records, std::size_t record_bytes,
    const float * sh_dc, std::size_t sh_dc_floats,
    const float * sh_rest, std::size_t sh_rest_floats) = 0;

  // Enqueues work on Ogre's current Metal command buffer. No command buffer is
  // committed and no result is read back on the CPU.
  virtual bool encode(const MetalViewParameters & parameters) = 0;
  virtual void clear() = 0;
  virtual const std::string & error() const = 0;
};

std::unique_ptr<MetalViewPreparation> makeMetalViewPreparation(
  Ogre::RenderSystem * render_system, const std::string & shader_path);

}  // namespace gaussian_splatting_rviz_plugins
