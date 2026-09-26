#pragma once

#include <memory>
#include <string>

#include "gaussian_splatting_rviz_plugins/gpu_view_preparation.hpp"

namespace Ogre
{
class RenderSystem;
}

namespace gaussian_splatting_rviz_plugins
{

// Reserved CUDA/CUB interop factory. A future implementation should register
// the Ogre OpenGL instance and indirect buffers with CUDA and implement this
// same contract; the display and projected raster material need no changes.
std::unique_ptr<GpuViewPreparation> makeCudaViewPreparation(
  Ogre::RenderSystem * render_system, const std::string & shader_root);

}  // namespace gaussian_splatting_rviz_plugins
