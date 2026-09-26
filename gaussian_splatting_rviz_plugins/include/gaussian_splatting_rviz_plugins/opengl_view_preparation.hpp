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

std::unique_ptr<GpuViewPreparation> makeOpenGlViewPreparation(
  Ogre::RenderSystem * render_system, const std::string & shader_path);

}  // namespace gaussian_splatting_rviz_plugins
