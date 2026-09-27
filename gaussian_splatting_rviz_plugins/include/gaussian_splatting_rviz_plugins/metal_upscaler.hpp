#pragma once

#include <memory>
#include <string>

namespace Ogre
{
class RenderSystem;
class Texture;
}

namespace gaussian_splatting_rviz_plugins
{

// MetalFX spatial upscaling of the offscreen splat target into a larger render
// texture, encoded into the frame's command buffer after the target is drawn.
// MetalFX writes an alpha of one everywhere, so the composite takes colour
// from the output and alpha from the input.
class MetalUpscaler
{
public:
  virtual ~MetalUpscaler() = default;
  // Upscales the input as it stands now into the output.
  virtual bool encode() = 0;
};

// Null, with `reason` set, where MetalFX or the device cannot do it. `input`
// and `output` are render textures of one pixel format, which outlive the
// upscaler.
std::unique_ptr<MetalUpscaler> makeMetalUpscaler(
  Ogre::RenderSystem * render_system, Ogre::Texture * input, Ogre::Texture * output,
  std::string & reason);

}  // namespace gaussian_splatting_rviz_plugins
