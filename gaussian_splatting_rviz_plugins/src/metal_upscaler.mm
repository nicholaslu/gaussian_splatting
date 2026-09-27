#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include "gaussian_splatting_rviz_plugins/metal_upscaler.hpp"

#include <dlfcn.h>

#include <memory>
#include <string>

#include <OgreHardwarePixelBuffer.h>
#include <OgreMetalDevice.h>
#include <OgreMetalRenderSystem.h>
#include <OgreMetalRenderTargetCommon.h>
#include <OgreRenderTexture.h>
#include <OgreTexture.h>

namespace gaussian_splatting_rviz_plugins
{
namespace
{

// MetalFX is looked up at run time rather than linked, so that an application
// that links this display statically needs no change to its link line, and a
// system without MetalFX simply goes without.
Class spatialScalerDescriptorClass()
{
  static Class found = [] {
      dlopen("/System/Library/Frameworks/MetalFX.framework/MetalFX", RTLD_LAZY | RTLD_LOCAL);
      return NSClassFromString(@"MTLFXSpatialScalerDescriptor");
    }();
  return found;
}

// The Metal texture a render texture's colour is read from: its resolved image
// when it is multisampled. Taken from the render target's colour attachment,
// which Ogre exposes, since MetalTexture's own accessors are not exported from
// the render system.
id<MTLTexture> colourTexture(Ogre::Texture * texture)
{
  Ogre::RenderTexture * target = texture->getBuffer()->getRenderTarget();
  Ogre::MetalRenderTargetCommon * common = nullptr;
  target->getCustomAttribute("MetalRenderTargetCommon", &common);
  if (!common || !common->mColourAttachmentDesc) {
    return nil;
  }
  MTLRenderPassColorAttachmentDescriptor * attachment = common->mColourAttachmentDesc;
  return attachment.resolveTexture ? attachment.resolveTexture : attachment.texture;
}

class MetalFxUpscaler final : public MetalUpscaler
{
public:
  MetalFxUpscaler(Ogre::MetalDevice * device, id<MTLFXSpatialScaler> scaler)
  : device_(device), scaler_(scaler)
  {
  }

  bool encode() override
  {
    id<MTLCommandBuffer> command_buffer = device_->mCurrentCommandBuffer;
    if (!command_buffer) {
      return false;
    }
    // MetalFX encodes its own passes, so no encoder of Ogre's may be open.
    device_->endAllEncoders();
    [scaler_ encodeToCommandBuffer:command_buffer];
    return true;
  }

private:
  Ogre::MetalDevice * device_;
  id<MTLFXSpatialScaler> scaler_;
};

}  // namespace

std::unique_ptr<MetalUpscaler> makeMetalUpscaler(
  Ogre::RenderSystem * render_system, Ogre::Texture * input, Ogre::Texture * output,
  std::string & reason)
{
  if (!render_system || render_system->getName().find("Metal") == std::string::npos) {
    reason = "MetalFX needs the Metal render system";
    return nullptr;
  }
  auto * metal = static_cast<Ogre::MetalRenderSystem *>(render_system);
  Ogre::MetalDevice * device = metal->getActiveDevice();
  if (!device || !device->mDevice || !input || !output) {
    reason = "the Metal device or the textures are unavailable";
    return nullptr;
  }
  Class descriptor_class = spatialScalerDescriptorClass();
  if (!descriptor_class) {
    reason = "MetalFX is not available on this system";
    return nullptr;
  }
  if (![descriptor_class supportsDevice:device->mDevice]) {
    reason = "MetalFX spatial upscaling is not supported on this GPU";
    return nullptr;
  }
  id<MTLTexture> input_texture = colourTexture(input);
  id<MTLTexture> output_texture = colourTexture(output);
  if (!input_texture || !output_texture) {
    reason = "Ogre did not expose the Metal textures";
    return nullptr;
  }

  MTLFXSpatialScalerDescriptor * descriptor = [[descriptor_class alloc] init];
  descriptor.colorTextureFormat = input_texture.pixelFormat;
  descriptor.outputTextureFormat = output_texture.pixelFormat;
  descriptor.inputWidth = input_texture.width;
  descriptor.inputHeight = input_texture.height;
  descriptor.outputWidth = output_texture.width;
  descriptor.outputHeight = output_texture.height;
  // The splats are drawn in display values, not linear light.
  descriptor.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
  id<MTLFXSpatialScaler> scaler = [descriptor newSpatialScalerWithDevice:device->mDevice];
  if (!scaler) {
    reason = "MetalFX could not create a spatial scaler for these textures";
    return nullptr;
  }
  if ((input_texture.usage & scaler.colorTextureUsage) != scaler.colorTextureUsage ||
    (output_texture.usage & scaler.outputTextureUsage) != scaler.outputTextureUsage)
  {
    reason = "the render textures lack the usage MetalFX needs";
    return nullptr;
  }
  scaler.colorTexture = input_texture;
  scaler.outputTexture = output_texture;
  scaler.inputContentWidth = input_texture.width;
  scaler.inputContentHeight = input_texture.height;
  reason.clear();
  return std::make_unique<MetalFxUpscaler>(device, scaler);
}

}  // namespace gaussian_splatting_rviz_plugins
