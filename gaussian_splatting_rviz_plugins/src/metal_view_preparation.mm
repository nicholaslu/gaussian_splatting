#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gaussian_splatting_rviz_plugins/metal_view_preparation.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

#include <OgreHardwareBufferManager.h>
#include <OgreMetalDevice.h>
#include <OgreMetalHardwareBufferManager.h>
#include <OgreMetalHardwareVertexBuffer.h>
#include <OgreMetalRenderSystem.h>
#include <OgreRenderable.h>

namespace gaussian_splatting_rviz_plugins
{
namespace
{

constexpr std::uint32_t kThreadsPerTile = 32;
constexpr std::uint32_t kElementsPerTile = 1024;
constexpr std::uint32_t kRadix = 256;

struct RadixParameters
{
  std::uint32_t count;
  std::uint32_t tile_count;
  std::uint32_t elements_per_tile;
  std::uint32_t shift;
};

class MetalViewPreparationImpl final : public MetalViewPreparation
{
public:
  MetalViewPreparationImpl(Ogre::MetalRenderSystem * render_system, const std::string & shader_path)
  : render_system_(render_system), device_(render_system ? render_system->getActiveDevice() : nullptr)
  {
    if (!device_ || !device_->mDevice) {
      error_ = "Ogre Metal device is unavailable";
      return;
    }

    @autoreleasepool {
      NSError * error = nil;
      NSString * path = [NSString stringWithUTF8String:shader_path.c_str()];
      NSString * source = [NSString stringWithContentsOfFile:path
                                                    encoding:NSUTF8StringEncoding
                                                       error:&error];
      if (!source) {
        setError("reading Metal preparation shader", error);
        return;
      }

      MTLCompileOptions * options = [MTLCompileOptions new];
      library_ = [device_->mDevice newLibraryWithSource:source options:options error:&error];
      if (!library_) {
        setError("compiling Metal preparation shader", error);
        return;
      }

      reset_pipeline_ = makePipeline(@"reset_view");
      key_pipeline_ = makePipeline(@"make_depth_keys");
      histogram_pipeline_ = makePipeline(@"radix_histogram");
      offset_pipeline_ = makePipeline(@"radix_scan_offsets");
      base_pipeline_ = makePipeline(@"radix_scan_digit_bases");
      scatter_pipeline_ = makePipeline(@"radix_scatter");
      gather_pipeline_ = makePipeline(@"gather_instances");
    }
  }

  ~MetalViewPreparationImpl() override
  {
    clear();
  }

  bool ready() const
  {
    return reset_pipeline_ && key_pipeline_ && histogram_pipeline_ && offset_pipeline_ &&
           base_pipeline_ && scatter_pipeline_ && gather_pipeline_;
  }

  bool configure(
    std::size_t count,
    const Ogre::HardwareVertexBufferSharedPtr & instance_buffer,
    const Ogre::Renderable * renderable) override
  {
    clear();
    if (!ready()) {
      return false;
    }
    if (count == 0 || count > std::numeric_limits<std::uint32_t>::max()) {
      error_ = "Metal preparation requires a non-zero uint32 splat count";
      return false;
    }

    count_ = static_cast<std::uint32_t>(count);
    tile_count_ = (count_ + kElementsPerTile - 1u) / kElementsPerTile;
    instance_buffer_ = instance_buffer;
    renderable_ = renderable;

    const NSUInteger record_bytes = count * sizeof(std::uint32_t) * 2u;
    const NSUInteger table_bytes =
      static_cast<NSUInteger>(tile_count_) * kRadix * sizeof(std::uint32_t);
    keys_a_ = newBuffer(record_bytes, "depth keys A");
    keys_b_ = newBuffer(record_bytes, "depth keys B");
    histograms_ = newBuffer(table_bytes, "radix histograms");
    offsets_ = newBuffer(table_bytes, "radix offsets");
    totals_ = newBuffer(kRadix * sizeof(std::uint32_t), "radix totals");
    digit_bases_ = newBuffer(kRadix * sizeof(std::uint32_t), "radix digit bases");
    visible_count_ = newBuffer(sizeof(std::uint32_t), "visible count");
    if (!keys_a_ || !keys_b_ || !histograms_ || !offsets_ || !totals_ ||
      !digit_bases_ || !visible_count_)
    {
      clear();
      return false;
    }

    auto & manager = Ogre::HardwareBufferManager::getSingleton();
    indirect_buffer_ = manager.createVertexBuffer(
      sizeof(std::uint32_t) * 5u, 1u, Ogre::HardwareBuffer::HBU_DYNAMIC_WRITE_ONLY);
    if (!indirect_buffer_) {
      error_ = "could not allocate Ogre indirect draw buffer";
      clear();
      return false;
    }
    render_system_->setIndirectDrawBuffer(renderable_, indirect_buffer_);
    error_.clear();
    return true;
  }

  bool uploadStaticData(
    const void * records, std::size_t record_bytes,
    const float * sh_dc, std::size_t sh_dc_floats,
    const float * sh_rest, std::size_t sh_rest_floats) override
  {
    if (!ready() || count_ == 0 || !records || !sh_dc) {
      error_ = "Metal static upload called before configuration";
      return false;
    }

    // Allocate replacements instead of overwriting buffers which may still be
    // referenced by an in-flight Ogre command buffer.
    id<MTLBuffer> splats = newBuffer(record_bytes, "splat records");
    id<MTLBuffer> dc = newBuffer(sh_dc_floats * sizeof(float), "SH DC");
    const std::size_t rest_bytes = std::max<std::size_t>(
      sh_rest_floats * sizeof(float), sizeof(float));
    id<MTLBuffer> rest = newBuffer(rest_bytes, "SH rest");
    if (!splats || !dc || !rest) {
      return false;
    }
    std::memcpy(splats.contents, records, record_bytes);
    std::memcpy(dc.contents, sh_dc, sh_dc_floats * sizeof(float));
    if (sh_rest && sh_rest_floats > 0) {
      std::memcpy(rest.contents, sh_rest, sh_rest_floats * sizeof(float));
    } else {
      std::memset(rest.contents, 0, rest_bytes);
    }
    splats_ = splats;
    sh_dc_ = dc;
    sh_rest_ = rest;
    error_.clear();
    return true;
  }

  bool encode(const MetalViewParameters & parameters) override
  {
    if (!ready() || !splats_ || !sh_dc_ || !sh_rest_ || !instance_buffer_ ||
      !indirect_buffer_ || parameters.count != count_)
    {
      error_ = "Metal view preparation is not fully configured";
      return false;
    }

    auto * metal_instances = static_cast<Ogre::MetalHardwareVertexBuffer *>(
      instance_buffer_.get());
    auto * metal_indirect = static_cast<Ogre::MetalHardwareVertexBuffer *>(
      indirect_buffer_.get());
    id<MTLBuffer> instances = metal_instances->getBufferNameForGpuWrite();
    id<MTLBuffer> indirect = metal_indirect->getBufferNameForGpuWrite();
    if (!instances || !indirect) {
      error_ = "Ogre did not expose the Metal output buffers";
      return false;
    }

    id<MTLComputeCommandEncoder> encoder = device_->getComputeEncoder();
    if (!encoder) {
      error_ = "Ogre did not provide a Metal compute encoder";
      return false;
    }
    encoder.label = @"Gaussian splat GPU view preparation";

    [encoder setComputePipelineState:reset_pipeline_];
    [encoder setBuffer:visible_count_ offset:0 atIndex:0];
    [encoder setBuffer:indirect offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    barrier(encoder);

    [encoder setComputePipelineState:key_pipeline_];
    [encoder setBuffer:splats_ offset:0 atIndex:0];
    [encoder setBuffer:keys_a_ offset:0 atIndex:1];
    [encoder setBuffer:visible_count_ offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    dispatchLinear(encoder, key_pipeline_, count_);
    barrier(encoder);

    id<MTLBuffer> input = keys_a_;
    id<MTLBuffer> output = keys_b_;
    for (std::uint32_t shift = 0; shift < 32u; shift += 8u) {
      const RadixParameters radix{count_, tile_count_, kElementsPerTile, shift};

      [encoder setComputePipelineState:histogram_pipeline_];
      [encoder setBuffer:input offset:0 atIndex:0];
      [encoder setBuffer:histograms_ offset:0 atIndex:1];
      [encoder setBytes:&radix length:sizeof(radix) atIndex:2];
      [encoder dispatchThreadgroups:MTLSizeMake(tile_count_, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(kThreadsPerTile, 1, 1)];
      barrier(encoder);

      [encoder setComputePipelineState:offset_pipeline_];
      [encoder setBuffer:histograms_ offset:0 atIndex:0];
      [encoder setBuffer:offsets_ offset:0 atIndex:1];
      [encoder setBuffer:totals_ offset:0 atIndex:2];
      [encoder setBytes:&radix length:sizeof(radix) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(kRadix, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      barrier(encoder);

      [encoder setComputePipelineState:base_pipeline_];
      [encoder setBuffer:totals_ offset:0 atIndex:0];
      [encoder setBuffer:digit_bases_ offset:0 atIndex:1];
      [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      barrier(encoder);

      [encoder setComputePipelineState:scatter_pipeline_];
      [encoder setBuffer:input offset:0 atIndex:0];
      [encoder setBuffer:output offset:0 atIndex:1];
      [encoder setBuffer:offsets_ offset:0 atIndex:2];
      [encoder setBuffer:digit_bases_ offset:0 atIndex:3];
      [encoder setBytes:&radix length:sizeof(radix) atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake(tile_count_, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(kThreadsPerTile, 1, 1)];
      barrier(encoder);
      std::swap(input, output);
    }

    [encoder setComputePipelineState:gather_pipeline_];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:splats_ offset:0 atIndex:1];
    [encoder setBuffer:sh_dc_ offset:0 atIndex:2];
    [encoder setBuffer:sh_rest_ offset:0 atIndex:3];
    [encoder setBuffer:visible_count_ offset:0 atIndex:4];
    [encoder setBuffer:instances offset:0 atIndex:5];
    [encoder setBuffer:indirect offset:0 atIndex:6];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:7];
    dispatchLinear(encoder, gather_pipeline_, count_);
    // A later Ogre render encoder consumes both buffers. Ending the compute
    // encoder is performed automatically when Ogre asks its device for that
    // render encoder, preserving order in the same command buffer.
    error_.clear();
    return true;
  }

  void clear() override
  {
    if (render_system_ && renderable_) {
      Ogre::HardwareVertexBufferSharedPtr none;
      render_system_->setIndirectDrawBuffer(renderable_, none);
    }
    renderable_ = nullptr;
    count_ = 0;
    tile_count_ = 0;
    instance_buffer_.reset();
    indirect_buffer_.reset();
    splats_ = nil;
    sh_dc_ = nil;
    sh_rest_ = nil;
    keys_a_ = nil;
    keys_b_ = nil;
    histograms_ = nil;
    offsets_ = nil;
    totals_ = nil;
    digit_bases_ = nil;
    visible_count_ = nil;
  }

  const std::string & error() const override
  {
    return error_;
  }

private:
  id<MTLComputePipelineState> makePipeline(NSString * name)
  {
    NSError * error = nil;
    id<MTLFunction> function = [library_ newFunctionWithName:name];
    if (!function) {
      error_ = "Metal preparation shader has no function " + std::string(name.UTF8String);
      return nil;
    }
    id<MTLComputePipelineState> pipeline =
      [device_->mDevice newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
      setError(std::string("creating pipeline ") + name.UTF8String, error);
    }
    return pipeline;
  }

  id<MTLBuffer> newBuffer(NSUInteger size, const char * label)
  {
    id<MTLBuffer> buffer = [device_->mDevice newBufferWithLength:std::max<NSUInteger>(size, 4u)
                                                       options:MTLResourceStorageModeShared];
    if (!buffer) {
      error_ = "Metal allocation failed for " + std::string(label) + " (" +
        std::to_string(size) + " bytes)";
    } else {
      buffer.label = [NSString stringWithUTF8String:label];
    }
    return buffer;
  }

  static void barrier(id<MTLComputeCommandEncoder> encoder)
  {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }

  static void dispatchLinear(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger count)
  {
    const NSUInteger width = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256u);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  }

  void setError(const std::string & operation, NSError * error)
  {
    error_ = operation;
    if (error) {
      error_ += ": ";
      error_ += error.localizedDescription.UTF8String;
    }
  }

  Ogre::MetalRenderSystem * render_system_ = nullptr;
  Ogre::MetalDevice * device_ = nullptr;
  const Ogre::Renderable * renderable_ = nullptr;
  Ogre::HardwareVertexBufferSharedPtr instance_buffer_;
  Ogre::HardwareVertexBufferSharedPtr indirect_buffer_;
  std::uint32_t count_ = 0;
  std::uint32_t tile_count_ = 0;
  std::string error_;

  id<MTLLibrary> library_ = nil;
  id<MTLComputePipelineState> reset_pipeline_ = nil;
  id<MTLComputePipelineState> key_pipeline_ = nil;
  id<MTLComputePipelineState> histogram_pipeline_ = nil;
  id<MTLComputePipelineState> offset_pipeline_ = nil;
  id<MTLComputePipelineState> base_pipeline_ = nil;
  id<MTLComputePipelineState> scatter_pipeline_ = nil;
  id<MTLComputePipelineState> gather_pipeline_ = nil;

  id<MTLBuffer> splats_ = nil;
  id<MTLBuffer> sh_dc_ = nil;
  id<MTLBuffer> sh_rest_ = nil;
  id<MTLBuffer> keys_a_ = nil;
  id<MTLBuffer> keys_b_ = nil;
  id<MTLBuffer> histograms_ = nil;
  id<MTLBuffer> offsets_ = nil;
  id<MTLBuffer> totals_ = nil;
  id<MTLBuffer> digit_bases_ = nil;
  id<MTLBuffer> visible_count_ = nil;
};

}  // namespace

std::unique_ptr<MetalViewPreparation> makeMetalViewPreparation(
  Ogre::RenderSystem * render_system, const std::string & shader_path)
{
  if (!render_system || render_system->getName().find("Metal") == std::string::npos) {
    return nullptr;
  }
  // Checking the selected backend before this cast avoids an RTTI dependency
  // when the weak-linked Metal plug-in is absent in an OpenGL process.
  auto * metal = static_cast<Ogre::MetalRenderSystem *>(render_system);
  auto preparation = std::make_unique<MetalViewPreparationImpl>(metal, shader_path);
  if (!preparation->ready()) {
    return preparation;
  }
  return preparation;
}

}  // namespace gaussian_splatting_rviz_plugins
