#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gaussian_splatting_rviz_plugins/metal_view_preparation.hpp"
#include "metal_view_preparation_core.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

#include <OgreHardwareBufferManager.h>
#include <OgreMetalDevice.h>
// Declares the manager OgreMetalHardwareVertexBuffer.h names without including.
#include <OgreMetalHardwareBufferManager.h>
#include <OgreMetalHardwareVertexBuffer.h>
#include <OgreMetalRenderSystem.h>
#include <OgreRenderable.h>

namespace gaussian_splatting_rviz_plugins
{
namespace
{

// Written from Metal's completion thread and read on the render thread. The
// generation lets clear() disown a count still on its way from a preparation
// of the previous scene.
struct Report
{
  std::atomic<std::int64_t> visible{-1};
  std::atomic<std::uint64_t> generation{0};
};

// Set and not "0": GSPLAT_PROFILE_GPU=0 means off, not on.
bool environmentFlag(const char * name)
{
  const char * value = std::getenv(name);
  return value && *value && std::strcmp(value, "0") != 0;
}

struct DrawIndexedArguments
{
  std::uint32_t index_count;
  std::uint32_t instance_count;
  std::uint32_t index_start;
  std::int32_t base_vertex;
  std::uint32_t base_instance;
};

// Takes the device, the command buffer and the output buffers from Ogre and
// hands the work to MetalViewPreparationCore, which the offline verifier
// drives directly.
class MetalViewPreparationImpl final : public MetalViewPreparation
{
public:
  MetalViewPreparationImpl(Ogre::MetalRenderSystem * render_system, const std::string & shader_path)
  : render_system_(render_system),
    device_(render_system ? render_system->getActiveDevice() : nullptr),
    report_(std::make_shared<Report>()),
    profiling_(environmentFlag("GSPLAT_PROFILE_GPU")),
    debug_projection_(environmentFlag("GSPLAT_DEBUG_PROJECTION"))
  {
    if (!device_ || !device_->mDevice) {
      error_ = "Ogre Metal device is unavailable";
      return;
    }
    core_ = std::make_unique<MetalViewPreparationCore>(device_->mDevice, shader_path);
    if (!core_->ready()) {
      error_ = core_->error();
    }
  }

  ~MetalViewPreparationImpl() override
  {
    clear();
  }

  bool configure(
    std::size_t count, std::uint32_t index_count,
    const Ogre::HardwareVertexBufferSharedPtr & instance_buffer,
    const Ogre::Renderable * renderable) override
  {
    clear();
    if (!core_ || !core_->ready()) {
      error_ = core_ ? core_->error() : "Metal preparation core is unavailable";
      return false;
    }
    if (count == 0 || count > std::numeric_limits<std::uint32_t>::max() || !instance_buffer ||
      !renderable)
    {
      error_ = "Metal preparation needs a non-zero uint32 splat count, an instance buffer "
        "and a renderable";
      return false;
    }
    if (!core_->configure(static_cast<std::uint32_t>(count), index_count)) {
      error_ = core_->error();
      return false;
    }

    auto & manager = Ogre::HardwareBufferManager::getSingleton();
    indirect_buffer_ = manager.createVertexBuffer(
      sizeof(DrawIndexedArguments), 1u, Ogre::HardwareBuffer::HBU_DYNAMIC_WRITE_ONLY);
    if (!indirect_buffer_) {
      error_ = "could not allocate Ogre indirect draw buffer";
      core_->clear();
      return false;
    }
    // Drawn from before the first preparation completes, so it has to describe
    // an empty draw rather than whatever the allocation held.
    const DrawIndexedArguments empty{index_count, 0u, 0u, 0, 0u};
    indirect_buffer_->writeData(0, sizeof(empty), &empty, true);

    instance_buffer_ = instance_buffer;
    renderable_ = renderable;
    render_system_->setIndirectDrawBuffer(renderable_, indirect_buffer_);
    error_.clear();
    return true;
  }

  bool uploadStaticData(
    const SplatRecord * records, std::size_t record_count,
    const float * sh_dc, std::size_t sh_dc_floats,
    const float * sh_rest, std::size_t sh_rest_floats) override
  {
    if (!core_) {
      error_ = "Metal preparation core is unavailable";
      return false;
    }
    const id<MTLCommandBuffer> last_user = last_user_;
    if (!core_->uploadStaticData(
        records, record_count, sh_dc, sh_dc_floats, sh_rest, sh_rest_floats, last_user))
    {
      error_ = core_->error();
      return false;
    }
    error_.clear();
    return true;
  }

  bool encode(
    const ViewParameters & parameters, const ProjectionParameters & projection) override
  {
    if (!core_ || !instance_buffer_ || !indirect_buffer_) {
      error_ = "Metal view preparation is not fully configured";
      return false;
    }
    auto * metal_instances =
      static_cast<Ogre::MetalHardwareVertexBuffer *>(instance_buffer_.get());
    auto * metal_indirect =
      static_cast<Ogre::MetalHardwareVertexBuffer *>(indirect_buffer_.get());
    id<MTLBuffer> instances = metal_instances->getBufferNameForGpuWrite();
    id<MTLBuffer> draw = metal_indirect->getBufferNameForGpuWrite();
    if (!instances || !draw) {
      error_ = "Ogre did not expose the Metal output buffers";
      return false;
    }
    return profiling_ ? encodeProfiled(parameters, projection, instances, draw) :
           encodeInFrame(parameters, projection, instances, draw);
  }

  std::int64_t lastVisibleCount() const override
  {
    return report_->visible.load();
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
    if (render_system_ && renderable_) {
      Ogre::HardwareVertexBufferSharedPtr none;
      render_system_->setIndirectDrawBuffer(renderable_, none);
    }
    renderable_ = nullptr;
    instance_buffer_.reset();
    indirect_buffer_.reset();
    if (core_) {
      core_->clear();
    }
    last_user_ = nil;
    report_->generation.fetch_add(1);
    report_->visible.store(-1);
    times_ = StageTimes{};
  }

  const std::string & error() const override
  {
    return error_;
  }

private:
  bool encodeInFrame(
    const ViewParameters & parameters, const ProjectionParameters & projection,
    id<MTLBuffer> instances, id<MTLBuffer> draw)
  {
    id<MTLComputeCommandEncoder> encoder = device_->getComputeEncoder();
    id<MTLCommandBuffer> command_buffer = device_->mCurrentCommandBuffer;
    if (!encoder || !command_buffer) {
      error_ = "Ogre did not provide a Metal compute encoder";
      return false;
    }
    encoder.label = @"Gaussian splat view preparation";
    // A later Ogre render encoder consumes the instance and draw buffers. Ogre
    // ends this compute encoder when it asks its device for that render
    // encoder, which keeps both in order within the same command buffer.
    if (!core_->encode(encoder, parameters, projection, instances, draw)) {
      error_ = core_->error();
      return false;
    }

    // The state buffer is written once per preparation, so a later frame
    // overwriting it can only replace one complete count with another.
    const std::shared_ptr<Report> report = report_;
    const std::uint64_t generation = report->generation.load();
    id<MTLBuffer> state = core_->stateBuffer();
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> finished) {
      if (finished.status == MTLCommandBufferStatusCompleted &&
        report->generation.load() == generation)
      {
        report->visible.store(
          static_cast<const MetalViewPreparationCore::State *>(state.contents)->visible);
      }
    }];
    last_user_ = command_buffer;
    error_.clear();
    return true;
  }

  bool encodeProfiled(
    const ViewParameters & parameters, const ProjectionParameters & projection,
    id<MTLBuffer> instances, id<MTLBuffer> draw)
  {
    id<MTLCommandQueue> queue = device_->mMainCommandQueue;
    if (!queue) {
      error_ = "Ogre did not expose its Metal command queue";
      return false;
    }
    // On Ogre's own queue, so these run after everything Ogre has committed
    // and have finished before this frame's command buffer, which is not yet
    // committed, draws from their output.
    bool ok = true;
    const auto stage = [&](NSString * label, auto && body) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        command_buffer.label = label;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        const bool encoded = body(encoder);
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        ok = ok && encoded && command_buffer.status == MTLCommandBufferStatusCompleted;
        return (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
      };

    id<MTLBuffer> sorted = nil;
    StageTimes times;
    times.cull_ms = stage(@"Gaussian splat preparation: cull and compact",
        [&](id<MTLComputeCommandEncoder> encoder) {
          return core_->encodeCull(encoder, parameters, draw);
        });
    times.sort_ms = stage(@"Gaussian splat preparation: sort",
        [&](id<MTLComputeCommandEncoder> encoder) {
          sorted = core_->encodeSort(encoder, 32u);
          return sorted != nil;
        });
    times.gather_ms = stage(@"Gaussian splat preparation: gather",
        [&](id<MTLComputeCommandEncoder> encoder) {
          if (!sorted) {
            return false;
          }
          core_->encodeGather(encoder, parameters, projection, sorted, instances);
          return true;
        });
    if (!ok) {
      error_ = core_->error().empty() ? "a GPU preparation stage failed" : core_->error();
      return false;
    }
    times_ = times;
    if (debug_projection_ && debug_frame_++ % 60 == 0) {
      dumpProjection(projection, sorted, instances);
    }
    report_->visible.store(
      static_cast<const MetalViewPreparationCore::State *>(core_->stateBuffer().contents)->visible);
    last_user_ = nil;
    error_.clear();
    return true;
  }

  // GSPLAT_DEBUG_PROJECTION: what the GPU wrote for a few sorted splats, next
  // to projectSplat() on the same records, with the parameters both used.
  void dumpProjection(
    const ProjectionParameters & q, id<MTLBuffer> sorted, id<MTLBuffer> instances)
  {
    const auto * state =
      static_cast<const MetalViewPreparationCore::State *>(core_->stateBuffer().contents);
    std::fprintf(stderr,
      "[gsplat debug] viewport %gx%g fovy %g eps2d %g antialiased %g sigma %g visible %u\n",
      q.viewport_size[0], q.viewport_size[1], q.fovy, q.eps2d, q.antialiased, q.sigma_radius,
      state->visible);
    for (int r = 0; r < 4; ++r) {
      std::fprintf(stderr, "[gsplat debug]   worldview %10.5g %10.5g %10.5g %10.5g   "
        "worldviewproj %10.5g %10.5g %10.5g %10.5g\n",
        q.worldview_rows[r * 4], q.worldview_rows[r * 4 + 1], q.worldview_rows[r * 4 + 2],
        q.worldview_rows[r * 4 + 3], q.worldviewproj_rows[r * 4],
        q.worldviewproj_rows[r * 4 + 1], q.worldviewproj_rows[r * 4 + 2],
        q.worldviewproj_rows[r * 4 + 3]);
    }
    id<MTLBuffer> splats = core_->splatBuffer();
    if (!sorted.contents || !instances.contents || !splats.contents) {
      std::fprintf(stderr, "[gsplat debug]   buffers are not CPU-visible (sorted %p instances %p "
        "splats %p)\n", sorted.contents, instances.contents, splats.contents);
      return;
    }
    const auto * pairs = static_cast<const std::uint32_t *>(sorted.contents);
    const auto * drawn = static_cast<const ProjectedInstance *>(instances.contents);
    const auto * records = static_cast<const SplatRecord *>(splats.contents);
    const std::uint32_t visible = state->visible;
    const std::uint32_t picks[3] = {0u, visible / 2u, visible > 0 ? visible - 1u : 0u};
    for (std::uint32_t k = 0; k < (visible > 0 ? 3u : 0u); ++k) {
      const std::uint32_t output = picks[k];
      const std::uint32_t index = pairs[2 * output + 1];
      const SplatRecord & record = records[index];
      const float no_colour[3] = {0.0f, 0.0f, 0.0f};
      ProjectedInstance cpu;
      projectSplat(q, record, no_colour, cpu);
      const ProjectedInstance & gpu = drawn[output];
      std::fprintf(stderr,
        "[gsplat debug]   #%u splat %u at (%g, %g, %g) scale (%g, %g, %g) opacity %g\n"
        "[gsplat debug]     gpu centre (%g, %g, %g) radius %g axes (%g, %g | %g, %g) rgba "
        "(%g, %g, %g, %g)\n"
        "[gsplat debug]     cpu centre (%g, %g, %g) radius %g axes (%g, %g | %g, %g) alpha %g\n",
        output, index, record.position[0], record.position[1], record.position[2],
        record.scale[0], record.scale[1], record.scale[2], record.opacity,
        gpu.centre[0], gpu.centre[1], gpu.centre[2], gpu.visible_radius, gpu.axes[0],
        gpu.axes[1], gpu.axes[2], gpu.axes[3], gpu.colour[0], gpu.colour[1], gpu.colour[2],
        gpu.colour[3],
        cpu.centre[0], cpu.centre[1], cpu.centre[2], cpu.visible_radius, cpu.axes[0],
        cpu.axes[1], cpu.axes[2], cpu.axes[3], cpu.colour[3]);
    }
  }

  Ogre::MetalRenderSystem * render_system_ = nullptr;
  Ogre::MetalDevice * device_ = nullptr;
  std::shared_ptr<Report> report_;
  bool profiling_ = false;
  bool debug_projection_ = false;
  std::uint64_t debug_frame_ = 0;
  std::unique_ptr<MetalViewPreparationCore> core_;
  const Ogre::Renderable * renderable_ = nullptr;
  Ogre::HardwareVertexBufferSharedPtr instance_buffer_;
  Ogre::HardwareVertexBufferSharedPtr indirect_buffer_;
  // Weak: Ogre presents each frame's drawable through that frame's command
  // buffer, so a strong reference here held the last prepared frame's drawable
  // for as long as the camera stayed still, starving the layer of drawables.
  // A buffer still on the GPU is kept alive by Metal and resolves; one that
  // has finished and been released resolves to nil, which needs no waiting.
  __weak id<MTLCommandBuffer> last_user_ = nil;
  StageTimes times_;
  std::string error_;
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
  return std::make_unique<MetalViewPreparationImpl>(
    static_cast<Ogre::MetalRenderSystem *>(render_system), shader_path);
}

}  // namespace gaussian_splatting_rviz_plugins
