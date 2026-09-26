// Offline acceptance check for GPU view preparation.
//
// Runs MetalViewPreparationCore - the class the display drives through Ogre,
// and so the same dispatch sequence - and the display's CPU path,
// prepareSplat() and projectSplat() from splat_view.hpp, on identical synthetic
// scenes and views, and requires them to agree: the visible count, the draw
// arguments, every entry of the depth order including ties, every colour, and
// every projected quad within float rounding.
//
// `colcon test` runs it through CTest with --quick. The full run adds a
// Garden-sized scene (5,834,784 splats at degree 3, about 2.7 GB) and times
// each GPU stage:
//
//   clang++ -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal \
//     -I gaussian_splatting_rviz_plugins/include -I gaussian_splatting_rviz_plugins/src \
//     gaussian_splatting_rviz_plugins/test/verify_gpu_preparation.mm \
//     gaussian_splatting_rviz_plugins/src/metal_view_preparation_core.mm \
//     -o /tmp/verify_gpu_preparation
//   /tmp/verify_gpu_preparation \
//     gaussian_splatting_rviz_plugins/ogre_media/materials/programs/Metal/gsplat_prepare.metal

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gaussian_splatting_rviz_plugins/splat_view.hpp"
#include "metal_view_preparation_core.h"
#include "preparation_test_scenes.hpp"

namespace
{
using namespace gaussian_splatting_rviz_plugins::verification;
using gaussian_splatting_rviz_plugins::MetalViewPreparationCore;
using gaussian_splatting_rviz_plugins::ProjectedInstance;
using gaussian_splatting_rviz_plugins::ProjectionParameters;
using gaussian_splatting_rviz_plugins::SplatRecord;
using gaussian_splatting_rviz_plugins::ViewParameters;
using gaussian_splatting_rviz_plugins::prepareSplat;
using gaussian_splatting_rviz_plugins::projectSplat;

struct Context
{
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  MetalViewPreparationCore * core = nullptr;
  id<MTLBuffer> instances = nil;
  id<MTLBuffer> draw = nil;
};

bool load(Context & context, const Scene & scene, id<MTLCommandBuffer> last_user)
{
  if (context.core->count() != scene.count) {
    if (!context.core->configure(scene.count, kIndexCount)) {
      return false;
    }
    context.instances = nil;
    const std::size_t instance_bytes = std::size_t(scene.count) * sizeof(ProjectedInstance);
    context.instances = [context.device newBufferWithLength:instance_bytes
                                                    options:MTLResourceStorageModeShared];
    context.draw = [context.device newBufferWithLength:sizeof(DrawIndexedArguments)
                                               options:MTLResourceStorageModeShared];
  }
  return context.instances && context.draw && context.core->uploadStaticData(
    scene.records.data(), scene.records.size(), scene.dc.data(), scene.dc.size(),
    scene.rest.empty() ? nullptr : scene.rest.data(), scene.rest.size(), last_user);
}

struct Run
{
  id<MTLCommandBuffer> command_buffer = nil;
  id<MTLBuffer> sorted = nil;
  double encode_ms = 0.0;
  double gpu_ms = 0.0;
};

Run prepare(Context & context, const ViewParameters & view, const ProjectionParameters & projection)
{
  Run run;
  run.command_buffer = [context.queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [run.command_buffer computeCommandEncoder];
  const auto start = std::chrono::steady_clock::now();
  run.sorted = context.core->encode(encoder, view, projection, context.instances, context.draw);
  run.encode_ms = milliseconds(std::chrono::steady_clock::now() - start);
  [encoder endEncoding];
  [run.command_buffer commit];
  [run.command_buffer waitUntilCompleted];
  run.gpu_ms = (run.command_buffer.GPUEndTime - run.command_buffer.GPUStartTime) * 1000.0;
  return run;
}

bool verify(
  const char * name, Context & context, const Scene & scene, const ViewParameters & view,
  const ProjectionParameters & projection, Run * finished = nullptr)
{
  const Reference reference = cpuReference(scene, view);
  Run run = prepare(context, view, projection);
  if (!run.sorted || run.command_buffer.status != MTLCommandBufferStatusCompleted) {
    std::printf("%-34s FAIL: %s\n", name, run.sorted ?
      run.command_buffer.error.localizedDescription.UTF8String : context.core->error().c_str());
    return false;
  }

  const auto * state =
    static_cast<const MetalViewPreparationCore::State *>(context.core->stateBuffer().contents);
  const auto * arguments = static_cast<const DrawIndexedArguments *>(context.draw.contents);
  const auto * pairs = static_cast<const std::uint32_t *>(run.sorted.contents);
  // The sort carries compacted slots; this maps each back to its splat.
  const auto * slot_splats =
    static_cast<const std::uint32_t *>(context.core->slotSplatBuffer().contents);
  const auto * drawn = static_cast<const ProjectedInstance *>(context.instances.contents);

  std::size_t order_bad = 0, key_bad = 0, quad_bad = 0, nonfinite = 0, colour_bad = 0;
  long first_bad = -1;
  long first_quad_bad = -1;
  double worst = 0.0;
  double worst_quad = 0.0;
  const std::size_t checked = std::min<std::size_t>(state->visible, reference.order.size());
  for (std::size_t i = 0; i < checked; ++i) {
    const std::uint32_t want_key = std::uint32_t(reference.order[i] >> 32);
    const std::uint32_t want_splat = std::uint32_t(reference.order[i]);
    const std::uint32_t slot = pairs[2 * i + 1];
    if (slot >= state->visible || slot_splats[slot] != want_splat) {
      ++order_bad;
      if (first_bad < 0) {
        first_bad = long(i);
      }
    }
    key_bad += pairs[2 * i] != want_key;
    ProjectedInstance expected;
    projectSplat(
      projection, scene.records[want_splat], &reference.colours[std::size_t(want_splat) * 3],
      expected);
    const double quad = quadDifference(drawn[i], expected, projection);
    if (!std::isfinite(quad)) {
      ++nonfinite;
    } else {
      worst_quad = std::max(worst_quad, quad);
      if (quad > 1.0) {
        ++quad_bad;
        if (first_quad_bad < 0) {
          first_quad_bad = long(i);
        }
      }
    }
    for (int c = 0; c < 3; ++c) {
      const float got = drawn[i].colour[c];
      if (!std::isfinite(got)) {
        ++nonfinite;
        continue;
      }
      // Against the projected reference, which also zeroes the colour of a
      // splat the projection rejects.
      const double difference = std::fabs(double(got) - double(expected.colour[c]));
      worst = std::max(worst, difference);
      colour_bad += difference > 1e-4;
    }
  }

  const bool pass = state->visible == reference.order.size() &&
    arguments->instance_count == state->visible && arguments->index_count == kIndexCount &&
    order_bad == 0 && key_bad == 0 && quad_bad == 0 && nonfinite == 0 && colour_bad == 0;
  std::printf(
    "%-34s N=%-8u visible cpu=%-8zu gpu=%-8u draw=%-8u order_bad=%zu key_bad=%zu "
    "colour max|d|=%.1e >1e-4:%zu quad worst/tolerance=%.2f bad=%zu nonfinite=%zu  gpu %.2f ms  %s",
    name, scene.count, reference.order.size(), state->visible, arguments->instance_count,
    order_bad, key_bad, worst, colour_bad, worst_quad, quad_bad, nonfinite, run.gpu_ms,
    pass ? "PASS" : "FAIL");
  if (first_bad >= 0) {
    std::printf("  first order mismatch at %ld", first_bad);
  }
  std::printf("\n");
  if (first_quad_bad >= 0) {
    const std::uint32_t splat = std::uint32_t(reference.order[std::size_t(first_quad_bad)]);
    ProjectedInstance expected;
    projectSplat(
      projection, scene.records[splat], &reference.colours[std::size_t(splat) * 3], expected);
    std::printf("  first quad mismatch at %ld, splat %u:\n", first_quad_bad, splat);
    printQuad("gpu", drawn[first_quad_bad]);
    printQuad("cpu", expected);
  }
  std::fflush(stdout);
  if (finished) {
    *finished = run;
  }
  return pass;
}

void timeStages(
  Context & context, const ViewParameters & view, const ProjectionParameters & projection)
{
  std::vector<double> encode, whole, cull, shade, sort, gather;
  for (int rep = 0; rep < 7; ++rep) {
    const Run run = prepare(context, view, projection);
    encode.push_back(run.encode_ms);
    whole.push_back(run.gpu_ms);
  }
  const auto stage = [&](auto && body) {
      id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      body(encoder);
      [encoder endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      return (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
    };
  for (int rep = 0; rep < 7; ++rep) {
    id<MTLBuffer> sorted = nil;
    cull.push_back(stage([&](id<MTLComputeCommandEncoder> encoder) {
        context.core->encodeCull(encoder, view, context.draw);
      }));
    shade.push_back(stage([&](id<MTLComputeCommandEncoder> encoder) {
        context.core->encodeShade(encoder, view, projection);
      }));
    sort.push_back(stage([&](id<MTLComputeCommandEncoder> encoder) {
        sorted = context.core->encodeSort(encoder);
      }));
    gather.push_back(stage([&](id<MTLComputeCommandEncoder> encoder) {
        context.core->encodeGather(encoder, sorted, context.instances);
      }));
  }
  const auto * state =
    static_cast<const MetalViewPreparationCore::State *>(context.core->stateBuffer().contents);
  std::printf("\nTiming, median of 7, %u of %u visible:\n", state->visible, view.count);
  std::printf("  CPU encode of one preparation      %8.3f ms\n", median(encode));
  std::printf("  GPU, one command buffer as shipped %8.2f ms\n", median(whole));
  std::printf(
    "  GPU by stage: cull+compact %.2f | shade+project %.2f | sort %.2f | gather %.2f ms\n",
    median(cull), median(shade), median(sort), median(gather));
  std::printf("  preparation scratch buffers %.0f MiB\n\n",
    double(context.core->scratchBytes()) / (1024.0 * 1024.0));
}

}  // namespace

int main(int argc, char ** argv)
{
  @autoreleasepool {
    bool quick = false;
    const char * shader = nullptr;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--quick") == 0) {
        quick = true;
      } else {
        shader = argv[i];
      }
    }
    if (!shader) {
      std::fprintf(stderr, "usage: %s [--quick] path/to/gsplat_prepare.metal\n", argv[0]);
      return 2;
    }

    Context context;
    context.device = MTLCreateSystemDefaultDevice();
    if (!context.device) {
      std::printf("no Metal device; skipping\n");
      return kSkipped;
    }
    context.queue = [context.device newCommandQueue];
    MetalViewPreparationCore core(context.device, shader);
    if (!core.ready()) {
      std::fprintf(stderr, "Metal preparation core failed: %s\n", core.error().c_str());
      return 1;
    }
    context.core = &core;
    std::printf("%s%s\n\n", context.device.name.UTF8String, quick ? " (quick)" : "");

    const std::vector<Case> cases = standardCases();
    int failures = 0;
    for (std::size_t n = 0; n < cases.size(); ++n) {
      const Case & c = cases[n];
      if (quick && c.count == kGardenCount) {
        continue;
      }
      Scene scene = makeScene(c.count, c.coefficients, c.layout);
      if (!load(context, scene, nil)) {
        std::printf("%-34s FAIL: %s\n", c.name, core.error().c_str());
        ++failures;
        continue;
      }
      const ViewParameters view = makeView(scene, c.flags, c.far_clip, c.dx, c.dy, c.dz);
      // Every other case applies the antialiasing compensation.
      const ProjectionParameters projection = makeProjection(n % 2 ? 1.0f : 0.0f);
      Run run;
      failures += !verify(c.name, context, scene, view, projection, &run);

      if (n == 0) {
        // The same buffers again under a narrower view: whatever the wider view
        // left beyond the new visible count must not be drawn.
        failures += !verify("  same buffers, narrower view", context, scene,
          makeView(scene, c.flags, c.far_clip, c.dx, c.dy, c.dz, 1.0f), projection);

        // Nothing visible at all.
        ViewParameters empty = view;
        empty.frustum_planes[3] = -1000.0f;
        failures += !verify("  nothing visible", context, scene, empty, projection);

        // New data of the same size, copied over the existing buffers.
        for (float & value : scene.dc) {
          value = -value;
        }
        if (!load(context, scene, run.command_buffer)) {
          std::printf("  re-upload FAIL: %s\n", core.error().c_str());
          ++failures;
        } else {
          failures += !verify("  re-uploaded in place", context, scene, view, projection);
        }
      }
      if (!quick && c.count == kGardenCount) {
        timeStages(context, view, projection);
      }
    }

    // A splat exactly at the camera: without the normalise guard its direction,
    // and so its colour, would be NaN on one side and not the other.
    Scene scene = makeScene(4096, 15, Layout::kRandom);
    scene.records[17].position[0] = 0.0f;
    scene.records[17].position[1] = 0.0f;
    scene.records[17].position[2] = 0.0f;
    scene.records[17].opacity = 0.9f;
    if (!load(context, scene, nil)) {
      std::printf("splat at the camera FAIL: %s\n", core.error().c_str());
      ++failures;
    } else {
      failures += !verify("splat exactly at the camera", context, scene,
        makeView(scene, 0, 100, 0.1f, 0.2f, 1.0f), makeProjection(0.0f));
    }

    std::printf("\n%d failing check(s)\n", failures);
    return failures == 0 ? 0 : 1;
  }
}
