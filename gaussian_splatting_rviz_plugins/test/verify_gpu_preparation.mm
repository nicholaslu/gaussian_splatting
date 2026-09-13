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

namespace
{
using gaussian_splatting_rviz_plugins::MetalViewPreparationCore;
using gaussian_splatting_rviz_plugins::ProjectedInstance;
using gaussian_splatting_rviz_plugins::ProjectionParameters;
using gaussian_splatting_rviz_plugins::SplatRecord;
using gaussian_splatting_rviz_plugins::ViewParameters;
using gaussian_splatting_rviz_plugins::kViewCulling;
using gaussian_splatting_rviz_plugins::kViewPerspective;
using gaussian_splatting_rviz_plugins::prepareSplat;
using gaussian_splatting_rviz_plugins::projectSplat;

constexpr std::uint32_t kIndexCount = 6;
constexpr int kSkipped = 77;  // CTest's skip code, for a machine with no Metal device
constexpr std::uint32_t kGardenCount = 5834784;

struct DrawIndexedArguments
{
  std::uint32_t index_count;
  std::uint32_t instance_count;
  std::uint32_t index_start;
  std::int32_t base_vertex;
  std::uint32_t base_instance;
};

double milliseconds(std::chrono::steady_clock::duration duration)
{
  return std::chrono::duration<double, std::milli>(duration).count();
}

double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

std::uint64_t random_state = 0x9E3779B97F4A7C15ull;

std::uint64_t next64()
{
  std::uint64_t x = random_state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return random_state = x;
}

float uniform(float low, float high)
{
  return low + (high - low) * static_cast<float>((next64() >> 40) & 0xFFFFFFu) / float(0xFFFFFF);
}

enum class Layout { kRandom, kGrid, kFarField };

struct Scene
{
  std::uint32_t count = 0;
  std::uint32_t coefficients = 0;
  std::vector<SplatRecord> records;
  std::vector<float> dc;
  std::vector<float> rest;
};

// kGrid puts every splat on one of 12 depths along the axis the views sort on,
// so almost every key ties and the order is decided by stability alone.
// kFarField reaches past the far plane at z = -100.
Scene makeScene(std::uint32_t count, std::uint32_t coefficients, Layout layout)
{
  Scene scene;
  scene.count = count;
  scene.coefficients = coefficients;
  scene.records.resize(count);
  scene.dc.resize(std::size_t(count) * 3);
  const std::size_t stride = std::size_t(coefficients) * 3;
  scene.rest.resize(std::size_t(count) * stride);
  for (std::uint32_t i = 0; i < count; ++i) {
    float x, y, z;
    if (layout == Layout::kGrid) {
      x = float(int(next64() % 9) - 4);
      y = float(int(next64() % 9) - 4);
      z = -float(1 + next64() % 12);
    } else if (layout == Layout::kFarField) {
      x = uniform(-5, 5);
      y = uniform(-5, 5);
      z = uniform(-300, 1);
    } else {
      x = uniform(-5, 5);
      y = uniform(-5, 5);
      z = uniform(-12, 1);
    }
    // One splat in ten sits below the alpha cutoff.
    const float opacity = next64() % 10 == 0 ? uniform(0.0f, 0.0039f) : uniform(0.004f, 1.0f);
    scene.records[i] = SplatRecord{{x, y, z}, opacity,
      {uniform(0.001f, 0.2f), uniform(0.001f, 0.2f), uniform(0.001f, 0.2f)}, 0.0f, {0, 0, 0, 1}};
    for (int c = 0; c < 3; ++c) {
      scene.dc[std::size_t(i) * 3 + c] = uniform(-1.5f, 1.5f);
    }
    for (std::size_t k = 0; k < stride; ++k) {
      scene.rest[i * stride + k] = uniform(-0.5f, 0.5f);
    }
  }
  return scene;
}

// A camera at the origin looking down -z, with a frustum `half_width` either
// side, near at 0.1 and far at 100. A far_clip of 0 declares the frustum
// infinite, so the far plane is skipped although it is present.
ViewParameters makeView(
  const Scene & scene, std::uint32_t flags, float far_clip,
  float dx, float dy, float dz, float half_width = 3.0f)
{
  ViewParameters view{};
  for (int r = 0; r < 4; ++r) {
    view.world_rows[r * 5] = 1.0f;
    view.view_rows[r * 5] = 1.0f;
  }
  // Ogre's FrustumPlane order - near, far, left, right, top, bottom - as (normal, d).
  const float planes[24] = {
    0, 0, -1, -0.1f, 0, 0, 1, 100.0f,
    1, 0, 0, half_width, -1, 0, 0, half_width,
    0, -1, 0, half_width, 0, 1, 0, half_width};
  std::memcpy(view.frustum_planes, planes, sizeof(planes));
  const float length = std::sqrt(dx * dx + dy * dy + dz * dz);
  view.local_sort_direction[0] = dx / length;
  view.local_sort_direction[1] = dy / length;
  view.local_sort_direction[2] = dz / length;
  view.world_scale = 1.0f;
  view.focal_y = 800.0f;
  view.near_clip = 0.1f;
  view.far_clip = far_clip;
  view.ortho_scale = 400.0f;
  view.sigma_radius = 3.0f;
  view.min_screen_radius = 1.5f;
  view.eps2d = 0.3f;
  view.count = scene.count;
  view.viewport_height = 1600;
  view.sh_coefficients = scene.coefficients;
  view.flags = flags;
  return view;
}

// A 1280x720 perspective camera at the origin looking down -z, near 0.1 and
// far 100, with Metal's [0, 1] depth range, as getProjectionMatrixWithRSDepth()
// gives it. The node and view transforms are identity, so worldview is too.
ProjectionParameters makeProjection(float antialiased)
{
  ProjectionParameters projection{};
  const float width = 1280.0f;
  const float height = 720.0f;
  const float fovy = 1.0f;
  const float near_clip = 0.1f;
  const float far_clip = 100.0f;
  const float focal = 1.0f / std::tan(fovy * 0.5f);
  const float rows[16] = {
    focal * height / width, 0, 0, 0,
    0, focal, 0, 0,
    0, 0, far_clip / (near_clip - far_clip), far_clip * near_clip / (near_clip - far_clip),
    0, 0, -1, 0};
  for (int r = 0; r < 4; ++r) {
    projection.worldview_rows[r * 5] = 1.0f;
  }
  std::memcpy(projection.worldviewproj_rows, rows, sizeof(rows));
  projection.viewport_size[0] = width;
  projection.viewport_size[1] = height;
  projection.viewport_size[2] = 1.0f / width;
  projection.viewport_size[3] = 1.0f / height;
  projection.fovy = fovy;
  projection.eps2d = 0.3f;
  projection.antialiased = antialiased;
  projection.sigma_radius = 3.0f;
  return projection;
}

struct Reference
{
  std::vector<std::uint64_t> order;
  std::vector<float> colours;
};

// The display's CPU path: prepareSplat() per splat, then the packed
// (key << 32 | index) words sorted. Every word is unique, so any correct sort
// of them gives the order the display's stable radix produces.
Reference cpuReference(const Scene & scene, const ViewParameters & view)
{
  Reference reference;
  reference.colours.assign(std::size_t(scene.count) * 3, 0.0f);
  const std::size_t stride = std::size_t(scene.coefficients) * 3;
  for (std::uint32_t i = 0; i < scene.count; ++i) {
    std::uint32_t key = 0;
    if (prepareSplat(
        view, scene.records[i], &scene.dc[std::size_t(i) * 3],
        stride > 0 ? &scene.rest[i * stride] : nullptr,
        &reference.colours[std::size_t(i) * 3], key))
    {
      reference.order.push_back((std::uint64_t(key) << 32) | i);
    }
  }
  std::sort(reference.order.begin(), reference.order.end());
  return reference;
}

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

// How far two projected quads disagree, in units of what float rounding can
// explain: above 1 is a real difference. Centres are measured in pixels, and
// the ellipses through a1 a1^T + a2 a2^T in square pixels, which does not care
// which way either axis points. Opacity carries the antialiasing compensation.
double scaledError(double got, double want, double absolute, double relative)
{
  return std::fabs(got - want) /
         (absolute + relative * std::max(std::fabs(got), std::fabs(want)));
}

double quadDifference(
  const ProjectedInstance & got, const ProjectedInstance & want,
  const ProjectionParameters & projection)
{
  const double half_width = 0.5 * projection.viewport_size[0];
  const double half_height = 0.5 * projection.viewport_size[1];
  const auto form = [half_width, half_height](const ProjectedInstance & q) {
      const double x1 = q.axes[0] * half_width, y1 = q.axes[1] * half_height;
      const double x2 = q.axes[2] * half_width, y2 = q.axes[3] * half_height;
      return std::array<double, 3>{x1 * x1 + x2 * x2, x1 * y1 + x2 * y2, y1 * y1 + y2 * y2};
    };
  const auto g = form(got);
  const auto w = form(want);
  const double trace = std::max(g[0] + g[2], w[0] + w[2]);
  double worst = 0.0;
  worst = std::max(worst, scaledError(
      got.centre[0] * half_width, want.centre[0] * half_width, 1e-2, 1e-5));
  worst = std::max(worst, scaledError(
      got.centre[1] * half_height, want.centre[1] * half_height, 1e-2, 1e-5));
  worst = std::max(worst, scaledError(got.centre[2], want.centre[2], 1e-5, 1e-5));
  for (int i = 0; i < 3; ++i) {
    worst = std::max(worst, std::fabs(g[i] - w[i]) / (1e-2 + 1e-4 * trace));
  }
  // The compensation divides ac - b^2 by its blurred counterpart, both small
  // differences of large numbers for a thin ellipse, so opacity - and the
  // visible radius taken from it - round further apart than the rest: a few
  // parts in 10^4, still some forty times below one 8-bit alpha step.
  worst = std::max(worst, scaledError(got.visible_radius, want.visible_radius, 1e-5, 1e-4));
  worst = std::max(worst, scaledError(got.colour[3], want.colour[3], 1e-5, 1e-3));
  return worst;
}

void printQuad(const char * label, const ProjectedInstance & q)
{
  std::printf(
    "    %s centre (%.7g, %.7g, %.7g) radius %.7g axes (%.7g, %.7g | %.7g, %.7g) rgba (%.5g, %.5g, "
    "%.5g, %.5g)\n", label, q.centre[0], q.centre[1], q.centre[2], q.visible_radius, q.axes[0],
    q.axes[1], q.axes[2], q.axes[3], q.colour[0], q.colour[1], q.colour[2], q.colour[3]);
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
    if (pairs[2 * i + 1] != want_splat) {
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
  std::vector<double> encode, whole, cull, sort, gather;
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
    sort.push_back(stage([&](id<MTLComputeCommandEncoder> encoder) {
        sorted = context.core->encodeSort(encoder);
      }));
    gather.push_back(stage([&](id<MTLComputeCommandEncoder> encoder) {
        context.core->encodeGather(encoder, view, projection, sorted, context.instances);
      }));
  }
  const auto * state =
    static_cast<const MetalViewPreparationCore::State *>(context.core->stateBuffer().contents);
  std::printf("\nTiming, median of 7, %u of %u visible:\n", state->visible, view.count);
  std::printf("  CPU encode of one preparation      %8.3f ms\n", median(encode));
  std::printf("  GPU, one command buffer as shipped %8.2f ms\n", median(whole));
  std::printf("  GPU by stage: cull+compact %.2f | sort %.2f | gather+SH+projection %.2f ms\n",
    median(cull), median(sort), median(gather));
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

    const std::uint32_t both = kViewCulling | kViewPerspective;
    struct Case
    {
      const char * name;
      std::uint32_t count, coefficients, flags;
      Layout layout;
      float far_clip, dx, dy, dz;
    };
    const Case cases[] = {
      {"random, degree 3", 200000, 15, both, Layout::kRandom, 100, 0.1f, 0.2f, 1.0f},
      {"ties on 12 depths, degree 3", 300000, 15, both, Layout::kGrid, 100, 0, 0, 1},
      {"culling off, degree 1", 150000, 3, 0, Layout::kRandom, 100, 0.1f, 0.2f, 1.0f},
      {"orthographic, degree 2", 150000, 8, kViewCulling, Layout::kRandom, 100, 0.1f, 0.2f, 1.0f},
      {"infinite far plane, degree 0", 150000, 0, both, Layout::kFarField, 0, 0.1f, 0.2f, 1.0f},
      {"N=1", 1, 15, 0, Layout::kRandom, 100, 0.1f, 0.2f, 1.0f},
      {"N=31", 31, 15, both, Layout::kRandom, 100, 0.1f, 0.2f, 1.0f},
      {"N=1023", 1023, 15, both, Layout::kRandom, 100, 0.1f, 0.2f, 1.0f},
      {"N=1025", 1025, 15, both, Layout::kRandom, 100, 0.1f, 0.2f, 1.0f},
      {"1,000,003 with ties, degree 3", 1000003, 15, both, Layout::kGrid, 100, 0, 0, 1},
      {"garden-sized, degree 3", kGardenCount, 15, both, Layout::kRandom, 100, 0.1f, 0.2f, 1.0f},
    };

    int failures = 0;
    for (const Case & c : cases) {
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
      const ProjectionParameters projection = makeProjection((&c - cases) % 2 ? 1.0f : 0.0f);
      Run run;
      failures += !verify(c.name, context, scene, view, projection, &run);

      if (&c == &cases[0]) {
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
