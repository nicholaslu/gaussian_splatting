// Scenes, views and the CPU reference shared by the offline verifiers of GPU
// view preparation: verify_gpu_preparation.mm drives the Metal core and
// verify_gl_preparation.cpp the OpenGL one, on the same cases, against the
// display's CPU path in splat_view.hpp.

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gaussian_splatting_rviz_plugins/splat_view.hpp"

namespace gaussian_splatting_rviz_plugins
{
namespace verification
{

constexpr std::uint32_t kIndexCount = 6;
constexpr int kSkipped = 77;  // CTest's skip code, for a machine with no GPU to test
constexpr std::uint32_t kGardenCount = 5834784;

struct DrawIndexedArguments
{
  std::uint32_t index_count;
  std::uint32_t instance_count;
  std::uint32_t index_start;
  std::int32_t base_vertex;
  std::uint32_t base_instance;
};

inline double milliseconds(std::chrono::steady_clock::duration duration)
{
  return std::chrono::duration<double, std::milli>(duration).count();
}

inline double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

inline std::uint64_t random_state = 0x9E3779B97F4A7C15ull;

inline std::uint64_t next64()
{
  std::uint64_t x = random_state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return random_state = x;
}

inline float uniform(float low, float high)
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
inline Scene makeScene(std::uint32_t count, std::uint32_t coefficients, Layout layout)
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
inline ViewParameters makeView(
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
// far 100, with a [0, 1] depth range as getProjectionMatrixWithRSDepth() gives
// it on Metal. The GPU is checked against projectSplat() with the same matrix,
// so the range does not matter to the checks. The node and view transforms are identity, so worldview is too.
inline ProjectionParameters makeProjection(float antialiased)
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
inline Reference cpuReference(const Scene & scene, const ViewParameters & view)
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

// How far two projected quads disagree, in units of what float rounding can
// explain: above 1 is a real difference. Centres are measured in pixels, and
// the ellipses through a1 a1^T + a2 a2^T in square pixels, which does not care
// which way either axis points. Opacity carries the antialiasing compensation.
inline double scaledError(double got, double want, double absolute, double relative)
{
  return std::fabs(got - want) /
         (absolute + relative * std::max(std::fabs(got), std::fabs(want)));
}

inline double quadDifference(
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

inline void printQuad(const char * label, const ProjectedInstance & q)
{
  std::printf(
    "    %s centre (%.7g, %.7g, %.7g) radius %.7g axes (%.7g, %.7g | %.7g, %.7g) rgba (%.5g, %.5g, "
    "%.5g, %.5g)\n", label, q.centre[0], q.centre[1], q.centre[2], q.visible_radius, q.axes[0],
    q.axes[1], q.axes[2], q.axes[3], q.colour[0], q.colour[1], q.colour[2], q.colour[3]);
}

struct Case
{
  const char * name;
  std::uint32_t count, coefficients, flags;
  Layout layout;
  float far_clip, dx, dy, dz;
};

inline std::vector<Case> standardCases()
{
  const std::uint32_t both = kViewCulling | kViewPerspective;
  return {
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
}

}  // namespace verification
}  // namespace gaussian_splatting_rviz_plugins
