// Offline acceptance check for OpenGL GPU view preparation.
//
// Runs OpenGlViewPreparationCore - the class the display drives through Ogre,
// and so the same dispatch sequence - headless through EGL, and the display's
// CPU path, prepareSplat() and projectSplat() from splat_view.hpp, on the cases
// verify_gpu_preparation.mm runs on Metal, and requires them to agree: the
// visible count, the draw arguments, every entry of the depth order including
// ties, every colour, and every projected quad within float rounding. It also
// checks what only the OpenGL path does: that instance slots past the visible
// count are empty, for a renderer that draws every instance.
//
// Where the order does not match, it says how: whether the GPU sorted its own
// keys correctly, how many ulp its keys are from the CPU's, and how far any
// splat moved. Quads and colours are compared with the splat actually drawn in
// each place, so a difference in order is not counted twice.
//
// `colcon test` runs it through CTest with --quick. The full run adds a
// Garden-sized scene and times each GPU stage:
//
//   ./verify_gl_preparation ogre_media/materials/programs/GLSL430/gsplat_prepare.comp

#include <GL/glew.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "gaussian_splatting_rviz_plugins/splat_view.hpp"
#include "opengl_view_preparation_core.hpp"
#include "preparation_test_scenes.hpp"

namespace
{
using namespace gaussian_splatting_rviz_plugins::verification;
using gaussian_splatting_rviz_plugins::DrawnSplat;
using gaussian_splatting_rviz_plugins::OpenGlViewPreparationCore;
using gaussian_splatting_rviz_plugins::ProjectedInstance;
using gaussian_splatting_rviz_plugins::ProjectionParameters;
using gaussian_splatting_rviz_plugins::ViewParameters;
using gaussian_splatting_rviz_plugins::projectSplat;

// An OpenGL 4.3 context with no surface, on the first EGL device if the driver
// offers devices and on the default display otherwise.
bool makeContext()
{
  EGLDisplay display = EGL_NO_DISPLAY;
  const auto query_devices = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
    eglGetProcAddress("eglQueryDevicesEXT"));
  const auto platform_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
    eglGetProcAddress("eglGetPlatformDisplayEXT"));
  if (query_devices && platform_display) {
    EGLDeviceEXT devices[8];
    EGLint count = 0;
    if (query_devices(8, devices, &count) && count > 0) {
      display = platform_display(EGL_PLATFORM_DEVICE_EXT, devices[0], nullptr);
    }
  }
  if (display == EGL_NO_DISPLAY) {
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  }
  if (display == EGL_NO_DISPLAY || !eglInitialize(display, nullptr, nullptr) ||
    !eglBindAPI(EGL_OPENGL_API))
  {
    return false;
  }
  const EGLint attributes[] = {
    EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3,
    EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, EGL_NONE};
  const EGLContext context =
    eglCreateContext(display, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attributes);
  if (context == EGL_NO_CONTEXT ||
    !eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context))
  {
    return false;
  }
  // GLEW loads the entry points the core uses through GLEW's macros, as Ogre's
  // GLEW does in the display. Distribution builds of GLEW target GLX: with an
  // EGL context they load the OpenGL entry points and then report that there
  // is no GLX display to look up GLX extensions on, which nothing here needs.
  glewExperimental = GL_TRUE;
  const GLenum loaded = glewInit();
#ifdef GLEW_ERROR_NO_GLX_DISPLAY
  return loaded == GLEW_OK || loaded == GLEW_ERROR_NO_GLX_DISPLAY;
#else
  return loaded == GLEW_OK;
#endif
}

template<typename T>
std::vector<T> readBuffer(GLuint buffer, std::size_t count)
{
  std::vector<T> out(count);
  if (count > 0) {
    glBindBuffer(GL_COPY_READ_BUFFER, buffer);
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, GLsizeiptr(count * sizeof(T)), out.data());
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
  }
  return out;
}

struct Context
{
  OpenGlViewPreparationCore * core = nullptr;
  GLuint instances = 0u;
};

bool load(Context & context, const Scene & scene)
{
  if (context.core->count() != scene.count) {
    if (!context.core->configure(scene.count, kIndexCount)) {
      return false;
    }
    if (context.instances) {
      glDeleteBuffers(1, &context.instances);
    }
    glGenBuffers(1, &context.instances);
    glBindBuffer(GL_ARRAY_BUFFER, context.instances);
    glBufferData(
      GL_ARRAY_BUFFER, GLsizeiptr(std::size_t(scene.count) * sizeof(ProjectedInstance)),
      nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
  return context.core->uploadStaticData(
    scene.records.data(), scene.records.size(), scene.dc.data(), scene.dc.size(),
    scene.rest.empty() ? nullptr : scene.rest.data(), scene.rest.size());
}

// One preparation as the display encodes it on stock Ogre: the core's own draw
// arguments, and the tail cleared. `incremental` makes it a frame of
// incremental preparation.
bool prepare(
  Context & context, const ViewParameters & view, const ProjectionParameters & projection,
  double * gpu_ms = nullptr, bool incremental = false, std::uint32_t colour_parts = 0)
{
  const auto start = std::chrono::steady_clock::now();
  context.core->setColourRefresh(colour_parts);
  const bool encoded = incremental ?
    context.core->encodeIncremental(view, projection, context.instances, 0u, true) :
    context.core->encode(view, projection, context.instances, 0u, true);
  glFinish();
  if (gpu_ms) {
    *gpu_ms = milliseconds(std::chrono::steady_clock::now() - start);
  }
  return encoded;
}

bool emptyInstance(const ProjectedInstance & q)
{
  return q.centre[0] == 0.0f && q.centre[1] == 0.0f && q.centre[2] == 2.0f &&
         q.visible_radius == 0.0f && q.axes[0] == 0.0f && q.axes[1] == 0.0f &&
         q.axes[2] == 0.0f && q.axes[3] == 0.0f && q.colour[0] == 0.0f && q.colour[1] == 0.0f &&
         q.colour[2] == 0.0f && q.colour[3] == 0.0f;
}

// Prepares for `prepared`, fully or as one incremental frame refreshing
// `colour_parts` of the colours, and checks the draw stream against the CPU
// path for `view`: its order and colours, projected with `projection`; or,
// given `colour_view`, the colours seen from that camera instead. The views
// differ only while an incremental build is under way.
bool verify(
  const char * name, Context & context, const Scene & scene, const ViewParameters & view,
  const ProjectionParameters & projection, const ViewParameters * prepared = nullptr,
  bool incremental = false, std::uint32_t colour_parts = 0,
  const ViewParameters * colour_view = nullptr)
{
  const Reference reference = cpuReference(scene, view);
  double gpu_ms = 0.0;
  if (!prepare(
      context, prepared ? *prepared : view, projection, &gpu_ms, incremental, colour_parts))
  {
    std::printf("%-34s FAIL: %s\n", name, context.core->error().c_str());
    return false;
  }

  const auto state =
    readBuffer<OpenGlViewPreparationCore::State>(context.core->stateBuffer(), 1)[0];
  const auto arguments =
    readBuffer<DrawIndexedArguments>(context.core->drawArgumentsBuffer(), 1)[0];
  const auto order = readBuffer<DrawnSplat>(
    context.core->drawnBuffer(), std::size_t(std::min(state.visible, scene.count)));
  const auto drawn = readBuffer<ProjectedInstance>(context.instances, scene.count);

  std::vector<std::size_t> rank(scene.count, std::numeric_limits<std::size_t>::max());
  for (std::size_t i = 0; i < reference.order.size(); ++i) {
    rank[std::uint32_t(reference.order[i])] = i;
  }

  std::size_t order_bad = 0, key_bad = 0, sort_bad = 0, quad_bad = 0, colour_bad = 0;
  std::size_t nonfinite = 0, tail_bad = 0, max_shift = 0;
  // Splats the projection culls, and those it culls or keeps within rounding.
  std::size_t culled = 0, borderline = 0;
  std::uint32_t max_key_ulp = 0;
  long first_bad = -1;
  long first_quad_bad = -1;
  std::uint32_t first_quad_splat = 0;
  double worst = 0.0;
  double worst_quad = 0.0;
  const std::size_t checked = std::min<std::size_t>(state.visible, reference.order.size());
  for (std::size_t i = 0; i < checked; ++i) {
    // The order keeps each survivor's key and splat index; ties must stay in
    // index order.
    const std::uint32_t key = order[i].key;
    const std::uint32_t splat = order[i].splat;
    if (i > 0) {
      sort_bad += !(key > order[i - 1].key ||
        (key == order[i - 1].key && splat > order[i - 1].splat));
    }
    const std::uint32_t want_key = std::uint32_t(reference.order[i] >> 32);
    const std::uint32_t want_splat = std::uint32_t(reference.order[i]);
    if (splat != want_splat) {
      ++order_bad;
      if (first_bad < 0) {
        first_bad = long(i);
      }
    }
    if (key != want_key) {
      ++key_bad;
      max_key_ulp = std::max(max_key_ulp, key > want_key ? key - want_key : want_key - key);
    }
    if (splat >= scene.count) {
      continue;
    }
    max_shift = std::max(max_shift, i > rank[splat] ? i - rank[splat] : rank[splat] - i);

    float colour[3];
    if (colour_view) {
      splatColour(scene, *colour_view, splat, colour);
    } else {
      std::memcpy(colour, &reference.colours[std::size_t(splat) * 3], sizeof(colour));
    }
    if (nearContributionThreshold(projection, scene.records[splat], colour)) {
      ++borderline;
      continue;
    }
    ProjectedInstance expected;
    projectSplat(projection, scene.records[splat], colour, expected);
    culled += expected.visible_radius == 0.0f;
    const double quad = quadDifference(drawn[i], expected, projection);
    if (!std::isfinite(quad)) {
      ++nonfinite;
    } else {
      worst_quad = std::max(worst_quad, quad);
      if (quad > 1.0) {
        ++quad_bad;
        if (first_quad_bad < 0) {
          first_quad_bad = long(i);
          first_quad_splat = splat;
        }
      }
    }
    for (int c = 0; c < 3; ++c) {
      const float got = drawn[i].colour[c];
      if (!std::isfinite(got)) {
        ++nonfinite;
        continue;
      }
      const double difference = std::fabs(double(got) - double(expected.colour[c]));
      worst = std::max(worst, difference);
      colour_bad += difference > 1e-4;
    }
  }
  for (std::size_t i = state.visible; i < scene.count; ++i) {
    tail_bad += !emptyInstance(drawn[i]);
  }

  const bool pass = state.visible == reference.order.size() &&
    arguments.instance_count == state.visible && arguments.index_count == kIndexCount &&
    sort_bad == 0 && order_bad == 0 && key_bad == 0 && quad_bad == 0 && nonfinite == 0 &&
    colour_bad == 0 && tail_bad == 0;
  std::printf(
    "%-34s N=%-8u visible cpu=%-8zu gpu=%-8u draw=%-8u order_bad=%zu key_bad=%zu "
    "colour max|d|=%.1e >1e-4:%zu quad worst/tolerance=%.2f bad=%zu nonfinite=%zu tail_bad=%zu  "
    "%.2f ms  %s\n",
    name, scene.count, reference.order.size(), state.visible, arguments.instance_count,
    order_bad, key_bad, worst, colour_bad, worst_quad, quad_bad, nonfinite, tail_bad, gpu_ms,
    pass ? "PASS" : "FAIL");
  if (projection.min_contribution > 0.0f) {
    std::printf("  below the minimum contribution: %zu, at it: %zu\n", culled, borderline);
  }
  if (order_bad || key_bad || sort_bad) {
    std::printf(
      "  sort_bad=%zu (GPU order against its own keys), max key difference %u ulp, "
      "largest move %zu places, first order mismatch at %ld\n",
      sort_bad, max_key_ulp, max_shift, first_bad);
  }
  if (first_quad_bad >= 0) {
    ProjectedInstance expected;
    projectSplat(
      projection, scene.records[first_quad_splat],
      &reference.colours[std::size_t(first_quad_splat) * 3], expected);
    std::printf("  first quad mismatch at %ld, splat %u:\n", first_quad_bad, first_quad_splat);
    printQuad("gpu", drawn[std::size_t(first_quad_bad)]);
    printQuad("cpu", expected);
  }
  std::fflush(stdout);
  return pass;
}

// Incremental preparation from an order for `from` towards one for `to`,
// projecting every frame with `projection`: four frames drawing the old order
// while the new one is built, then the frame that completes it.
int verifyIncremental(
  Context & context, const Scene & scene, const ViewParameters & from,
  const ViewParameters & to, const ProjectionParameters & projection)
{
  int failures = !verify("  incremental: starting order", context, scene, from, projection);
  const char * names[4] = {"  incremental: culling, old order",
    "  incremental: half coloured, old order", "  incremental: coloured, old order",
    "  incremental: half sorted, old order"};
  for (const char * name : names) {
    failures += !verify(name, context, scene, from, projection, &to, true);
  }
  failures += !verify("  incremental: new order", context, scene, to, projection, &to, true);

  // Colours refreshed for the camera every frame, then half of them a frame:
  // once both halves have been refreshed at the same camera, all of them are
  // its colours, on the old order and on the new.
  for (std::uint32_t parts : {1u, 2u}) {
    failures += !verify(
      "  recolour: starting order", context, scene, from, projection,
      nullptr, false, parts);
    for (int frame = 1; frame <= 4; ++frame) {
      if (frame < int(parts)) {
        prepare(context, to, projection, nullptr, true, parts);
        continue;
      }
      failures += !verify(
        parts == 1 ? "  recolour every frame, old order" :
        "  recolour half a frame, old order", context, scene, from, projection, &to, true,
        parts, &to);
    }
    failures += !verify(
      parts == 1 ? "  recolour every frame, new order" :
      "  recolour half a frame, new order", context, scene, to, projection, &to, true, parts);
  }
  return failures;
}

void timeStages(
  Context & context, const ViewParameters & view, const ProjectionParameters & projection)
{
  std::vector<double> whole, cull, shade, sort, gather;
  std::vector<double> keys, compact_count, compact_scan, compact_scatter;
  std::vector<double> histogram, radix_scan, radix_scatter;
  for (int rep = 0; rep < 2; ++rep) {
    prepare(context, view, projection);
  }
  for (int rep = 0; rep < 7; ++rep) {
    double ms = 0.0;
    prepare(context, view, projection, &ms);
    whole.push_back(ms);
  }
  context.core->setProfiling(true);
  for (int rep = 0; rep < 7; ++rep) {
    prepare(context, view, projection);
    const OpenGlViewPreparationCore::Timings & timings = context.core->lastTimings();
    cull.push_back(timings.cull_ms);
    shade.push_back(timings.shade_ms);
    sort.push_back(timings.sort_ms);
    gather.push_back(timings.gather_ms);
    keys.push_back(timings.keys_ms);
    compact_count.push_back(timings.compact_count_ms);
    compact_scan.push_back(timings.compact_scan_ms);
    compact_scatter.push_back(timings.compact_scatter_ms);
    histogram.push_back(timings.radix_histogram_ms);
    radix_scan.push_back(timings.radix_scan_ms);
    radix_scatter.push_back(timings.radix_scatter_ms);
  }
  // Incremental frames: every step of a build, in turn, and the draw, with
  // colours refreshed not at all, half a frame, and every frame.
  std::vector<double> frames[3];
  for (std::uint32_t parts : {0u, 2u, 1u}) {
    prepare(context, view, projection, nullptr, false, parts);
    for (int frame = 0; frame < 30; ++frame) {
      prepare(context, view, projection, nullptr, true, parts);
      const OpenGlViewPreparationCore::Timings & timings = context.core->lastTimings();
      frames[parts].push_back(
        timings.cull_ms + timings.shade_ms + timings.sort_ms + timings.gather_ms);
    }
  }
  context.core->setProfiling(false);
  const auto state =
    readBuffer<OpenGlViewPreparationCore::State>(context.core->stateBuffer(), 1)[0];
  std::printf("\nTiming, median of 7, %u of %u visible:\n", state.visible, view.count);
  std::printf("  encode and glFinish, as shipped     %8.2f ms\n", median(whole));
  std::printf(
    "  GPU by stage: cull+compact %.2f | shade %.2f | sort %.2f | project+gather %.2f ms\n",
    median(cull), median(shade), median(sort), median(gather));
  const char * refresh[3] = {"no colour refresh", "every colour", "half the colours"};
  for (std::uint32_t parts : {0u, 2u, 1u}) {
    std::printf(
      "  GPU per incremental frame, %s: median %.2f, largest %.2f ms\n",
      refresh[parts], median(frames[parts]),
      *std::max_element(frames[parts].begin(), frames[parts].end()));
  }
  std::printf(
    "    cull: keys %.2f | compact count %.2f | compact scan %.2f | compact scatter %.2f ms\n",
    median(keys), median(compact_count), median(compact_scan), median(compact_scatter));
  std::printf(
    "    sort, four passes: histogram %.2f | scan %.2f | scatter %.2f ms\n",
    median(histogram), median(radix_scan), median(radix_scatter));
  std::printf(
    "  preparation scratch buffers %.0f MiB\n\n",
    double(context.core->scratchBytes()) / (1024.0 * 1024.0));
}

std::string readFile(const char * path)
{
  std::ifstream stream(path);
  return stream ?
         std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()) :
         std::string();
}

}  // namespace

int main(int argc, char ** argv)
{
  bool quick = false;
  bool subgroups = true;
  const char * shader = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) {
      quick = true;
    } else if (std::strcmp(argv[i], "--no-subgroups") == 0) {
      subgroups = false;
    } else {
      shader = argv[i];
    }
  }
  if (!shader) {
    std::fprintf(
      stderr, "usage: %s [--quick] [--no-subgroups] path/to/gsplat_prepare.comp\n", argv[0]);
    return 2;
  }
  const std::string source = readFile(shader);
  if (source.empty()) {
    std::fprintf(stderr, "could not read %s\n", shader);
    return 2;
  }
  if (!makeContext()) {
    std::printf("no OpenGL 4.3 context through EGL; skipping\n");
    return kSkipped;
  }

  Context context;
  OpenGlViewPreparationCore core(source, subgroups);
  if (!core.ready()) {
    std::fprintf(stderr, "OpenGL preparation core failed: %s\n", core.error().c_str());
    return 1;
  }
  context.core = &core;
  std::printf(
    "%s | %s%s, lanes ranked %s\n\n", reinterpret_cast<const char *>(glGetString(GL_RENDERER)),
    reinterpret_cast<const char *>(glGetString(GL_VERSION)), quick ? " (quick)" : "",
    core.subgroupBallot() ? "by subgroup ballot" : "in shared memory");

  const std::vector<Case> cases = standardCases();
  int failures = 0;
  for (std::size_t n = 0; n < cases.size(); ++n) {
    const Case & c = cases[n];
    if (quick && c.count == kGardenCount) {
      continue;
    }
    Scene scene = makeScene(c.count, c.coefficients, c.layout);
    if (!load(context, scene)) {
      std::printf("%-34s FAIL: %s\n", c.name, core.error().c_str());
      ++failures;
      continue;
    }
    const ViewParameters view = makeView(scene, c.flags, c.far_clip, c.dx, c.dy, c.dz);
    // Every other case applies the antialiasing compensation.
    const ProjectionParameters projection = makeProjection(n % 2 ? 1.0f : 0.0f);
    failures += !verify(c.name, context, scene, view, projection);

    if (n == 0) {
      // The same buffers again under a narrower view: whatever the wider view
      // left beyond the new visible count must be emptied, and not drawn.
      failures += !verify(
        "  same buffers, narrower view", context, scene,
        makeView(scene, c.flags, c.far_clip, c.dx, c.dy, c.dz, 1.0f), projection);

      // Nothing visible at all.
      ViewParameters empty = view;
      empty.frustum_planes[3] = -1000.0f;
      failures += !verify("  nothing visible", context, scene, empty, projection);

      // New data of the same size, copied over the existing buffers.
      for (float & value : scene.dc) {
        value = -value;
      }
      if (!load(context, scene)) {
        std::printf("  re-upload FAIL: %s\n", core.error().c_str());
        ++failures;
      } else {
        failures += !verify("  re-uploaded in place", context, scene, view, projection);
      }

      // Another view and camera: a different sort axis and a narrower
      // frustum for the order, a different viewport for the projection.
      ProjectionParameters moved = projection;
      moved.viewport_size[0] = 1600.0f;
      moved.viewport_size[1] = 900.0f;
      moved.viewport_size[2] = 1.0f / 1600.0f;
      moved.viewport_size[3] = 1.0f / 900.0f;
      moved.fovy = 0.8f;
      // The camera moves too, so the new view's colours differ from the old.
      ViewParameters to = makeView(scene, c.flags, c.far_clip, c.dx + 0.3f, c.dy, c.dz, 2.0f);
      to.local_camera[0] = 0.6f;
      to.local_camera[1] = -0.4f;
      to.local_camera[2] = 0.3f;
      failures += verifyIncremental(context, scene, view, to, moved);

      // Splats adding less than 20 opaque pixels left out: the synthetic
      // splats are large, and this leaves out about one in fifteen.
      ProjectionParameters thinned = projection;
      thinned.min_contribution = 20.0f;
      failures += !verify("  minimum contribution 20", context, scene, view, thinned);
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
  if (!load(context, scene)) {
    std::printf("splat at the camera FAIL: %s\n", core.error().c_str());
    ++failures;
  } else {
    failures += !verify(
      "splat exactly at the camera", context, scene,
      makeView(scene, 0, 100, 0.1f, 0.2f, 1.0f), makeProjection(0.0f));
  }

  if (context.instances) {
    glDeleteBuffers(1, &context.instances);
  }
  std::printf("\n%d failing check(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
