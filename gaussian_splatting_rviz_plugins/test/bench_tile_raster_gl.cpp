// The OpenGL counterpart of bench_tile_raster.mm, headless through EGL: the
// display's quad per splat through gsplat_projected.vert and gsplat.frag,
// blended into an RGBA8 target, against the compute tile rasteriser, on the
// same prepared instances of a trained scene seen from its training cameras.
//
//   ./bench_tile_raster_gl ogre_media/materials/programs dataset/pretrained_models/garden \
//     [--camera N]... [--width W]... [--capacity PAIRS_PER_SPLAT] [--out DIR]

#include <GL/glew.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "gaussian_splatting_rviz_plugins/splat_view.hpp"
#include "opengl_view_preparation_core.hpp"
#include "preparation_test_scenes.hpp"
#include "real_scene.hpp"

namespace
{
using namespace gaussian_splatting_rviz_plugins::verification;
using gaussian_splatting_rviz_plugins::OpenGlViewPreparationCore;
using gaussian_splatting_rviz_plugins::ProjectedInstance;

// As in verify_gl_preparation.cpp.
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
  glewExperimental = GL_TRUE;
  const GLenum loaded = glewInit();
#ifdef GLEW_ERROR_NO_GLX_DISPLAY
  return loaded == GLEW_OK || loaded == GLEW_ERROR_NO_GLX_DISPLAY;
#else
  return loaded == GLEW_OK;
#endif
}

std::string readFile(const std::string & path)
{
  std::ifstream file(path);
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

GLuint shader(GLenum type, const std::string & source)
{
  const char * text = source.c_str();
  GLuint object = glCreateShader(type);
  glShaderSource(object, 1, &text, nullptr);
  glCompileShader(object);
  GLint ok = GL_FALSE;
  glGetShaderiv(object, GL_COMPILE_STATUS, &ok);
  if (ok != GL_TRUE) {
    char log[4096];
    glGetShaderInfoLog(object, sizeof(log), nullptr, log);
    std::fprintf(stderr, "shader: %s\n", log);
    return 0u;
  }
  return object;
}

GLuint texture(std::uint32_t width, std::uint32_t height, GLenum format = GL_RGBA8)
{
  GLuint name = 0u;
  glGenTextures(1, &name);
  glBindTexture(GL_TEXTURE_2D, name);
  glTexStorage2D(GL_TEXTURE_2D, 1, format, GLsizei(width), GLsizei(height));
  glBindTexture(GL_TEXTURE_2D, 0);
  return name;
}

std::vector<std::uint8_t> readTexture(GLuint name, std::uint32_t width, std::uint32_t height)
{
  std::vector<std::uint8_t> pixels(std::size_t(width) * height * 4);
  glBindTexture(GL_TEXTURE_2D, name);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
  glBindTexture(GL_TEXTURE_2D, 0);
  return pixels;
}

std::vector<std::uint16_t> readHalfTexture(GLuint name, std::uint32_t width, std::uint32_t height)
{
  std::vector<std::uint16_t> halves(std::size_t(width) * height * 4);
  glBindTexture(GL_TEXTURE_2D, name);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_HALF_FLOAT, halves.data());
  glBindTexture(GL_TEXTURE_2D, 0);
  return halves;
}

// GPU time of whatever `body` issues.
template<typename Body>
double timed(Body && body)
{
  GLuint query = 0u;
  glGenQueries(1, &query);
  glBeginQuery(GL_TIME_ELAPSED, query);
  body();
  glEndQuery(GL_TIME_ELAPSED);
  GLuint64 nanoseconds = 0;
  glGetQueryObjectui64v(query, GL_QUERY_RESULT, &nanoseconds);
  glDeleteQueries(1, &query);
  return double(nanoseconds) / 1.0e6;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s PROGRAM_DIR MODEL_DIR [--camera N]... [--width W]... "
      "[--capacity PAIRS_PER_SPLAT] [--out DIR] [--no-subgroups]\n", argv[0]);
    return 2;
  }
  const std::string programs = argv[1];
  const std::string model = argv[2];
  std::vector<int> cameras;
  std::vector<std::uint32_t> widths;
  std::vector<double> forwards;
  double capacity_per_splat = 8.0;
  bool subgroups = true;
  std::string out;
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--no-subgroups") == 0) {
      subgroups = false;
    } else if (i + 1 < argc && std::strcmp(argv[i], "--camera") == 0) {
      cameras.push_back(std::atoi(argv[++i]));
    } else if (i + 1 < argc && std::strcmp(argv[i], "--width") == 0) {
      widths.push_back(std::uint32_t(std::atoi(argv[++i])));
    } else if (i + 1 < argc && std::strcmp(argv[i], "--forward") == 0) {
      forwards.push_back(std::atof(argv[++i]));
    } else if (i + 1 < argc && std::strcmp(argv[i], "--capacity") == 0) {
      capacity_per_splat = std::atof(argv[++i]);
    } else if (i + 1 < argc && std::strcmp(argv[i], "--out") == 0) {
      out = argv[++i];
    }
  }
  if (cameras.empty()) {
    cameras = {0};
  }
  if (widths.empty()) {
    widths = {1297};
  }
  if (forwards.empty()) {
    forwards = {0.0};
  }
  if (!makeContext()) {
    std::printf("no OpenGL 4.3 context through EGL; skipping\n");
    return kSkipped;
  }

  OpenGlViewPreparationCore core(
    readFile(programs + "/GLSL430/gsplat_prepare.comp"), subgroups);
  if (!core.ready() || !core.rasterReady()) {
    std::fprintf(stderr, "setup failed: %s\n", core.error().c_str());
    return 1;
  }
  const GLuint vertex = shader(GL_VERTEX_SHADER, readFile(programs + "/GLSL/gsplat_projected.vert"));
  const GLuint fragment = shader(GL_FRAGMENT_SHADER, readFile(programs + "/GLSL/gsplat.frag"));
  if (!vertex || !fragment) {
    return 1;
  }
  const GLuint quads = glCreateProgram();
  glAttachShader(quads, vertex);
  glAttachShader(quads, fragment);
  glLinkProgram(quads);

  Scene scene;
  std::string error;
  if (!loadPly(model + "/point_cloud/iteration_30000/point_cloud.ply", scene, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::printf("%s, %s: %u splats; subgroup ballots %s\n",
    reinterpret_cast<const char *>(glGetString(GL_RENDERER)),
    reinterpret_cast<const char *>(glGetString(GL_VERSION)), scene.count,
    core.subgroupBallot() ? "on" : "off");
  if (!core.configure(scene.count, kIndexCount) ||
    !core.uploadStaticData(scene.records.data(), scene.records.size(), scene.dc.data(),
    scene.dc.size(), scene.rest.data(), scene.rest.size()))
  {
    std::fprintf(stderr, "upload failed: %s\n", core.error().c_str());
    return 1;
  }

  GLuint instances = 0u;
  glGenBuffers(1, &instances);
  glBindBuffer(GL_ARRAY_BUFFER, instances);
  glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(std::size_t(scene.count) * sizeof(ProjectedInstance)),
    nullptr, GL_DYNAMIC_COPY);
  const float corners[8] = {-1, -1, 1, -1, 1, 1, -1, 1};
  const std::uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
  GLuint vao = 0u, corner_buffer = 0u, index_buffer = 0u;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(1, &corner_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, corner_buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(corners), corners, GL_STATIC_DRAW);
  const GLint corner = glGetAttribLocation(quads, "uv2");
  glEnableVertexAttribArray(GLuint(corner));
  glVertexAttribPointer(GLuint(corner), 2, GL_FLOAT, GL_FALSE, 0, nullptr);
  glBindBuffer(GL_ARRAY_BUFFER, instances);
  const char * names[3] = {"uv0", "uv1", "colour"};
  for (int a = 0; a < 3; ++a) {
    const GLint location = glGetAttribLocation(quads, names[a]);
    glEnableVertexAttribArray(GLuint(location));
    glVertexAttribPointer(GLuint(location), 4, GL_FLOAT, GL_FALSE, sizeof(ProjectedInstance),
      reinterpret_cast<const void *>(std::uintptr_t(16 * a)));
    glVertexAttribDivisor(GLuint(location), 1);
  }
  glGenBuffers(1, &index_buffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glUseProgram(quads);
  glUniform1f(glGetUniformLocation(quads, "render_target_flipping"), 1.0f);
  glUseProgram(0);

  constexpr int kRepeats = 9;
  for (int camera_index : cameras) {
   for (double forward : forwards) {
    TrainingCamera camera;
    if (!loadCamera(model + "/cameras.json", camera_index, camera, error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    moveForward(camera, forward);
    for (std::uint32_t width : widths) {
      const CameraView view = makeCameraView(scene, camera, width, false);
      const auto capacity = std::uint32_t(std::min(
          double(scene.count) * capacity_per_splat, double(0x7fffffffu / 8u)));
      if (!core.configureRaster(view.width, view.height, capacity)) {
        std::fprintf(stderr, "raster setup failed: %s\n", core.error().c_str());
        return 1;
      }
      const GLuint quad_target = texture(view.width, view.height);
      const GLuint tile_target = texture(view.width, view.height);
      const GLuint half_target = texture(view.width, view.height, GL_RGBA16F);
      GLuint framebuffer = 0u;
      glGenFramebuffers(1, &framebuffer);
      glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, quad_target, 0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      GLuint half_framebuffer = 0u;
      glGenFramebuffers(1, &half_framebuffer);
      glBindFramebuffer(GL_FRAMEBUFFER, half_framebuffer);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, half_target, 0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      GLuint target_framebuffer = framebuffer;

      const auto draw_quads = [&]() {
          glBindFramebuffer(GL_FRAMEBUFFER, target_framebuffer);
          glViewport(0, 0, GLsizei(view.width), GLsizei(view.height));
          glClearColor(0, 0, 0, 0);
          glClear(GL_COLOR_BUFFER_BIT);
          glDisable(GL_DEPTH_TEST);
          glEnable(GL_BLEND);
          glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
          glUseProgram(quads);
          glBindVertexArray(vao);
          glBindBuffer(GL_DRAW_INDIRECT_BUFFER, core.drawArgumentsBuffer());
          glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_SHORT, nullptr);
          glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
          glBindVertexArray(0);
          glUseProgram(0);
          glDisable(GL_BLEND);
          glBindFramebuffer(GL_FRAMEBUFFER, 0);
        };

      std::vector<double> prepare, quad, tile, bin, sort, tiles;
      for (int rep = 0; rep < kRepeats; ++rep) {
        prepare.push_back(timed([&]() {
            core.encode(view.view, view.projection, instances, 0u, false);
          }));
        quad.push_back(timed(draw_quads));
        tile.push_back(timed([&]() {core.encodeRaster(instances, tile_target);}));
        GLuint sorted = 0u;
        bin.push_back(timed([&]() {core.encodeRasterBin(instances);}));
        sort.push_back(timed([&]() {sorted = core.encodeRasterSort();}));
        tiles.push_back(timed([&]() {core.encodeRasterTiles(sorted, instances, tile_target);}));
      }
      target_framebuffer = half_framebuffer;
      std::vector<double> half_quad;
      for (int rep = 0; rep < kRepeats; ++rep) {
        half_quad.push_back(timed(draw_quads));
      }
      glFinish();
      const GLenum gl_error = glGetError();
      if (gl_error != GL_NO_ERROR) {
        std::fprintf(stderr, "OpenGL error 0x%x\n", gl_error);
        return 1;
      }

      OpenGlViewPreparationCore::State state{}, raster{};
      glBindBuffer(GL_COPY_READ_BUFFER, core.stateBuffer());
      glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(state), &state);
      glBindBuffer(GL_COPY_READ_BUFFER, core.rasterStateBuffer());
      glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(raster), &raster);
      glBindBuffer(GL_COPY_READ_BUFFER, 0);
      const std::vector<std::uint8_t> quad_image = readTexture(quad_target, view.width, view.height);
      const std::vector<std::uint8_t> tile_image = readTexture(tile_target, view.width, view.height);
      const ImageDifference difference = compareImages(quad_image, tile_image);
      const std::vector<std::uint8_t> half_image =
        halfToUnorm8(readHalfTexture(half_target, view.width, view.height));
      const ImageDifference quad_error = compareImages(half_image, quad_image);
      const ImageDifference tile_error = compareImages(half_image, tile_image);
      std::printf(
        "\ncamera %d, %.1f forward, at %ux%u: %u of %u visible, %u tile pairs (%.2f per visible "
        "splat)%s\n",
        camera_index, forward, view.width, view.height, state.visible, scene.count, raster.written,
        state.visible ? double(raster.written) / state.visible : 0.0,
        raster.written > capacity ? "  OVERFLOW: raise --capacity" : "");
      std::printf("  preparation (unchanged)          %7.2f ms\n", median(prepare));
      std::printf("  quads through the render pipeline %7.2f ms\n", median(quad));
      std::printf("  quads into a half float target    %7.2f ms\n", median(half_quad));
      std::printf("  tile rasteriser                   %7.2f ms  (bin %.2f, sort %.2f, "
        "tiles %.2f)\n", median(tile), median(bin), median(sort), median(tiles));
      std::printf("  images: mean |d| %.3f, largest %d, PSNR %.1f dB, %.3f%% of pixels > 2 "
        "steps apart\n", difference.mean, difference.largest, difference.psnr,
        100.0 * difference.beyond_two);
      std::printf("  against quads blended in half float: quads PSNR %.1f dB (%.3f%% > 2 steps), "
        "tiles PSNR %.1f dB (%.3f%% > 2 steps)\n", quad_error.psnr, 100.0 * quad_error.beyond_two,
        tile_error.psnr, 100.0 * tile_error.beyond_two);
      std::fflush(stdout);
      if (!out.empty()) {
        const std::string stem = out + "/gl_camera" + std::to_string(camera_index) + "_f" +
          std::to_string(int(forward * 10)) + "_" + std::to_string(view.width);
        writePpm(stem + "_quads.ppm", quad_image, view.width, view.height, true);
        writePpm(stem + "_tiles.ppm", tile_image, view.width, view.height, true);
      }
      glDeleteFramebuffers(1, &framebuffer);
      glDeleteFramebuffers(1, &half_framebuffer);
      glDeleteTextures(1, &quad_target);
      glDeleteTextures(1, &tile_target);
      glDeleteTextures(1, &half_target);
    }
   }
  }
  return 0;
}
