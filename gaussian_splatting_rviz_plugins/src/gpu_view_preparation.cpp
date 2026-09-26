#include "gaussian_splatting_rviz_plugins/gpu_view_preparation.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>

#include <OgreRenderSystem.h>

#ifdef GSPLAT_HAS_METAL_PREPARATION
#include "gaussian_splatting_rviz_plugins/metal_view_preparation.hpp"
#endif
#ifdef GSPLAT_HAS_OPENGL_PREPARATION
#include "gaussian_splatting_rviz_plugins/opengl_view_preparation.hpp"
#endif
#ifdef GSPLAT_HAS_CUDA_PREPARATION
#include "gaussian_splatting_rviz_plugins/cuda_view_preparation.hpp"
#endif

namespace gaussian_splatting_rviz_plugins
{
namespace
{

std::string requestedBackend()
{
  const char * environment = std::getenv("GSPLAT_GPU_BACKEND");
  std::string value = environment && *environment ? environment : "auto";
  std::transform(
    value.begin(), value.end(), value.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  return value;
}

bool rendererContains(Ogre::RenderSystem * render_system, const char * token)
{
  return render_system && render_system->getName().find(token) != std::string::npos;
}

}  // namespace

std::unique_ptr<GpuViewPreparation> makeGpuViewPreparation(
  Ogre::RenderSystem * render_system, const std::string & shader_root,
  std::string & fallback_reason)
{
  fallback_reason.clear();
  const std::string requested = requestedBackend();
  if (requested == "cpu") {
    fallback_reason = "CPU preparation explicitly selected by GSPLAT_GPU_BACKEND";
    return nullptr;
  }
  if (requested != "auto" && requested != "metal" && requested != "opengl" &&
    requested != "gl" && requested != "cuda")
  {
    fallback_reason = "unknown GSPLAT_GPU_BACKEND='" + requested +
      "' (expected auto, metal, opengl, cuda, or cpu)";
    return nullptr;
  }

  const bool metal_renderer = rendererContains(render_system, "Metal");
  // RViz currently loads Ogre's legacy GL renderer. Its modern compatibility
  // context can run GL 4.3 compute while retaining the GLSL 1.20 materials.
  // Do not static_cast an eventual GL3Plus renderer to GLRenderSystem.
  const bool open_gl_renderer = rendererContains(render_system, "OpenGL Rendering Subsystem");

  // CUDA will be an optional OpenGL-buffer interop backend. It intentionally
  // has a distinct selection point here so adding CUB later does not alter the
  // display, ProjectedInstance layout, or either native compute backend.
  if (requested == "cuda") {
#ifdef GSPLAT_HAS_CUDA_PREPARATION
    return makeCudaViewPreparation(render_system, shader_root);
#else
    fallback_reason = "CUDA preparation was requested but this build has no CUDA backend";
    return nullptr;
#endif
  }

  if ((requested == "auto" && metal_renderer) || requested == "metal") {
#ifdef GSPLAT_HAS_METAL_PREPARATION
    if (!metal_renderer) {
      fallback_reason = "Metal preparation was requested for a non-Metal renderer";
      return nullptr;
    }
    return makeMetalViewPreparation(
      render_system, shader_root + "/Metal/gsplat_prepare.metal");
#else
    fallback_reason = "the Metal preparation backend was not built";
    return nullptr;
#endif
  }

  if ((requested == "auto" && open_gl_renderer) || requested == "opengl" || requested == "gl") {
#ifdef GSPLAT_HAS_CUDA_PREPARATION
    if (requested == "auto") {
      if (auto cuda = makeCudaViewPreparation(render_system, shader_root)) {
        if (cuda->error().empty()) {
          return cuda;
        }
      }
    }
#endif
#ifdef GSPLAT_HAS_OPENGL_PREPARATION
    if (!open_gl_renderer) {
      fallback_reason = "OpenGL preparation was requested for a non-OpenGL renderer";
      return nullptr;
    }
    return makeOpenGlViewPreparation(
      render_system, shader_root + "/GLSL430/gsplat_prepare.comp");
#else
    fallback_reason = "the OpenGL preparation backend was not built";
    return nullptr;
#endif
  }

  fallback_reason = "no GPU preparation backend matches renderer '" +
    (render_system ? render_system->getName() : std::string("none")) + "'";
  return nullptr;
}

}  // namespace gaussian_splatting_rviz_plugins
