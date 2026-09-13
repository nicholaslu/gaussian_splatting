#include "gaussian_splatting_rviz_plugins/gaussian_splatting_display.hpp"
#ifdef GSPLAT_HAS_METAL_PREPARATION
#include "gaussian_splatting_rviz_plugins/metal_view_preparation.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <thread>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <OgreAxisAlignedBox.h>
#include <OgreCamera.h>
#include <OgreGpuProgramParams.h>
#include <OgreHardwarePixelBuffer.h>
#include <OgreHardwareBufferManager.h>
#include <OgreMaterialManager.h>
#include <OgrePass.h>
#include <OgreRectangle2D.h>
#include <OgreRenderSystem.h>
#include <OgreRenderTexture.h>
#include <OgreResourceGroupManager.h>
#include <OgreRoot.h>
#include <OgreSceneManager.h>
#include <OgreSceneNode.h>
#include <OgreSimpleRenderable.h>
#include <OgreSphere.h>
#include <OgreTechnique.h>
#include <OgreTextureManager.h>
#include <OgreVertexIndexData.h>
#include <OgreViewport.h>

#include "ament_index_cpp/get_package_share_directory.hpp"
#ifdef RVIZ_STATIC_PLUGINS_ONLY
// An application that links this plug-in registers it by calling
// registerStaticPlugins(); there is no class loader to export to, and the
// registration object would only be dropped by the linker anyway.
#define PLUGINLIB_EXPORT_CLASS(class_type, base_class_type)
#else
#include "pluginlib/class_list_macros.hpp"
#endif
#include "rclcpp/logging.hpp"
#include "rviz_common/display_context.hpp"
#include "rviz_common/frame_manager_iface.hpp"
#include "rviz_common/view_controller.hpp"
#include "rviz_common/render_panel.hpp"
#include "rviz_common/properties/bool_property.hpp"
#include "rviz_common/view_manager.hpp"
#include "rviz_rendering/render_window.hpp"

namespace gaussian_splatting_rviz_plugins
{
namespace
{

constexpr const char * kResourceGroup = "GaussianSplattingRviz";
constexpr const char * kMaterialName = "GaussianSplatting/RViz";
constexpr const char * kCompositeMaterialName = "GaussianSplatting/Composite";
constexpr const char * kDepthMaterialName = "GaussianSplatting/DepthOnly";
constexpr const char * kDepthMaterialScheme = "GaussianDepthOnly";

// Splats get a render queue of their own so the offscreen pass can be told to
// draw that queue and nothing else. Sitting just after RENDER_QUEUE_MAIN keeps
// them on top of the opaque scene, which is where an alpha blended cloud
// belongs anyway.
constexpr Ogre::uint8 kSplatRenderQueue = Ogre::RENDER_QUEUE_MAIN + 1;

// Spherical harmonics basis constants, as used by the 3DGS reference
// implementation. Degree 0 alone gives rgb = sh_dc * kSHC0 + 0.5; the rest add
// the view dependence that a specular surface is almost entirely made of.
constexpr float kSHC0 = 0.28209479177387814f;
constexpr std::uint8_t kMaxShDegree = 3;
constexpr float kSHC1 = 0.4886025119029199f;
constexpr float kSHC2[5] = {
  1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
  -1.0925484305920792f, 0.5462742152960396f,
};
constexpr float kSHC3[7] = {
  -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
  0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
  -0.5900435899266435f,
};

// Each splat is expanded from one instance record and a shared quad. RViz's
// patched Ogre Metal render system maps instance buffers to
// MTLVertexStepFunctionPerInstance; OpenGL uses the same Ogre declaration.
//
// This was an octagon for a while. Both shapes circumscribe the circle at
// which the fragment shader's alpha cutoff bites, so neither clips anything
// the shader would keep and the two render identically; the octagon only
// removes 17.2% of the square's dead corner area, which is a fill-rate trade.
// Measurement says that trade is the wrong way round here. Frame time against
// visible splats, on a 5.8 million splat scene:
//
//   octagon   13.8 ms + 31.4 ms per million
//   quad      16.0 ms + 20.3 ms per million
//
// 1.55x cheaper per splat, and Render Scale does not move either of them - the
// corners the octagon saves are free, while its 8 vertices and 6 triangles per
// splat against the quad's 4 and 2 are not, since the whole per-splat
// projection below runs once per vertex.
//
// The per-splat record is read from a texture rather than streamed per
// instance, so a depth sort rewrites one index and one colour per splat rather
// than the whole record. Three RGBA32F texels hold it, and the width is a
// whole number of splats so that a splat never straddles a row.
//
// The width is chosen per scene rather than fixed. A fixed 768 holds 256
// splats per row, so the texture grows in height alone: 5.8 million splats ask
// for 22,793 rows, which Metal refuses on a Mac and accepts on an M5 iPad -
// Apple GPU family 10 allows a 32,768 side where families 7 to 9 allow 16,384,
// and the failure is an aborted process, not a returned error. A square-ish
// layout grows as the square root instead, so the same scene is 4,185 x 4,183
// and no limit is anywhere near. Small scenes keep the old width, which is
// already square enough for them.
constexpr std::size_t kTexelsPerSplat = 3;
constexpr unsigned int kMinSplatDataWidth = 768;
static_assert(
  kMinSplatDataWidth % kTexelsPerSplat == 0, "a texture row must hold whole splats");

// The smallest 2D texture side among the render systems this runs on: Metal on
// Apple GPU families 7 to 9, and OpenGL 3.3, both 16,384. Ogre 1.12 does not
// report the device's own limit, so this is a constant rather than a query -
// affordable because the square layout above only reaches it at 89 million
// splats, which is 4 GB of texture and unreachable for other reasons first.
constexpr unsigned int kMaxTextureDimension = 16384;

// GLSL 120 has no integer texel fetch, so the OpenGL path addresses the data
// texture in floats and is exact only below 2^24 texels.
constexpr std::size_t kMaxGlslExactTexels = 1u << 24;

// The smallest whole number of splats' worth of texels that is at least wanted.
unsigned int roundUpToWholeSplats(std::size_t wanted)
{
  const std::size_t rounded =
    (wanted + kTexelsPerSplat - 1) / kTexelsPerSplat * kTexelsPerSplat;
  return static_cast<unsigned int>(rounded);
}

unsigned int splatDataWidth(std::size_t count)
{
  const auto texels = static_cast<double>(count * kTexelsPerSplat);
  const auto square = static_cast<std::size_t>(std::ceil(std::sqrt(texels)));
  return std::max(kMinSplatDataWidth, roundUpToWholeSplats(square));
}

unsigned int splatDataRows(std::size_t count)
{
  const std::size_t texels = count * kTexelsPerSplat;
  const std::size_t width = splatDataWidth(count);
  return static_cast<unsigned int>((texels + width - 1) / width);
}

constexpr std::size_t kVerticesPerSplat = 4;
constexpr std::size_t kIndicesPerSplat = 6;
constexpr float kCorners[kVerticesPerSplat][2] = {
  {-1.0f, -1.0f},
  {1.0f, -1.0f},
  {1.0f, 1.0f},
  {-1.0f, 1.0f},
};

bool usesMetalRenderSystem()
{
  Ogre::RenderSystem * render_system = Ogre::Root::getSingleton().getRenderSystem();
  return render_system && render_system->getName().find("Metal") != std::string::npos;
}

template<typename T>
T clamp(T value, T low, T high)
{
  return std::max(low, std::min(value, high));
}

// Splits [0, count) across the calling thread and a few others, calling
// body(begin, end, worker). Threads are created per call rather than pooled:
// at tens of microseconds each that is a fraction of a percent of the work
// being split, and it keeps the display free of a pool's lifetime and
// shutdown handling. Below the threshold the split costs more than it saves,
// so the body runs inline as worker 0.
constexpr std::size_t kParallelThreshold = 32768;

std::size_t workerCount(std::size_t items)
{
  if (items < kParallelThreshold) {
    return 1;
  }
  const unsigned hardware = std::thread::hardware_concurrency();
  return std::max<std::size_t>(1, std::min<std::size_t>(hardware, 8));
}

template<typename Body>
void parallelFor(std::size_t count, std::size_t workers, Body && body)
{
  if (workers <= 1 || count == 0) {
    body(0, count, 0);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  for (std::size_t worker = 1; worker < workers; ++worker) {
    pool.emplace_back(
      [&body, count, workers, worker] {
        body(count * worker / workers, count * (worker + 1) / workers, worker);
      });
  }
  body(0, count / workers, 0);
  for (std::thread & thread : pool) {
    thread.join();
  }
}

}  // namespace

// Resolves the depth viewport's material scheme in three tiers:
//
// 1. A material-provided GaussianDepthOnly technique is selected by Ogre
//    before this listener is called.
// 2. Simple opaque rigid materials use a shared minimal depth-only technique.
// 3. Everything else gets a cached clone of its original technique with colour
//    writes disabled. This preserves custom vertex deformation, alpha reject,
//    and transparent/depth-write semantics for plugins unknown to us.
class DepthSchemeResolver : public Ogre::MaterialManager::Listener
{
public:
  struct Stats
  {
    std::size_t fast = 0;
    std::size_t conservative = 0;
    std::size_t unresolved = 0;
  };

  explicit DepthSchemeResolver(std::string owner_suffix)
  : owner_suffix_(std::move(owner_suffix))
  {
    depth_material_ = Ogre::MaterialManager::getSingleton().getByName(
      kDepthMaterialName, kResourceGroup);
    if (depth_material_) {
      depth_material_->load();
      fast_clockwise_ = depth_material_->getTechnique("Clockwise");
      fast_anticlockwise_ = depth_material_->getTechnique("Anticlockwise");
      fast_no_cull_ = depth_material_->getTechnique("NoCull");
      no_depth_ = depth_material_->getTechnique("NoDepth");
    }
  }

  ~DepthSchemeResolver() override
  {
    auto & manager = Ogre::MaterialManager::getSingleton();
    for (auto & entry : conservative_materials_) {
      manager.remove(entry.second->getHandle());
    }
  }

  void beginFrame()
  {
    stats_ = {};
  }

  Stats stats() const
  {
    return stats_;
  }

  Ogre::Technique * handleSchemeNotFound(
    unsigned short,
    const Ogre::String & scheme_name,
    Ogre::Material * original_material,
    unsigned short lod_index,
    const Ogre::Renderable *) override
  {
    if (scheme_name != kDepthMaterialScheme || !original_material) {
      ++stats_.unresolved;
      return nullptr;
    }

    Ogre::Technique * source = findSourceTechnique(original_material, lod_index);
    if (!source) {
      ++stats_.unresolved;
      return nullptr;
    }

    if (!contributesDepth(source) && no_depth_) {
      ++stats_.conservative;
      return no_depth_;
    }

    if (isFastPathSafe(source)) {
      Ogre::Technique * fast = fastTechnique(source->getPass(0)->getCullingMode());
      if (fast) {
        ++stats_.fast;
        return fast;
      }
    }

    ++stats_.conservative;
    return conservativeTechnique(original_material, source, lod_index);
  }

private:
  using CacheKey = std::pair<Ogre::ResourceHandle, unsigned short>;

  static Ogre::Technique * findSourceTechnique(
    Ogre::Material * material, unsigned short lod_index)
  {
    material->load();

    Ogre::Technique * exact = nullptr;
    Ogre::Technique * nearest_lower = nullptr;
    Ogre::Technique * first_default = nullptr;
    Ogre::Technique * first_supported = nullptr;

    for (Ogre::Technique * technique : material->getSupportedTechniques()) {
      if (!first_supported) {
        first_supported = technique;
      }
      if (technique->getSchemeName() != Ogre::MaterialManager::DEFAULT_SCHEME_NAME) {
        continue;
      }
      if (!first_default) {
        first_default = technique;
      }
      if (technique->getLodIndex() == lod_index) {
        exact = technique;
        break;
      }
      if (technique->getLodIndex() < lod_index &&
        (!nearest_lower || technique->getLodIndex() > nearest_lower->getLodIndex()))
      {
        nearest_lower = technique;
      }
    }

    if (exact) {
      return exact;
    }
    if (nearest_lower) {
      return nearest_lower;
    }
    return first_default ? first_default : first_supported;
  }

  static bool isSimpleRigidVertexProgram(const Ogre::String & name)
  {
    if (name.empty()) {
      return true;
    }

    // RViz's Metal programs below differ in colour, texture, or lighting only;
    // all transform the POSITION semantic directly by the world-view-project
    // matrix. Custom and point-cloud programs deliberately fall back.
    return name == "rviz/flat_vp" || name == "rviz/vertex_color_vp" ||
           name == "rviz/texture_vp" || name == "rviz/lit_vp" ||
           name == "rviz/lit_texture_vp" || name == "rviz/metal/depth_vp";
  }

  static bool isFastPathSafe(Ogre::Technique * technique)
  {
    if (!technique || technique->getNumPasses() != 1) {
      return false;
    }

    Ogre::Pass * pass = technique->getPass(0);
    return pass && pass->getDepthCheckEnabled() && pass->getDepthWriteEnabled() &&
           !pass->isTransparent() &&
           pass->getAlphaRejectFunction() == Ogre::CMPF_ALWAYS_PASS &&
           isSimpleRigidVertexProgram(pass->getVertexProgramName());
  }

  static bool contributesDepth(Ogre::Technique * technique)
  {
    if (!technique) {
      return false;
    }
    for (unsigned short i = 0; i < technique->getNumPasses(); ++i) {
      Ogre::Pass * pass = technique->getPass(i);
      if (pass && pass->getDepthCheckEnabled() && pass->getDepthWriteEnabled()) {
        return true;
      }
    }
    return false;
  }

  Ogre::Technique * fastTechnique(Ogre::CullingMode mode) const
  {
    switch (mode) {
      case Ogre::CULL_CLOCKWISE:
        return fast_clockwise_;
      case Ogre::CULL_ANTICLOCKWISE:
        return fast_anticlockwise_;
      case Ogre::CULL_NONE:
        return fast_no_cull_;
    }
    return nullptr;
  }

  Ogre::Technique * conservativeTechnique(
    Ogre::Material * original_material, Ogre::Technique * source, unsigned short lod_index)
  {
    const CacheKey key(original_material->getHandle(), lod_index);
    auto cached = conservative_materials_.find(key);
    if (cached != conservative_materials_.end()) {
      return cached->second->getTechnique(0);
    }

    auto & manager = Ogre::MaterialManager::getSingleton();
    const Ogre::String name =
      "GaussianSplatting/DepthFallback/" + owner_suffix_ + "/" +
      std::to_string(original_material->getHandle()) + "/" + std::to_string(lod_index);
    Ogre::MaterialPtr material = manager.create(name, kResourceGroup);
    Ogre::Technique * technique = material->createTechnique();
    *technique = *source;
    technique->setSchemeName(kDepthMaterialScheme);
    for (unsigned short i = 0; i < technique->getNumPasses(); ++i) {
      technique->getPass(i)->setColourWriteEnabled(false);
    }
    material->load();
    conservative_materials_.emplace(key, material);
    return technique;
  }

  std::string owner_suffix_;
  Ogre::MaterialPtr depth_material_;
  Ogre::Technique * fast_clockwise_ = nullptr;
  Ogre::Technique * fast_anticlockwise_ = nullptr;
  Ogre::Technique * fast_no_cull_ = nullptr;
  Ogre::Technique * no_depth_ = nullptr;
  std::map<CacheKey, Ogre::MaterialPtr> conservative_materials_;
  Stats stats_;
};

class GaussianSplatRenderable : public Ogre::SimpleRenderable
{
public:
  explicit GaussianSplatRenderable(const Ogre::String & name)
  : Ogre::SimpleRenderable(name)
  {}

  ~GaussianSplatRenderable() override
  {
    OGRE_DELETE mRenderOp.vertexData;
    OGRE_DELETE mRenderOp.indexData;
    mRenderOp.vertexData = nullptr;
    mRenderOp.indexData = nullptr;
  }

  void setGeometry(
    Ogre::VertexData * vertex_data, Ogre::IndexData * index_data, std::size_t instance_count)
  {
    mRenderOp.operationType = Ogre::RenderOperation::OT_TRIANGLE_LIST;
    mRenderOp.useIndexes = true;
    mRenderOp.vertexData = vertex_data;
    mRenderOp.indexData = index_data;
    mRenderOp.numberOfInstances = instance_count;
    mRenderOp.useGlobalInstancingVertexBufferIsAvailable = false;
  }

  void setInstanceCount(std::size_t instance_count)
  {
    mRenderOp.numberOfInstances = instance_count;
  }

  Ogre::Real getBoundingRadius() const override
  {
    if (mBox.isNull() || mBox.isInfinite()) {
      return 0.0f;
    }
    return Ogre::Math::Sqrt(
      std::max(
        mBox.getMaximum().squaredLength(),
        mBox.getMinimum().squaredLength()));
  }

  Ogre::Real getSquaredViewDepth(const Ogre::Camera * camera) const override
  {
    if (!camera || mBox.isNull() || mBox.isInfinite()) {
      return 0.0f;
    }
    Ogre::Vector3 centre = mBox.getCenter();
    if (getParentSceneNode()) {
      centre = getParentSceneNode()->_getFullTransform() * centre;
    }
    return (camera->getDerivedPosition() - centre).squaredLength();
  }
};

GaussianSplattingDisplay::GaussianSplattingDisplay()
{
  const std::string suffix = std::to_string(reinterpret_cast<std::uintptr_t>(this));
  mesh_name_ = "GaussianSplattingRvizMesh_" + suffix;
  material_name_ = std::string(kMaterialName) + "_" + suffix;

  sigma_radius_property_ = new rviz_common::properties::FloatProperty(
    "Sigma Radius", 3.0f,
    "Extent of the rasterised quad in standard deviations. Fill rate scales "
    "with the square of this value, so lowering it is the cheapest way to "
    "trade accuracy for speed: 3.0 matches the reference rasteriser, 2.5 "
    "truncates the Gaussian at 4% of its peak, and 2.0 at 13% while costing "
    "2.25x fewer fragments.",
    this);
  sigma_radius_property_->setMin(0.5f);
  sigma_radius_property_->setMax(4.0f);

  culling_property_ = new rviz_common::properties::BoolProperty(
    "Visibility Culling", true,
    "Skip splats whose conservative 3D extent is fully outside the camera "
    "frustum, plus splats below the optional screen-size threshold. This also "
    "reduces the set that must be sorted and uploaded each time the camera moves.",
    this);

  min_screen_radius_property_ = new rviz_common::properties::FloatProperty(
    "Minimum Screen Radius", 0.0f,
    "Cull splats whose conservative visible radius is smaller than this many "
    "pixels. Zero preserves every in-frustum splat. Values around 0.5-1.0 can "
    "reduce distant detail and overdraw in large scenes.",
    culling_property_);
  min_screen_radius_property_->setMin(0.0f);
  min_screen_radius_property_->setMax(8.0f);

  offscreen_property_ = new rviz_common::properties::BoolProperty(
    "Offscreen Rendering", true,
    "Rasterise the splats into their own target and composite the result over "
    "the scene, instead of drawing them straight into the window.\n\n"
    "This is worth switching on even at full resolution: RViz configures the "
    "window for 4x MSAA, which costs four samples per fragment and does "
    "nothing for alpha blended splats, since they have no geometric edges to "
    "antialias. The offscreen target has no MSAA, which measured roughly 3x "
    "faster with no visible difference.\n\n"
    "The scene is drawn into the target once beforehand to lay down depth, so "
    "the splats are occluded by other displays exactly as they are when drawn "
    "straight into the window. At Render Scale below 1.0 that depth is lower "
    "resolution, so the occlusion boundary gets correspondingly coarser.",
    this);

  render_scale_property_ = new rviz_common::properties::FloatProperty(
    "Render Scale", 1.0f,
    "Fraction of the viewport resolution the splats are rasterised at. Fill "
    "rate scales with the square of this value, so 0.5 costs a quarter of the "
    "fragments in exchange for softer splats. Only the splats are affected; "
    "every other display stays sharp. 1.0 is a 1:1 composite and loses "
    "nothing.",
    offscreen_property_);
  render_scale_property_->setMin(0.25f);
  render_scale_property_->setMax(1.0f);

}

GaussianSplattingDisplay::~GaussianSplattingDisplay()
{
  destroyRenderTarget();

  if (depth_scheme_resolver_) {
    Ogre::MaterialManager::getSingleton().removeListener(
      depth_scheme_resolver_.get(), kDepthMaterialScheme);
    depth_scheme_resolver_.reset();
  }

  clearMesh();

  if (material_) {
    Ogre::MaterialManager::getSingleton().remove(material_);
    material_.reset();
  }

  if (composite_material_) {
    Ogre::MaterialManager::getSingleton().remove(composite_material_);
    composite_material_.reset();
  }

  if (composite_node_) {
    scene_manager_->destroySceneNode(composite_node_);
    composite_node_ = nullptr;
  }
  delete composite_rect_;
  composite_rect_ = nullptr;

  if (splat_node_) {
    scene_manager_->destroySceneNode(splat_node_);
    splat_node_ = nullptr;
  }
}

void GaussianSplattingDisplay::onInitialize()
{
  rviz_common::MessageFilterDisplay<GaussianSplats>::onInitialize();

  registerOgreResources();
#ifdef GSPLAT_HAS_METAL_PREPARATION
  if (usesMetalRenderSystem() && !std::getenv("GSPLAT_DISABLE_GPU_PREPARATION")) {
    const std::string shader_path =
      ament_index_cpp::get_package_share_directory("gaussian_splatting_rviz_plugins") +
      "/ogre_media/materials/programs/Metal/gsplat_prepare.metal";
    metal_view_preparation_ = makeMetalViewPreparation(
      Ogre::Root::getSingleton().getRenderSystem(), shader_path);
    if (!metal_view_preparation_ || !metal_view_preparation_->error().empty()) {
      const std::string error = metal_view_preparation_ ?
        metal_view_preparation_->error() : "Metal backend factory returned no implementation";
      setStatus(
        rviz_common::properties::StatusProperty::Warn, "GPU preparation",
        QString::fromStdString(error + "; using CPU fallback"));
      metal_view_preparation_.reset();
    } else {
      setStatus(
        rviz_common::properties::StatusProperty::Ok, "GPU preparation",
        "Metal compute pipelines loaded");
    }
  }
#endif
  depth_scheme_resolver_ = std::make_unique<DepthSchemeResolver>(mesh_name_);
  Ogre::MaterialManager::getSingleton().addListener(
    depth_scheme_resolver_.get(), kDepthMaterialScheme);
  splat_node_ = scene_manager_->getRootSceneNode()->createChildSceneNode();
}

void GaussianSplattingDisplay::reset()
{
  rviz_common::MessageFilterDisplay<GaussianSplats>::reset();
  clearMesh();
}

void GaussianSplattingDisplay::update(float wall_dt, float ros_dt)
{
  rviz_common::MessageFilterDisplay<GaussianSplats>::update(wall_dt, ros_dt);
  updateRenderTarget();
  if (splat_texture_) {
    setStatus(
      rviz_common::properties::StatusProperty::Ok, "Offscreen",
      QString("splats rasterised at %1x%2").arg(rtt_width_).arg(rtt_height_));
    if (depth_scheme_resolver_) {
      const auto stats = depth_scheme_resolver_->stats();
      setStatus(
        rviz_common::properties::StatusProperty::Ok, "Depth materials",
        QString("%1 fast, %2 conservative, %3 unresolved")
        .arg(stats.fast).arg(stats.conservative).arg(stats.unresolved));
    }
  } else {
    deleteStatus("Offscreen");
    deleteStatus("Depth materials");
  }
  applyShaderParams();
  sortIndexBuffer();
  if (renderable_) {
    if (using_gpu_preparation_) {
      setStatus(
        rviz_common::properties::StatusProperty::Ok, "Visible splats",
        QString("GPU indirect count / %1 total (no CPU readback)").arg(splat_count_));
      setStatus(
        rviz_common::properties::StatusProperty::Ok, "GPU preparation",
        QString("Metal queued in %1 ms CPU time; exact 32-bit depth order")
        .arg(last_sort_ms_, 0, 'f', 3));
      deleteStatus("CPU preparation");
      deleteStatus("CPU view stages");
    } else {
      setStatus(
        rviz_common::properties::StatusProperty::Ok, "Visible splats",
        QString("%1 / %2 instances").arg(visible_splat_count_).arg(splat_count_));
      setStatus(
        rviz_common::properties::StatusProperty::Ok, "CPU preparation",
        QString("message upload %1 ms, view total %2 ms")
        .arg(last_upload_ms_, 0, 'f', 2)
        .arg(last_sort_ms_, 0, 'f', 2));
      setStatus(
        rviz_common::properties::StatusProperty::Ok, "CPU view stages",
        QString("setup %1, cull/SH %2, merge %3, radix %4, gather %5, instance upload %6 ms")
        .arg(last_setup_ms_, 0, 'f', 2)
        .arg(last_cull_sh_ms_, 0, 'f', 2)
        .arg(last_merge_ms_, 0, 'f', 2)
        .arg(last_radix_ms_, 0, 'f', 2)
        .arg(last_gather_ms_, 0, 'f', 2)
        .arg(last_instance_upload_ms_, 0, 'f', 2));
    }
  } else {
    deleteStatus("Visible splats");
    deleteStatus("CPU preparation");
    deleteStatus("CPU view stages");
  }
}

void GaussianSplattingDisplay::processMessage(GaussianSplats::ConstSharedPtr msg)
{
  if (!msg) {
    return;
  }

  std::size_t count = 0;
  if (!validate(*msg, count)) {
    clearMesh();
    return;
  }

  Ogre::Vector3 frame_position;
  Ogre::Quaternion frame_orientation;
  if (!context_->getFrameManager()->getTransform(msg->header, frame_position, frame_orientation)) {
    setMissingTransformToFixedFrame(msg->header.frame_id);
    return;
  }

  setTransformOk();
  splat_node_->setPosition(frame_position);
  splat_node_->setOrientation(frame_orientation);

  if (count == 0) {
    clearMesh();
    return;
  }

  // Streaming sources (feed-forward models) emit a constant primitive count
  // because their output is pixel-aligned, so this reallocates only once.
  if (count != splat_count_) {
    allocateMesh(count);
  }

  // Held for as long as the harmonics are read out of it, which is until the
  // next message or until the display is cleared. Assigning here releases the
  // previous message, so the pointers into it are dropped in the same breath
  // rather than left dangling until uploadSplats re-points them.
  sh_dc_ = nullptr;
  sh_rest_ = nullptr;
  message_ = msg;

  eps2d_ = msg->eps2d;
  antialiased_ =
    msg->rasterize_mode == GaussianSplats::RASTERIZE_MODE_ANTIALIASED ? 1.0f : 0.0f;
  applyShaderParams();
  const auto upload_start = std::chrono::steady_clock::now();
  uploadSplats(*msg, count);
  last_upload_ms_ = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - upload_start).count();
  sortIndexBuffer();
}

bool GaussianSplattingDisplay::validate(const GaussianSplats & msg, std::size_t & count)
{
  if (msg.type == GaussianSplats::TYPE_2DGS) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Message",
      "TYPE_2DGS is not rendered yet: 2D Gaussians need ray/disc intersection "
      "rather than the EWA projection this display implements.");
    return false;
  }
  if (msg.type != GaussianSplats::TYPE_3DGS) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Message",
      QString("Unknown primitive type %1.").arg(msg.type));
    return false;
  }

  if (msg.means.size() % 3 != 0) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Message",
      "means length is not a multiple of 3.");
    return false;
  }

  count = msg.means.size() / 3;
  const auto expect =
    [&](const char * name, std::size_t actual, std::size_t wanted) {
      if (actual == wanted) {
        return true;
      }
      setStatus(
        rviz_common::properties::StatusProperty::Error, "Message",
        QString("%1 has %2 values, expected %3 for %4 primitives.")
        .arg(name).arg(actual).arg(wanted).arg(count));
      return false;
    };

  if (!expect("scales", msg.scales.size(), count * 3) ||
    !expect("quats", msg.quats.size(), count * 4) ||
    !expect("opacities", msg.opacities.size(), count) ||
    !expect("sh_dc", msg.sh_dc.size(), count * 3))
  {
    return false;
  }

  if (msg.sh_degree > kMaxShDegree) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Message",
      QString("sh_degree is %1; this display evaluates up to %2.")
      .arg(msg.sh_degree).arg(kMaxShDegree));
    return false;
  }

  // (degree + 1)^2 coefficients in total, of which the first is sh_dc. The
  // stride into sh_rest is derived from this, so a wrong size here would be
  // read as a different degree rather than rejected.
  const std::size_t coefficients =
    (msg.sh_degree + 1u) * (msg.sh_degree + 1u) - 1u;
  if (!expect("sh_rest", msg.sh_rest.size(), count * coefficients * 3)) {
    return false;
  }

  // Refused here rather than at createManual(): Metal reports an oversized
  // texture by aborting the process, so there is nothing to catch downstream.
  const unsigned int rows = splatDataRows(count);
  if (rows > kMaxTextureDimension) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Message",
      QString("%1 splats need a %2 x %3 data texture, past the %4 this "
      "render system allows.")
      .arg(count).arg(splatDataWidth(count)).arg(rows).arg(kMaxTextureDimension));
    return false;
  }

  // The OpenGL vertex program indexes texels in floats, so past 2^24 texels it
  // reads the wrong record for some splats. Metal indexes in uint and is exact.
  if (!usesMetalRenderSystem() && count * kTexelsPerSplat > kMaxGlslExactTexels) {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Message",
      QString("%1 splats: past %2, the OpenGL shader's float texel index is no "
      "longer exact and some splats will read a neighbour's record.")
      .arg(count).arg(kMaxGlslExactTexels / kTexelsPerSplat));
    return true;
  }

  setStatus(
    rviz_common::properties::StatusProperty::Ok, "Message",
    QString("%1 splats").arg(count));
  return true;
}

void GaussianSplattingDisplay::registerOgreResources()
{
  if (resources_registered_) {
    return;
  }

  auto & rgm = Ogre::ResourceGroupManager::getSingleton();
  if (!rgm.resourceGroupExists(kResourceGroup)) {
    rgm.createResourceGroup(kResourceGroup);
  }

  const std::string share_dir =
    ament_index_cpp::get_package_share_directory("gaussian_splatting_rviz_plugins");
  const std::string media_dir = share_dir + "/ogre_media";

  if (usesMetalRenderSystem()) {
    rgm.addResourceLocation(media_dir + "/materials/scriptsMetal", "FileSystem", kResourceGroup);
    rgm.addResourceLocation(media_dir + "/materials/programs/Metal", "FileSystem", kResourceGroup);
  } else {
    rgm.addResourceLocation(media_dir + "/materials/scripts", "FileSystem", kResourceGroup);
    rgm.addResourceLocation(media_dir + "/materials/programs/GLSL", "FileSystem", kResourceGroup);
  }
  rgm.initialiseResourceGroup(kResourceGroup);

  // Clone the shared material so that eps2d / rasterize_mode set from one
  // message cannot leak into another display instance.
  Ogre::MaterialPtr base =
    Ogre::MaterialManager::getSingleton().getByName(kMaterialName, kResourceGroup);
  if (!base) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "Material",
      QString("Material '%1' was not found in %2. This usually means the shader "
      "failed to compile; see Ogre.log in the directory rviz2 was started from.")
      .arg(kMaterialName).arg(usesMetalRenderSystem() ? "Metal" : "GLSL"));
    resources_registered_ = true;
    return;
  }

  material_ = base->clone(material_name_, true, kResourceGroup);
  material_->load();
  setStatus(rviz_common::properties::StatusProperty::Ok, "Material", "loaded");

  resources_registered_ = true;
}

void GaussianSplattingDisplay::clearMesh()
{
#ifdef GSPLAT_HAS_METAL_PREPARATION
  if (metal_view_preparation_) {
    metal_view_preparation_->clear();
  }
#endif
  if (renderable_) {
    if (splat_node_) {
      splat_node_->detachObject(renderable_.get());
    }
    renderable_.reset();
  }

  instance_buffer_.reset();

  splat_count_ = 0;
  visible_splat_count_ = 0;
  positions_.clear();
  order_.clear();
  order_scratch_.clear();
  records_.clear();
  draw_instances_.clear();
  colours_.clear();
  message_.reset();
  sh_dc_ = nullptr;
  sh_rest_ = nullptr;
  sh_coefficients_ = 0;
  if (splat_data_texture_) {
    Ogre::TextureManager::getSingleton().remove(splat_data_texture_);
    splat_data_texture_.reset();
  }
  last_camera_position_ = Ogre::Vector3::ZERO;
  last_camera_direction_ = Ogre::Vector3::ZERO;
  last_viewport_width_ = 0;
  last_viewport_height_ = 0;
  last_sigma_radius_ = -1.0f;
  last_min_screen_radius_ = -1.0f;
  last_culling_enabled_ = false;
  last_upload_ms_ = 0.0;
  last_sort_ms_ = 0.0;
  last_setup_ms_ = 0.0;
  last_cull_sh_ms_ = 0.0;
  last_merge_ms_ = 0.0;
  last_radix_ms_ = 0.0;
  last_gather_ms_ = 0.0;
  last_instance_upload_ms_ = 0.0;
  using_gpu_preparation_ = false;
  sort_dirty_ = true;
}

void GaussianSplattingDisplay::allocateMesh(std::size_t count)
{
  clearMesh();

  splat_count_ = count;

  positions_.resize(count);
  order_.reserve(count);
  order_scratch_.resize(count);
  records_.resize(count);
  draw_instances_.resize(count);
  colours_.assign(count * 3, 0.0f);
  createSplatDataTexture(count);

  auto * vertex_data = OGRE_NEW Ogre::VertexData();
  vertex_data->vertexCount = kVerticesPerSplat;

  // A float rather than an integer type: indices are exact in float32 up to
  // 2^24, far past any splat count that fits in memory, and this keeps the
  // attribute a plain float on both back ends.
  Ogre::VertexDeclaration * declaration = vertex_data->vertexDeclaration;
  declaration->addElement(
    0, offsetof(DrawInstance, index), Ogre::VET_FLOAT1, Ogre::VES_TEXTURE_COORDINATES, 0);
  declaration->addElement(
    0, offsetof(DrawInstance, colour), Ogre::VET_FLOAT3, Ogre::VES_DIFFUSE);
  declaration->addElement(
    1, 0, Ogre::VET_FLOAT2, Ogre::VES_TEXTURE_COORDINATES, 2);

  auto & hbm = Ogre::HardwareBufferManager::getSingleton();
  const auto dynamic = Ogre::HardwareBuffer::HBU_DYNAMIC_WRITE_ONLY;
  const auto stat1c = Ogre::HardwareBuffer::HBU_STATIC_WRITE_ONLY;

  instance_buffer_ = hbm.createVertexBuffer(sizeof(DrawInstance), count, dynamic);
  instance_buffer_->setIsInstanceData(true);
  instance_buffer_->setInstanceDataStepRate(1);
  vertex_data->vertexBufferBinding->setBinding(0, instance_buffer_);

  // The corner offsets never change, so this buffer is written once here and
  // left alone for the lifetime of the mesh.
  Ogre::HardwareVertexBufferSharedPtr corner_buffer =
    hbm.createVertexBuffer(sizeof(float) * 2, kVerticesPerSplat, stat1c);
  float corner_data[kVerticesPerSplat * 2];
  for (std::size_t corner = 0; corner < kVerticesPerSplat; ++corner) {
    corner_data[corner * 2] = kCorners[corner][0];
    corner_data[corner * 2 + 1] = kCorners[corner][1];
  }
  corner_buffer->writeData(0, corner_buffer->getSizeInBytes(), corner_data, true);
  vertex_data->vertexBufferBinding->setBinding(1, corner_buffer);

  auto * index_data = OGRE_NEW Ogre::IndexData();
  index_data->indexCount = kIndicesPerSplat;
  index_data->indexBuffer = hbm.createIndexBuffer(
    Ogre::HardwareIndexBuffer::IT_16BIT, kIndicesPerSplat, stat1c);
  const std::uint16_t quad_indices[kIndicesPerSplat] = {
    0, 1, 2,
    0, 2, 3,
  };
  index_data->indexBuffer->writeData(
    0, index_data->indexBuffer->getSizeInBytes(), quad_indices, true);

  renderable_ = std::make_unique<GaussianSplatRenderable>(mesh_name_);
  renderable_->setGeometry(vertex_data, index_data, count);
  renderable_->setMaterial(material_);
  renderable_->setBoundingBox(Ogre::AxisAlignedBox::BOX_INFINITE);
  renderable_->setVisible(!splat_texture_);
  renderable_->setRenderQueueGroup(kSplatRenderQueue);
  splat_node_->attachObject(renderable_.get());

#ifdef GSPLAT_HAS_METAL_PREPARATION
  if (metal_view_preparation_ && !metal_view_preparation_->configure(
      count, instance_buffer_, renderable_.get()))
  {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "GPU preparation",
      QString::fromStdString(metal_view_preparation_->error() + "; using CPU fallback"));
  }
#endif
}

void GaussianSplattingDisplay::createSplatDataTexture(std::size_t count)
{
  if (count == 0) {
    return;
  }

  auto & texture_manager = Ogre::TextureManager::getSingleton();
  splat_data_texture_ = texture_manager.createManual(
    mesh_name_ + "/data", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
    Ogre::TEX_TYPE_2D, splatDataWidth(count), splatDataRows(count), 0,
    Ogre::PF_FLOAT32_RGBA, Ogre::TU_DYNAMIC_WRITE_ONLY);

  // The shader reads texels by integer coordinate, so this never samples and
  // never filters; the state is set anyway because a pass with a texture unit
  // gets a sampler regardless, and RGBA32F is not filterable on Apple GPUs.
  if (material_) {
    Ogre::Pass * pass = material_->getTechnique(0)->getPass(0);
    if (pass->getNumTextureUnitStates() == 0) {
      pass->createTextureUnitState();
    }
    Ogre::TextureUnitState * unit = pass->getTextureUnitState(0);
    unit->setTexture(splat_data_texture_);
    unit->setTextureFiltering(Ogre::TFO_NONE);
    unit->setTextureAddressingMode(Ogre::TextureUnitState::TAM_CLAMP);
  }

  applyShaderParams();
}

void GaussianSplattingDisplay::writeSplatDataTexture()
{
  if (!splat_data_texture_ || records_.empty()) {
    return;
  }

  // The last row is only partly used, so lock and fill row by row rather than
  // handing the whole buffer a short source box.
  const Ogre::HardwarePixelBufferSharedPtr buffer = splat_data_texture_->getBuffer();
  buffer->lock(Ogre::HardwareBuffer::HBL_DISCARD);
  const Ogre::PixelBox & box = buffer->getCurrentLock();
  auto * destination = reinterpret_cast<float *>(box.data);
  const std::size_t row_floats = box.rowPitch * 4;
  const std::size_t floats_per_splat = kTexelsPerSplat * 4;
  const std::size_t splats_per_row = splat_data_texture_->getWidth() / kTexelsPerSplat;

  const std::size_t rows = (records_.size() + splats_per_row - 1) / splats_per_row;
  parallelFor(
    rows, workerCount(records_.size()), [&](std::size_t begin, std::size_t end, std::size_t) {
      for (std::size_t row = begin; row < end; ++row) {
        const std::size_t first = row * splats_per_row;
        const std::size_t here = std::min(splats_per_row, records_.size() - first);
        std::memcpy(
          destination + row * row_floats, &records_[first],
          here * floats_per_splat * sizeof(float));
      }
    });
  buffer->unlock();
}

void GaussianSplattingDisplay::uploadSplats(const GaussianSplats & msg, std::size_t count)
{
  // Each worker writes a disjoint range of positions_ and records_, and keeps
  // its own bounds so the reduction needs no lock.
  const std::size_t workers = workerCount(count);
  std::vector<Ogre::AxisAlignedBox> worker_bounds(workers);
  parallelFor(
    count, workers, [&](std::size_t begin, std::size_t end, std::size_t worker) {
      Ogre::AxisAlignedBox local;
      local.setNull();
      for (std::size_t i = begin; i < end; ++i) {
        const Ogre::Vector3 position(msg.means[i * 3], msg.means[i * 3 + 1], msg.means[i * 3 + 2]);
        positions_[i] = position;
        local.merge(position);

        SplatRecord & record = records_[i];
        std::copy_n(&msg.means[i * 3], 3, record.position);
        record.opacity = clamp(msg.opacities[i], 0.0f, 1.0f);
        std::copy_n(&msg.scales[i * 3], 3, record.scale);
        std::copy_n(&msg.quats[i * 4], 4, record.quat);
        record.pad0 = 0.0f;
      }
      worker_bounds[worker] = local;
    });

  Ogre::AxisAlignedBox bounds;
  bounds.setNull();
  for (const Ogre::AxisAlignedBox & local : worker_bounds) {
    bounds.merge(local);
  }

  // Colour cannot be baked in here: beyond degree 0 it depends on the view
  // direction, so the coefficients are kept and evaluated whenever the camera
  // moves. That is the same trigger the depth sort already uses.
  // Pointed at, not copied: message_ holds the message alive for exactly as
  // long as these stay in use.
  sh_coefficients_ = (msg.sh_degree + 1u) * (msg.sh_degree + 1u) - 1u;
  sh_dc_ = msg.sh_dc.data();
  sh_rest_ = msg.sh_rest.empty() ? nullptr : msg.sh_rest.data();

  writeSplatDataTexture();

#ifdef GSPLAT_HAS_METAL_PREPARATION
  if (metal_view_preparation_ && !metal_view_preparation_->uploadStaticData(
      records_.data(), records_.size() * sizeof(SplatRecord),
      sh_dc_, msg.sh_dc.size(), sh_rest_, msg.sh_rest.size()))
  {
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "GPU preparation",
      QString::fromStdString(metal_view_preparation_->error() + "; using CPU fallback"));
  }
#endif

  if (renderable_) {
    renderable_->setBoundingBox(bounds);
  }

  // Force a fresh depth sort now that the geometry has changed.
  sort_dirty_ = true;
}

void GaussianSplattingDisplay::applyShaderParams()
{
  if (!material_) {
    return;
  }

  Ogre::Technique * technique = material_->getBestTechnique();
  if (!technique || technique->getNumPasses() == 0) {
    return;
  }

  Ogre::Pass * pass = technique->getPass(0);
  if (!pass->hasVertexProgram()) {
    return;
  }

  // Pushed every frame rather than on change: setNamedConstant is a cheap
  // write into the parameter block, and this avoids needing moc for a
  // property-changed slot.
  const Ogre::GpuProgramParametersSharedPtr params = pass->getVertexProgramParameters();
  params->setNamedConstant("eps2d", eps2d_);
  params->setNamedConstant("antialiased", antialiased_);
  params->setNamedConstant("sigma_radius", sigma_radius_property_->getFloat());

  // The shader turns a splat index into a texel coordinate, so it needs the
  // texture's dimensions. GLSL also needs the reciprocals, since GLSL 120 has
  // no integer texel fetch and has to sample at normalised coordinates.
  const float width = splat_data_texture_ ?
    static_cast<float>(splat_data_texture_->getWidth()) : 1.0f;
  const float height = splat_data_texture_ ?
    static_cast<float>(splat_data_texture_->getHeight()) : 1.0f;
  params->setNamedConstant(
    "splat_data_size", Ogre::Vector4(width, height, 1.0f / width, 1.0f / height));
}

void GaussianSplattingDisplay::writeSortedInstances()
{
  if (!instance_buffer_ || !renderable_) {
    return;
  }

  const auto gather_start = std::chrono::steady_clock::now();
  // The shader reads the record from the data texture, so only the index and
  // the view-dependent colour move: sixteen bytes per splat, not a whole
  // record. Each worker writes a disjoint range of draw_instances_.
  parallelFor(
    order_.size(), workerCount(order_.size()),
    [&](std::size_t begin, std::size_t end, std::size_t) {
      for (std::size_t output = begin; output < end; ++output) {
        const auto splat = static_cast<std::uint32_t>(order_[output]);
        DrawInstance & instance = draw_instances_[output];
        instance.index = static_cast<float>(splat);
        std::copy_n(&colours_[splat * 3], 3, instance.colour);
      }
    });
  const auto gather_end = std::chrono::steady_clock::now();
  last_gather_ms_ = std::chrono::duration<double, std::milli>(
    gather_end - gather_start).count();

  visible_splat_count_ = order_.size();
  renderable_->setInstanceCount(visible_splat_count_);
  const auto upload_start = std::chrono::steady_clock::now();
  if (visible_splat_count_ > 0) {
    instance_buffer_->writeData(
      0, visible_splat_count_ * sizeof(DrawInstance), draw_instances_.data(), true);
  }
  last_instance_upload_ms_ = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - upload_start).count();
}

namespace
{

// Maps a float onto a uint32 whose unsigned order matches the float's ordering,
// so depths can be sorted by their bits: flip the sign bit for positives, and
// invert everything for negatives, whose magnitude ordering runs backwards.
// This also gives NaN a defined place, which the old comparator did not.
std::uint32_t depthKey(float depth)
{
  std::uint32_t bits;
  std::memcpy(&bits, &depth, sizeof(bits));
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Least-significant-byte-first radix sort on the high word, i.e. on the depth
// key, leaving the packed index along for the ride. Four linear passes rather
// than the n log n random accesses a comparison sort spends here.
void radixSortByHighWord(
  std::vector<std::uint64_t> & values, std::vector<std::uint64_t> & scratch)
{
  const std::size_t count = values.size();
  if (count < 2) {
    return;
  }
  if (scratch.size() < count) {
    scratch.resize(count);
  }

  std::uint64_t * source = values.data();
  std::uint64_t * target = scratch.data();
  for (int shift = 32; shift < 64; shift += 8) {
    std::size_t offset[257] = {0};
    for (std::size_t i = 0; i < count; ++i) {
      ++offset[((source[i] >> shift) & 0xFF) + 1];
    }
    for (int bucket = 0; bucket < 256; ++bucket) {
      offset[bucket + 1] += offset[bucket];
    }
    for (std::size_t i = 0; i < count; ++i) {
      target[offset[(source[i] >> shift) & 0xFF]++] = source[i];
    }
    std::swap(source, target);
  }

  // Four passes is even, so the result is already back in values; the copy is
  // only here so the pass count can change without silently drawing garbage.
  if (source != values.data()) {
    std::memcpy(values.data(), source, count * sizeof(std::uint64_t));
  }
}

// Evaluates the spherical harmonics for one splat along a unit view direction.
// This is eval_sh() from the 3DGS reference implementation, degree for degree;
// `rest` is laid out coefficient major, matching sh_rest in GaussianSplats.msg.
// Like the reference it clamps below at zero and not above, leaving highlights
// brighter than white to be resolved by blending.
void evaluateColour(
  const float * dc, const float * rest, std::size_t coefficients, const Ogre::Vector3 & view,
  float * colour)
{
  float channel[3];
  for (int c = 0; c < 3; ++c) {
    channel[c] = kSHC0 * dc[c] + 0.5f;
  }

  const float x = view.x;
  const float y = view.y;
  const float z = view.z;

  if (coefficients >= 3) {
    for (int c = 0; c < 3; ++c) {
      channel[c] += -kSHC1 * y * rest[c] + kSHC1 * z * rest[3 + c] - kSHC1 * x * rest[6 + c];
    }
  }
  if (coefficients >= 8) {
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    for (int c = 0; c < 3; ++c) {
      channel[c] +=
        kSHC2[0] * xy * rest[9 + c] +
        kSHC2[1] * yz * rest[12 + c] +
        kSHC2[2] * (2.0f * zz - xx - yy) * rest[15 + c] +
        kSHC2[3] * xz * rest[18 + c] +
        kSHC2[4] * (xx - yy) * rest[21 + c];
    }
  }
  if (coefficients >= 15) {
    const float xx = x * x, yy = y * y, zz = z * z, xy = x * y;
    for (int c = 0; c < 3; ++c) {
      channel[c] +=
        kSHC3[0] * y * (3.0f * xx - yy) * rest[24 + c] +
        kSHC3[1] * xy * z * rest[27 + c] +
        kSHC3[2] * y * (4.0f * zz - xx - yy) * rest[30 + c] +
        kSHC3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * rest[33 + c] +
        kSHC3[4] * x * (4.0f * zz - xx - yy) * rest[36 + c] +
        kSHC3[5] * z * (xx - yy) * rest[39 + c] +
        kSHC3[6] * x * (xx - 3.0f * yy) * rest[42 + c];
    }
  }

  for (int c = 0; c < 3; ++c) {
    colour[c] = std::max(0.0f, channel[c]);
  }
}

}  // namespace

void GaussianSplattingDisplay::sortIndexBuffer()
{
  if (!instance_buffer_ || positions_.empty() || !sh_dc_ || !context_ ||
    !context_->getViewManager())
  {
    return;
  }

  rviz_common::ViewController * view_controller = context_->getViewManager()->getCurrent();
  if (!view_controller) {
    return;
  }

  Ogre::Camera * camera = view_controller->getCamera();
  if (!camera) {
    return;
  }

  const Ogre::Vector3 camera_position = camera->getDerivedPosition();
  const Ogre::Vector3 camera_direction = camera->getDerivedDirection();
  Ogre::Viewport * viewport = camera->getViewport();
  const unsigned int viewport_width = viewport ? viewport->getActualWidth() : 0;
  const unsigned int viewport_height = viewport ? viewport->getActualHeight() : 0;
  const bool culling_enabled = culling_property_->getBool();
  const float min_screen_radius = min_screen_radius_property_->getFloat();
  const float sigma_radius = sigma_radius_property_->getFloat();
  if (!sort_dirty_ &&
    (camera_position - last_camera_position_).squaredLength() < 0.0001f &&
    (camera_direction - last_camera_direction_).squaredLength() < 0.000001f &&
    viewport_width == last_viewport_width_ && viewport_height == last_viewport_height_ &&
    culling_enabled == last_culling_enabled_ &&
    std::abs(min_screen_radius - last_min_screen_radius_) < 0.0001f &&
    std::abs(sigma_radius - last_sigma_radius_) < 0.0001f)
  {
    return;
  }

  last_camera_position_ = camera_position;
  last_camera_direction_ = camera_direction;
  last_viewport_width_ = viewport_width;
  last_viewport_height_ = viewport_height;
  last_culling_enabled_ = culling_enabled;
  last_min_screen_radius_ = min_screen_radius;
  last_sigma_radius_ = sigma_radius;
  sort_dirty_ = false;
  const auto sort_start = std::chrono::steady_clock::now();

  order_.clear();

  const Ogre::Vector3 local_sort_direction =
    splat_node_->_getDerivedOrientation().Inverse() * (-camera_direction);

  // Spherical harmonics are expressed in the splats' own frame, so the view
  // direction has to be taken into that frame rather than the fixed frame.
  // This is exactly why the publisher corrects a scene's orientation with a
  // transform instead of rotating the data: rotating it would need the
  // coefficients rotated too, by Wigner D matrices for degree >= 1.
  const Ogre::Vector3 local_camera =
    splat_node_->_getFullTransform().inverse() * camera_position;
  const std::size_t rest_stride = sh_coefficients_ * 3;

  const Ogre::Matrix4 world_transform = splat_node_->_getFullTransform();
  const Ogre::Vector3 derived_scale = splat_node_->_getDerivedScale();
  const float world_scale = std::max(
    std::abs(derived_scale.x),
    std::max(std::abs(derived_scale.y), std::abs(derived_scale.z)));
  const Ogre::Matrix4 view_matrix = camera->getViewMatrix(true);
  const float alpha_cutoff = 1.0f / 255.0f;
  const bool perspective = camera->getProjectionType() == Ogre::PT_PERSPECTIVE;
  const float focal_y = viewport_height > 0 && perspective ?
    static_cast<float>(viewport_height) * 0.5f /
    std::tan(static_cast<float>(camera->getFOVy().valueRadians()) * 0.5f) : 0.0f;
  const float near_clip = camera->getNearClipDistance();
  const float far_clip = camera->getFarClipDistance();
  const float ortho_scale = perspective ? 0.0f :
    std::abs(static_cast<float>(camera->getProjectionMatrix()[1][1])) *
    static_cast<float>(viewport_height) * 0.5f;

  // Everything the loop needs from the camera is taken here, on the render
  // thread. Camera::isVisible() and getProjectionMatrix() update lazily cached
  // state on first call, which several workers calling at once would race on;
  // reading them now leaves the loop with nothing but plain values. The plane
  // test below is Frustum::isVisible(Sphere) verbatim, far-plane case included.
  Ogre::Plane frustum_planes[6];
  std::copy_n(camera->getFrustumPlanes(), 6, frustum_planes);

#ifdef GSPLAT_HAS_METAL_PREPARATION
  if (metal_view_preparation_) {
    MetalViewParameters parameters{};
    const auto copy_matrix = [](const Ogre::Matrix4 & matrix, float * rows) {
        for (int row = 0; row < 4; ++row) {
          for (int column = 0; column < 4; ++column) {
            rows[row * 4 + column] = static_cast<float>(matrix[row][column]);
          }
        }
      };
    copy_matrix(world_transform, parameters.world_rows);
    copy_matrix(view_matrix, parameters.view_rows);
    for (int plane = 0; plane < 6; ++plane) {
      const bool disabled_far = plane == Ogre::FRUSTUM_PLANE_FAR && far_clip == 0.0f;
      parameters.frustum_planes[plane * 4] =
        disabled_far ? 0.0f : static_cast<float>(frustum_planes[plane].normal.x);
      parameters.frustum_planes[plane * 4 + 1] =
        disabled_far ? 0.0f : static_cast<float>(frustum_planes[plane].normal.y);
      parameters.frustum_planes[plane * 4 + 2] =
        disabled_far ? 0.0f : static_cast<float>(frustum_planes[plane].normal.z);
      parameters.frustum_planes[plane * 4 + 3] =
        disabled_far ? 0.0f : static_cast<float>(frustum_planes[plane].d);
    }
    parameters.local_camera[0] = local_camera.x;
    parameters.local_camera[1] = local_camera.y;
    parameters.local_camera[2] = local_camera.z;
    parameters.local_sort_direction[0] = local_sort_direction.x;
    parameters.local_sort_direction[1] = local_sort_direction.y;
    parameters.local_sort_direction[2] = local_sort_direction.z;
    parameters.world_scale = world_scale;
    parameters.focal_y = focal_y;
    parameters.near_clip = near_clip;
    parameters.far_clip = far_clip;
    parameters.ortho_scale = ortho_scale;
    parameters.sigma_radius = sigma_radius;
    parameters.min_screen_radius = min_screen_radius;
    parameters.eps2d = eps2d_;
    parameters.count = static_cast<std::uint32_t>(splat_count_);
    parameters.viewport_height = viewport_height;
    parameters.sh_coefficients = static_cast<std::uint32_t>(sh_coefficients_);
    parameters.flags = (culling_enabled ? 1u : 0u) | (perspective ? 2u : 0u);

    if (metal_view_preparation_->encode(parameters)) {
      if (!using_gpu_preparation_) {
        RCLCPP_INFO(
          rclcpp::get_logger("gaussian_splatting_rviz_plugins"),
          "Metal GPU view preparation enabled: %zu splats, exact 32-bit radix, indirect draw",
          splat_count_);
      }
      renderable_->setInstanceCount(splat_count_);
      using_gpu_preparation_ = true;
      last_setup_ms_ = 0.0;
      last_cull_sh_ms_ = 0.0;
      last_merge_ms_ = 0.0;
      last_radix_ms_ = 0.0;
      last_gather_ms_ = 0.0;
      last_instance_upload_ms_ = 0.0;
      last_sort_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - sort_start).count();
      return;
    }

    setStatus(
      rviz_common::properties::StatusProperty::Warn, "GPU preparation",
      QString::fromStdString(metal_view_preparation_->error() + "; using CPU fallback"));
  }
#endif

  using_gpu_preparation_ = false;

  const std::size_t workers = workerCount(splat_count_);
  cull_partitions_.resize(workers);
  const auto cull_start = std::chrono::steady_clock::now();
  last_setup_ms_ = std::chrono::duration<double, std::milli>(
    cull_start - sort_start).count();
  parallelFor(
    splat_count_, workers, [&](std::size_t begin, std::size_t end, std::size_t worker) {
    std::vector<std::uint64_t> & partition = cull_partitions_[worker];
    partition.clear();
    partition.reserve(end - begin);

    for (std::uint32_t i = begin; i < end; ++i) {
      const SplatRecord & record = records_[i];
      if (record.opacity < alpha_cutoff) {
        continue;
      }

      const float visible_radius = std::min(
        sigma_radius,
        std::sqrt(2.0f * std::log(record.opacity / alpha_cutoff)));
      const float max_sigma = std::max(
        record.scale[0], std::max(record.scale[1], record.scale[2]));
      const float world_radius = visible_radius * max_sigma * world_scale;
      const Ogre::Vector3 world_position = world_transform * positions_[i];

      if (culling_enabled) {
        bool outside = false;
        for (int plane = 0; plane < 6; ++plane) {
          if (plane == Ogre::FRUSTUM_PLANE_FAR && far_clip == 0.0f) {
            continue;
          }
          if (frustum_planes[plane].getDistance(world_position) < -world_radius) {
            outside = true;
            break;
          }
        }
        if (outside) {
          continue;
        }
      }

      if (culling_enabled && min_screen_radius > 0.0f && viewport_height > 0) {
        float screen_radius = 0.0f;
        if (perspective) {
          const Ogre::Vector3 view_position = view_matrix * world_position;
          const float depth = -view_position.z;
          if (depth <= 0.0f) {
            continue;
          }
          screen_radius = focal_y * world_radius / std::max(depth - world_radius, near_clip);
        } else {
          screen_radius = ortho_scale * world_radius;
        }

        // eps2d adds this minimum variance in screen space before rasterisation.
        const float blur_radius = visible_radius * std::sqrt(std::max(eps2d_, 0.0f));
        screen_radius = std::sqrt(screen_radius * screen_radius + blur_radius * blur_radius);
        if (screen_radius < min_screen_radius) {
          continue;
        }
      }

      // Only the survivors get a colour: at a typical camera this is a third of
      // the scene, and the evaluation is the most expensive thing in the loop.
      evaluateColour(
        &sh_dc_[i * 3],
        rest_stride > 0 ? &sh_rest_[i * rest_stride] : nullptr,
        sh_coefficients_,
        (positions_[i] - local_camera).normalisedCopy(),
        &colours_[i * 3]);

      const std::uint32_t key = depthKey(positions_[i].dotProduct(local_sort_direction));
      partition.push_back((static_cast<std::uint64_t>(key) << 32) | i);
    }
  });
  const auto cull_end = std::chrono::steady_clock::now();
  last_cull_sh_ms_ = std::chrono::duration<double, std::milli>(
    cull_end - cull_start).count();

  // Concatenated in worker order, so the input to the sort does not depend on
  // how the workers happened to interleave.
  const auto merge_start = std::chrono::steady_clock::now();
  for (const std::vector<std::uint64_t> & partition : cull_partitions_) {
    order_.insert(order_.end(), partition.begin(), partition.end());
  }
  const auto merge_end = std::chrono::steady_clock::now();
  last_merge_ms_ = std::chrono::duration<double, std::milli>(
    merge_end - merge_start).count();

  const auto radix_start = std::chrono::steady_clock::now();
  radixSortByHighWord(order_, order_scratch_);
  const auto radix_end = std::chrono::steady_clock::now();
  last_radix_ms_ = std::chrono::duration<double, std::milli>(
    radix_end - radix_start).count();

  writeSortedInstances();
  last_sort_ms_ = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - sort_start).count();

  if (std::getenv("GSPLAT_PROFILE_CPU")) {
    RCLCPP_INFO(
      rclcpp::get_logger("gaussian_splatting_rviz_plugins"),
      "CPU view preparation: total=%.3f ms setup=%.3f cull_sh=%.3f merge=%.3f "
      "radix=%.3f gather=%.3f instance_upload=%.3f visible=%zu total_splats=%zu",
      last_sort_ms_, last_setup_ms_, last_cull_sh_ms_, last_merge_ms_,
      last_radix_ms_, last_gather_ms_, last_instance_upload_ms_,
      visible_splat_count_, splat_count_);
  }
}

Ogre::Viewport * GaussianSplattingDisplay::mainViewport() const
{
  if (!context_ || !context_->getViewManager()) {
    return nullptr;
  }
  rviz_common::RenderPanel * panel = context_->getViewManager()->getRenderPanel();
  if (!panel || !panel->getRenderWindow()) {
    return nullptr;
  }
  return rviz_rendering::RenderWindowOgreAdapter::getOgreViewport(panel->getRenderWindow());
}

void GaussianSplattingDisplay::setSplatsVisible(bool visible)
{
  if (renderable_) {
    renderable_->setVisible(visible);
  }
}

void GaussianSplattingDisplay::preRenderTargetUpdate(const Ogre::RenderTargetEvent & event)
{
  if (!splat_texture_) {
    return;
  }

  const bool is_offscreen = event.source == splat_texture_->getBuffer()->getRenderTarget();

  // The quad belongs only in the main window; the splats never do, since the
  // window gets them through the quad. Which of the offscreen target's two
  // passes draws the splats is decided in preViewportUpdate().
  if (composite_rect_) {
    composite_rect_->setVisible(!is_offscreen && event.source == main_target_);
  }
  if (!is_offscreen && renderable_) {
    renderable_->setVisible(false);
  }
}

void GaussianSplattingDisplay::preViewportUpdate(const Ogre::RenderTargetViewportEvent & event)
{
  if (!splat_texture_) {
    return;
  }

  const bool is_depth_pass = event.source == depth_viewport_;
  const bool is_splat_pass = event.source == rtt_viewport_;
  if (!is_depth_pass && !is_splat_pass) {
    return;
  }

  if (renderable_) {
    renderable_->setVisible(is_splat_pass);
  }

  if (is_splat_pass) {
    // The depth-only viewport deliberately leaves colour writes disabled. In
    // OpenGL, glClear honours that write mask, so the automatic transparent
    // clear at the start of this viewport otherwise does nothing and old splat
    // frames accumulate into long trails as the camera moves. Metal attachment
    // clears ignore the previous pipeline's mask, which is why the bug was
    // backend-specific. Restore an ordinary colour state before Ogre performs
    // the viewport clear; the splat material installs its blend state again
    // before drawing.
    if (Ogre::RenderSystem * render_system = Ogre::Root::getSingleton().getRenderSystem()) {
      render_system->setColourBlendState(Ogre::ColourBlendState());
    }
  }

  if (is_depth_pass && depth_scheme_resolver_) {
    depth_scheme_resolver_->beginFrame();
  }

  // The depth pass renders every queue except the hidden splat entity. The
  // colour pass renders only splats, preserving the depth written above.
  scene_manager_->clearSpecialCaseRenderQueues();
  if (is_splat_pass) {
    scene_manager_->setSpecialCaseRenderQueueMode(Ogre::SceneManager::SCRQM_INCLUDE);
    scene_manager_->addSpecialCaseRenderQueue(kSplatRenderQueue);
  } else {
    scene_manager_->setSpecialCaseRenderQueueMode(Ogre::SceneManager::SCRQM_EXCLUDE);
  }
}

void GaussianSplattingDisplay::postRenderTargetUpdate(const Ogre::RenderTargetEvent &)
{
  // Neither belongs in the selection and pick textures, which RViz renders
  // without notifying this listener, so leave both hidden between targets.
  setSplatsVisible(false);
  if (composite_rect_) {
    composite_rect_->setVisible(false);
  }

  // Back to rendering every queue for whatever target comes next.
  scene_manager_->clearSpecialCaseRenderQueues();
  scene_manager_->setSpecialCaseRenderQueueMode(Ogre::SceneManager::SCRQM_EXCLUDE);
}

void GaussianSplattingDisplay::destroyRenderTarget()
{
  if (splat_texture_) {
    Ogre::RenderTexture * target = splat_texture_->getBuffer()->getRenderTarget();
    target->removeListener(this);
    target->removeAllViewports();
    Ogre::TextureManager::getSingleton().remove(splat_texture_);
    splat_texture_.reset();
  }

  if (main_target_) {
    main_target_->removeListener(this);
    main_target_ = nullptr;
  }

  rtt_viewport_ = nullptr;
  depth_viewport_ = nullptr;
  rtt_width_ = 0;
  rtt_height_ = 0;

  // Back to drawing straight into the scene.
  if (composite_rect_) {
    composite_rect_->setVisible(false);
  }
  if (renderable_) {
    renderable_->setVisible(true);
  }
}

void GaussianSplattingDisplay::updateRenderTarget()
{
  const float scale = render_scale_property_->getFloat();
  Ogre::Viewport * main_viewport = mainViewport();

  if (!offscreen_property_->getBool() || !main_viewport) {
    if (splat_texture_) {
      destroyRenderTarget();
    }
    return;
  }

  const auto width = static_cast<unsigned int>(
    std::max(1.0f, static_cast<float>(main_viewport->getActualWidth()) * scale));
  const auto height = static_cast<unsigned int>(
    std::max(1.0f, static_cast<float>(main_viewport->getActualHeight()) * scale));
  Ogre::Camera * camera = main_viewport->getCamera();
  if (!camera) {
    return;
  }

  // Switching view controller swaps in a different camera, which the offscreen
  // viewport has to follow or it renders from a stale one.
  if (splat_texture_ && width == rtt_width_ && height == rtt_height_ &&
    rtt_viewport_ && rtt_viewport_->getCamera() == camera)
  {
    return;
  }

  destroyRenderTarget();

  if (!composite_material_) {
    Ogre::MaterialPtr base =
      Ogre::MaterialManager::getSingleton().getByName(kCompositeMaterialName, kResourceGroup);
    if (!base) {
      setStatus(
        rviz_common::properties::StatusProperty::Error, "Material",
        QString("Material '%1' was not found; cannot render at reduced scale.")
        .arg(kCompositeMaterialName));
      offscreen_property_->setBool(false);
      return;
    }
    composite_material_ = base->clone(material_name_ + "_Composite", true, kResourceGroup);
  }

  splat_texture_ = Ogre::TextureManager::getSingleton().createManual(
    mesh_name_ + "_RTT", kResourceGroup, Ogre::TEX_TYPE_2D, width, height, 0,
    Ogre::PF_A8R8G8B8, Ogre::TU_RENDERTARGET);

  Ogre::RenderTexture * target = splat_texture_->getBuffer()->getRenderTarget();

  // Two passes over the same target. The first draws the rest of the scene
  // purely to lay down depth; the second clears only the colour, so that depth
  // survives and the splats are occluded by the scene exactly as they are when
  // drawn straight into the window. The first pass's colour is thrown away by
  // that clear, which is the price of not having to override every material in
  // the scene with a depth-only technique.
  depth_viewport_ = target->addViewport(camera, 0);
  depth_viewport_->setClearEveryFrame(true, Ogre::FBT_COLOUR | Ogre::FBT_DEPTH);
  depth_viewport_->setBackgroundColour(Ogre::ColourValue(0.0f, 0.0f, 0.0f, 0.0f));
  depth_viewport_->setOverlaysEnabled(false);
  depth_viewport_->setShadowsEnabled(false);
  depth_viewport_->setMaterialScheme(kDepthMaterialScheme);

  rtt_viewport_ = target->addViewport(camera, 1);
  // Transparent, and the splats write premultiplied alpha, so compositing with
  // "one one_minus_src_alpha" reproduces drawing them straight into the scene.
  // Colour only: the depth left by the pass above is what occludes them.
  rtt_viewport_->setClearEveryFrame(true, Ogre::FBT_COLOUR);
  rtt_viewport_->setBackgroundColour(Ogre::ColourValue(0.0f, 0.0f, 0.0f, 0.0f));
  rtt_viewport_->setOverlaysEnabled(false);
  rtt_viewport_->setShadowsEnabled(false);
  target->addListener(this);
  // MetalRenderTexture's constructor leaves mActive false and never clears it
  // (OgreMetalRenderTexture.mm), so RenderSystem::_updateAllRenderTargets()
  // skips the target and it is never drawn. The GL render textures leave the
  // base class default of true. RViz never hit this because its selection
  // buffers call RenderTarget::update() directly rather than relying on the
  // auto-update pass.
  target->setActive(true);

  main_target_ = main_viewport->getTarget();
  if (main_target_) {
    main_target_->addListener(this);
  }

  rtt_width_ = width;
  rtt_height_ = height;

  if (!composite_rect_) {
    composite_rect_ = new Ogre::Rectangle2D(true);
    composite_rect_->setCorners(-1.0f, 1.0f, 1.0f, -1.0f);
    composite_rect_->setBoundingBox(Ogre::AxisAlignedBox::BOX_INFINITE);
    composite_rect_->setRenderQueueGroup(Ogre::RENDER_QUEUE_OVERLAY);
    composite_node_ = scene_manager_->getRootSceneNode()->createChildSceneNode();
    composite_node_->attachObject(composite_rect_);
  }

  Ogre::Pass * pass = composite_material_->getTechnique(0)->getPass(0);
  pass->getTextureUnitState(0)->setTextureName(splat_texture_->getName());
  composite_material_->load();
  composite_rect_->setMaterial(composite_material_);

  setSplatsVisible(false);
}

}  // namespace gaussian_splatting_rviz_plugins

PLUGINLIB_EXPORT_CLASS(
  gaussian_splatting_rviz_plugins::GaussianSplattingDisplay,
  rviz_common::Display)
