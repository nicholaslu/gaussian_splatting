#include "gaussian_splatting_rviz_plugins/gaussian_splatting_display.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <map>
#include <numeric>
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
#include "pluginlib/class_list_macros.hpp"
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

// Degree 0 spherical harmonics basis, as used by the 3DGS reference
// implementation: rgb = sh_dc * kSHC0 + 0.5
constexpr float kSHC0 = 0.28209479177387814f;

// Each splat is expanded from one instance record and a shared octagon. The
// octagon circumscribes the unit circle, so it preserves every fragment that
// the Gaussian shader can keep while removing 17.2% of the bounding square's
// corner area. RViz's patched Ogre Metal render system maps instance buffers
// to MTLVertexStepFunctionPerInstance; OpenGL uses the same Ogre declaration.
constexpr std::size_t kVerticesPerSplat = 8;
constexpr std::size_t kIndicesPerSplat = 18;
constexpr float kOctagonTangent = 0.41421356237f;
constexpr float kCorners[kVerticesPerSplat][2] = {
  {1.0f, kOctagonTangent},
  {kOctagonTangent, 1.0f},
  {-kOctagonTangent, 1.0f},
  {-1.0f, kOctagonTangent},
  {-1.0f, -kOctagonTangent},
  {-kOctagonTangent, -1.0f},
  {kOctagonTangent, -1.0f},
  {1.0f, -kOctagonTangent},
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
    "Offscreen Rendering", false,
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

  automatic_render_scale_property_ = new rviz_common::properties::BoolProperty(
    "Automatic Render Scale", false,
    "Adjust the splat render target to hold the requested frame rate. Render "
    "Scale remains the upper quality limit; other RViz displays stay at native resolution.",
    offscreen_property_);

  minimum_render_scale_property_ = new rviz_common::properties::FloatProperty(
    "Minimum Render Scale", 0.35f,
    "Lowest scale automatic control may select. Lower values recover more "
    "frame rate at the cost of softer splats.",
    automatic_render_scale_property_);
  minimum_render_scale_property_->setMin(0.25f);
  minimum_render_scale_property_->setMax(1.0f);

  target_frame_rate_property_ = new rviz_common::properties::FloatProperty(
    "Target Frame Rate", 55.0f,
    "Frame-rate target used by automatic render scaling. A small margin below "
    "RViz's 60 Hz cap leaves room for transient work.",
    automatic_render_scale_property_);
  target_frame_rate_property_->setMin(10.0f);
  target_frame_rate_property_->setMax(240.0f);
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
  updateAutomaticRenderScale(wall_dt);
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
    setStatus(
      rviz_common::properties::StatusProperty::Ok, "Visible splats",
      QString("%1 / %2 instances").arg(visible_splat_count_).arg(splat_count_));
    setStatus(
      rviz_common::properties::StatusProperty::Ok, "CPU preparation",
      QString("upload %1 ms, cull/sort %2 ms")
      .arg(last_upload_ms_, 0, 'f', 2)
      .arg(last_sort_ms_, 0, 'f', 2));
  } else {
    deleteStatus("Visible splats");
    deleteStatus("CPU preparation");
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
  indices_.clear();
  instances_.clear();
  sorted_instances_.clear();
  last_camera_position_ = Ogre::Vector3::ZERO;
  last_camera_direction_ = Ogre::Vector3::ZERO;
  last_viewport_width_ = 0;
  last_viewport_height_ = 0;
  last_sigma_radius_ = -1.0f;
  last_min_screen_radius_ = -1.0f;
  last_culling_enabled_ = false;
  last_upload_ms_ = 0.0;
  last_sort_ms_ = 0.0;
  sort_dirty_ = true;
}

void GaussianSplattingDisplay::allocateMesh(std::size_t count)
{
  clearMesh();

  splat_count_ = count;

  positions_.resize(count);
  indices_.resize(count);
  std::iota(indices_.begin(), indices_.end(), 0);
  instances_.resize(count);
  sorted_instances_.resize(count);

  auto * vertex_data = OGRE_NEW Ogre::VertexData();
  vertex_data->vertexCount = kVerticesPerSplat;

  Ogre::VertexDeclaration * declaration = vertex_data->vertexDeclaration;
  declaration->addElement(
    0, offsetof(SplatInstance, position), Ogre::VET_FLOAT3, Ogre::VES_POSITION);
  declaration->addElement(
    0, offsetof(SplatInstance, colour), Ogre::VET_FLOAT4, Ogre::VES_DIFFUSE);
  declaration->addElement(
    0, offsetof(SplatInstance, scale), Ogre::VET_FLOAT3, Ogre::VES_TEXTURE_COORDINATES, 0);
  declaration->addElement(
    0, offsetof(SplatInstance, quat), Ogre::VET_FLOAT4, Ogre::VES_TEXTURE_COORDINATES, 1);
  declaration->addElement(
    1, 0, Ogre::VET_FLOAT2, Ogre::VES_TEXTURE_COORDINATES, 2);

  auto & hbm = Ogre::HardwareBufferManager::getSingleton();
  const auto dynamic = Ogre::HardwareBuffer::HBU_DYNAMIC_WRITE_ONLY;
  const auto stat1c = Ogre::HardwareBuffer::HBU_STATIC_WRITE_ONLY;

  instance_buffer_ = hbm.createVertexBuffer(sizeof(SplatInstance), count, dynamic);
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
  const std::uint16_t octagon_indices[kIndicesPerSplat] = {
    0, 1, 2,
    0, 2, 3,
    0, 3, 4,
    0, 4, 5,
    0, 5, 6,
    0, 6, 7,
  };
  index_data->indexBuffer->writeData(
    0, index_data->indexBuffer->getSizeInBytes(), octagon_indices, true);

  renderable_ = std::make_unique<GaussianSplatRenderable>(mesh_name_);
  renderable_->setGeometry(vertex_data, index_data, count);
  renderable_->setMaterial(material_);
  renderable_->setBoundingBox(Ogre::AxisAlignedBox::BOX_INFINITE);
  renderable_->setVisible(!splat_texture_);
  renderable_->setRenderQueueGroup(kSplatRenderQueue);
  splat_node_->attachObject(renderable_.get());
}

void GaussianSplattingDisplay::uploadSplats(const GaussianSplats & msg, std::size_t count)
{
  Ogre::AxisAlignedBox bounds;
  bounds.setNull();

  for (std::size_t i = 0; i < count; ++i) {
    const Ogre::Vector3 position(msg.means[i * 3], msg.means[i * 3 + 1], msg.means[i * 3 + 2]);
    positions_[i] = position;
    bounds.merge(position);

    // Degree 0 spherical harmonics only. The higher order coefficients are
    // carried by the message but evaluating them needs the view direction per
    // frame, which does not fit in a vertex attribute.
    const float r = clamp(msg.sh_dc[i * 3] * kSHC0 + 0.5f, 0.0f, 1.0f);
    const float g = clamp(msg.sh_dc[i * 3 + 1] * kSHC0 + 0.5f, 0.0f, 1.0f);
    const float b = clamp(msg.sh_dc[i * 3 + 2] * kSHC0 + 0.5f, 0.0f, 1.0f);
    const float a = clamp(msg.opacities[i], 0.0f, 1.0f);

    SplatInstance & instance = instances_[i];
    std::copy_n(&msg.means[i * 3], 3, instance.position);
    instance.colour[0] = r;
    instance.colour[1] = g;
    instance.colour[2] = b;
    instance.colour[3] = a;
    std::copy_n(&msg.scales[i * 3], 3, instance.scale);
    std::copy_n(&msg.quats[i * 4], 4, instance.quat);
  }

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
}

void GaussianSplattingDisplay::writeSortedInstances()
{
  if (!instance_buffer_ || !renderable_) {
    return;
  }

  for (std::size_t output = 0; output < indices_.size(); ++output) {
    sorted_instances_[output] = instances_[indices_[output]];
  }

  visible_splat_count_ = indices_.size();
  renderable_->setInstanceCount(visible_splat_count_);
  if (visible_splat_count_ > 0) {
    instance_buffer_->writeData(
      0, visible_splat_count_ * sizeof(SplatInstance), sorted_instances_.data(), true);
  }
}

void GaussianSplattingDisplay::sortIndexBuffer()
{
  if (!instance_buffer_ || positions_.empty() || !context_ || !context_->getViewManager()) {
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

  indices_.clear();
  indices_.reserve(splat_count_);

  const Ogre::Matrix4 world_transform = splat_node_->_getFullTransform();
  const Ogre::Vector3 derived_scale = splat_node_->_getDerivedScale();
  const float world_scale = std::max(
    std::abs(derived_scale.x),
    std::max(std::abs(derived_scale.y), std::abs(derived_scale.z)));
  const Ogre::Matrix4 view_matrix = camera->getViewMatrix(true);
  const float alpha_cutoff = 1.0f / 255.0f;
  const float focal_y = viewport_height > 0 && camera->getProjectionType() == Ogre::PT_PERSPECTIVE ?
    static_cast<float>(viewport_height) * 0.5f /
    std::tan(static_cast<float>(camera->getFOVy().valueRadians()) * 0.5f) : 0.0f;

  for (std::uint32_t i = 0; i < splat_count_; ++i) {
    const SplatInstance & instance = instances_[i];
    if (instance.colour[3] < alpha_cutoff) {
      continue;
    }

    const float visible_radius = std::min(
      sigma_radius,
      std::sqrt(2.0f * std::log(instance.colour[3] / alpha_cutoff)));
    const float max_sigma = std::max(
      instance.scale[0], std::max(instance.scale[1], instance.scale[2]));
    const float world_radius = visible_radius * max_sigma * world_scale;
    const Ogre::Vector3 world_position = world_transform * positions_[i];

    if (culling_enabled && !camera->isVisible(Ogre::Sphere(world_position, world_radius))) {
      continue;
    }

    if (culling_enabled && min_screen_radius > 0.0f && viewport_height > 0) {
      float screen_radius = 0.0f;
      if (camera->getProjectionType() == Ogre::PT_PERSPECTIVE) {
        const Ogre::Vector3 view_position = view_matrix * world_position;
        const float depth = -view_position.z;
        if (depth <= 0.0f) {
          continue;
        }
        screen_radius = focal_y * world_radius /
          std::max(depth - world_radius, camera->getNearClipDistance());
      } else {
        const Ogre::Matrix4 projection = camera->getProjectionMatrix();
        screen_radius =
          std::abs(static_cast<float>(projection[1][1])) * viewport_height * 0.5f * world_radius;
      }

      // eps2d adds this minimum variance in screen space before rasterisation.
      const float blur_radius = visible_radius * std::sqrt(std::max(eps2d_, 0.0f));
      screen_radius = std::sqrt(screen_radius * screen_radius + blur_radius * blur_radius);
      if (screen_radius < min_screen_radius) {
        continue;
      }
    }

    indices_.push_back(i);
  }

  const Ogre::Vector3 local_sort_direction =
    splat_node_->_getDerivedOrientation().Inverse() * (-camera_direction);

  std::sort(indices_.begin(), indices_.end(), [&](std::uint32_t a, std::uint32_t b) {
      return positions_[a].dotProduct(local_sort_direction) <
             positions_[b].dotProduct(local_sort_direction);
    });

  writeSortedInstances();
  last_sort_ms_ = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - sort_start).count();
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

void GaussianSplattingDisplay::updateAutomaticRenderScale(float wall_dt)
{
  const float maximum_scale = render_scale_property_->getFloat();
  const bool enabled = offscreen_property_->getBool() &&
    automatic_render_scale_property_->getBool();
  if (!enabled) {
    effective_render_scale_ = maximum_scale;
    automatic_scale_elapsed_ = 0.0f;
    automatic_scale_was_enabled_ = false;
    deleteStatus("Automatic scale");
    return;
  }

  float minimum_scale = minimum_render_scale_property_->getFloat();
  minimum_scale = std::min(minimum_scale, maximum_scale);
  if (!automatic_scale_was_enabled_) {
    effective_render_scale_ = maximum_scale;
    automatic_scale_elapsed_ = 0.0f;
    automatic_scale_was_enabled_ = true;
  }
  effective_render_scale_ = clamp(effective_render_scale_, minimum_scale, maximum_scale);

  // Ogre updates lastFPS once per second. Sampling more frequently would act
  // repeatedly on the same value and recreate the RTT unnecessarily.
  automatic_scale_elapsed_ += clamp(wall_dt, 0.0f, 0.25f);
  if (automatic_scale_elapsed_ < 1.0f) {
    return;
  }
  automatic_scale_elapsed_ = 0.0f;

  Ogre::Viewport * viewport = mainViewport();
  if (!viewport || !viewport->getTarget()) {
    return;
  }
  const float fps = viewport->getTarget()->getStatistics().lastFPS;
  const float target_fps = target_frame_rate_property_->getFloat();
  if (fps > 1.0f && target_fps > 1.0f) {
    float next_scale = effective_render_scale_;
    if (fps < target_fps * 0.95f && effective_render_scale_ > minimum_scale) {
      // When splat fill dominates, FPS is approximately inverse-square in the
      // render scale. Leave another 3% of headroom after the predicted step.
      next_scale *= std::sqrt(fps / target_fps) * 0.97f;
      next_scale = std::floor(next_scale * 20.0f) / 20.0f;
    } else if (fps > target_fps * 1.08f && effective_render_scale_ < maximum_scale) {
      // Recovery is intentionally slower than degradation to avoid oscillation.
      next_scale += 0.05f;
    }
    effective_render_scale_ = clamp(next_scale, minimum_scale, maximum_scale);
  }

  setStatus(
    rviz_common::properties::StatusProperty::Ok, "Automatic scale",
    QString("%1x at %2 fps (target %3)")
    .arg(effective_render_scale_, 0, 'f', 2)
    .arg(fps, 0, 'f', 1)
    .arg(target_fps, 0, 'f', 0));
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
  const float scale = effective_render_scale_;
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
