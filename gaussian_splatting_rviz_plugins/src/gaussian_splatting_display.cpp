#include "gaussian_splatting_rviz_plugins/gaussian_splatting_display.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include <OgreAxisAlignedBox.h>
#include <OgreCamera.h>
#include <OgreEntity.h>
#include <OgreGpuProgramParams.h>
#include <OgreHardwareBufferManager.h>
#include <OgreMaterialManager.h>
#include <OgreMeshManager.h>
#include <OgrePass.h>
#include <OgreResourceGroupManager.h>
#include <OgreRoot.h>
#include <OgreSceneManager.h>
#include <OgreSceneNode.h>
#include <OgreSubMesh.h>
#include <OgreTechnique.h>
#include <OgreVertexIndexData.h>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rviz_common/display_context.hpp"
#include "rviz_common/frame_manager_iface.hpp"
#include "rviz_common/view_controller.hpp"
#include "rviz_common/view_manager.hpp"

namespace gaussian_splatting_rviz_plugins
{
namespace
{

constexpr const char * kResourceGroup = "GaussianSplattingRviz";
constexpr const char * kMaterialName = "GaussianSplatting/RViz";

// Degree 0 spherical harmonics basis, as used by the 3DGS reference
// implementation: rgb = sh_dc * kSHC0 + 0.5
constexpr float kSHC0 = 0.28209479177387814f;

// Each splat is expanded into a screen-aligned quad. Ogre's Metal render
// system advertises RSC_VERTEX_BUFFER_INSTANCE_DATA but hardcodes
// MTLVertexStepFunctionPerVertex, so hardware instancing cannot be used and
// the per-splat attributes are duplicated across the four corners instead.
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

}  // namespace

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
}

GaussianSplattingDisplay::~GaussianSplattingDisplay()
{
  clearMesh();

  if (material_) {
    Ogre::MaterialManager::getSingleton().remove(material_);
    material_.reset();
  }

  if (splat_node_) {
    scene_manager_->destroySceneNode(splat_node_);
    splat_node_ = nullptr;
  }
}

void GaussianSplattingDisplay::onInitialize()
{
  rviz_common::MessageFilterDisplay<GaussianSplats>::onInitialize();

  registerOgreResources();
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
  applyShaderParams();
  sortIndexBuffer();
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
  uploadSplats(*msg, count);
  writeDrawIndices();
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
  if (entity_) {
    scene_manager_->destroyEntity(entity_);
    entity_ = nullptr;
  }

  if (mesh_) {
    Ogre::MeshManager::getSingleton().remove(mesh_->getHandle());
    mesh_.reset();
  }

  position_buffer_.reset();
  colour_buffer_.reset();
  scale_buffer_.reset();
  quat_buffer_.reset();
  index_buffer_.reset();

  splat_count_ = 0;
  positions_.clear();
  indices_.clear();
  draw_indices_.clear();
  last_camera_position_ = Ogre::Vector3::ZERO;
  last_camera_direction_ = Ogre::Vector3::ZERO;
}

void GaussianSplattingDisplay::allocateMesh(std::size_t count)
{
  clearMesh();

  splat_count_ = count;
  const std::size_t vertex_count = count * kVerticesPerSplat;

  positions_.resize(count);
  indices_.resize(count);
  std::iota(indices_.begin(), indices_.end(), 0);
  draw_indices_.resize(count * kIndicesPerSplat);

  position_data_.resize(vertex_count * 3);
  colour_data_.resize(vertex_count * 4);
  scale_data_.resize(vertex_count * 3);
  quat_data_.resize(vertex_count * 4);

  mesh_ = Ogre::MeshManager::getSingleton().createManual(mesh_name_, kResourceGroup);
  Ogre::SubMesh * submesh = mesh_->createSubMesh();
  submesh->useSharedVertices = false;
  submesh->operationType = Ogre::RenderOperation::OT_TRIANGLE_LIST;
  submesh->setMaterialName(material_name_, kResourceGroup);
  submesh->vertexData = OGRE_NEW Ogre::VertexData();
  submesh->vertexData->vertexCount = vertex_count;

  Ogre::VertexDeclaration * declaration = submesh->vertexData->vertexDeclaration;
  declaration->addElement(0, 0, Ogre::VET_FLOAT3, Ogre::VES_POSITION);
  declaration->addElement(1, 0, Ogre::VET_FLOAT4, Ogre::VES_DIFFUSE);
  declaration->addElement(2, 0, Ogre::VET_FLOAT3, Ogre::VES_TEXTURE_COORDINATES, 0);
  declaration->addElement(3, 0, Ogre::VET_FLOAT4, Ogre::VES_TEXTURE_COORDINATES, 1);
  declaration->addElement(4, 0, Ogre::VET_FLOAT2, Ogre::VES_TEXTURE_COORDINATES, 2);

  auto & hbm = Ogre::HardwareBufferManager::getSingleton();
  const auto dynamic = Ogre::HardwareBuffer::HBU_DYNAMIC_WRITE_ONLY;
  const auto stat1c = Ogre::HardwareBuffer::HBU_STATIC_WRITE_ONLY;

  auto bind =
    [&](unsigned short source, std::size_t components, Ogre::HardwareBuffer::Usage usage) {
      Ogre::HardwareVertexBufferSharedPtr buffer =
        hbm.createVertexBuffer(sizeof(float) * components, vertex_count, usage);
      submesh->vertexData->vertexBufferBinding->setBinding(source, buffer);
      return buffer;
    };

  position_buffer_ = bind(0, 3, dynamic);
  colour_buffer_ = bind(1, 4, dynamic);
  scale_buffer_ = bind(2, 3, dynamic);
  quat_buffer_ = bind(3, 4, dynamic);

  // The corner offsets never change, so this buffer is written once here and
  // left alone for the lifetime of the mesh.
  Ogre::HardwareVertexBufferSharedPtr corner_buffer = bind(4, 2, stat1c);
  std::vector<float> corner_data(vertex_count * 2);
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t corner = 0; corner < kVerticesPerSplat; ++corner) {
      const std::size_t vertex = i * kVerticesPerSplat + corner;
      corner_data[vertex * 2 + 0] = kCorners[corner][0];
      corner_data[vertex * 2 + 1] = kCorners[corner][1];
    }
  }
  corner_buffer->writeData(0, corner_buffer->getSizeInBytes(), corner_data.data(), true);

  index_buffer_ = hbm.createIndexBuffer(
    Ogre::HardwareIndexBuffer::IT_32BIT, count * kIndicesPerSplat, dynamic);
  submesh->indexData->indexCount = count * kIndicesPerSplat;
  submesh->indexData->indexBuffer = index_buffer_;

  // Bounds are refreshed by uploadSplats(); start from something valid so the
  // mesh can be loaded before any data has been written.
  mesh_->_setBounds(Ogre::AxisAlignedBox::BOX_INFINITE);
  mesh_->_setBoundingSphereRadius(std::numeric_limits<Ogre::Real>::max());
  mesh_->load();

  entity_ = scene_manager_->createEntity(mesh_name_);
  entity_->setMaterialName(material_name_, kResourceGroup);
  splat_node_->attachObject(entity_);
}

void GaussianSplattingDisplay::uploadSplats(const GaussianSplats & msg, std::size_t count)
{
  Ogre::AxisAlignedBox bounds;
  bounds.setNull();
  Ogre::Real radius_sq = 0.0f;

  for (std::size_t i = 0; i < count; ++i) {
    const Ogre::Vector3 position(msg.means[i * 3], msg.means[i * 3 + 1], msg.means[i * 3 + 2]);
    positions_[i] = position;
    bounds.merge(position);
    radius_sq = std::max(radius_sq, position.squaredLength());

    // Degree 0 spherical harmonics only. The higher order coefficients are
    // carried by the message but evaluating them needs the view direction per
    // frame, which does not fit in a vertex attribute.
    const float r = clamp(msg.sh_dc[i * 3] * kSHC0 + 0.5f, 0.0f, 1.0f);
    const float g = clamp(msg.sh_dc[i * 3 + 1] * kSHC0 + 0.5f, 0.0f, 1.0f);
    const float b = clamp(msg.sh_dc[i * 3 + 2] * kSHC0 + 0.5f, 0.0f, 1.0f);
    const float a = clamp(msg.opacities[i], 0.0f, 1.0f);

    for (std::size_t corner = 0; corner < kVerticesPerSplat; ++corner) {
      const std::size_t vertex = i * kVerticesPerSplat + corner;

      position_data_[vertex * 3 + 0] = msg.means[i * 3];
      position_data_[vertex * 3 + 1] = msg.means[i * 3 + 1];
      position_data_[vertex * 3 + 2] = msg.means[i * 3 + 2];

      colour_data_[vertex * 4 + 0] = r;
      colour_data_[vertex * 4 + 1] = g;
      colour_data_[vertex * 4 + 2] = b;
      colour_data_[vertex * 4 + 3] = a;

      scale_data_[vertex * 3 + 0] = msg.scales[i * 3];
      scale_data_[vertex * 3 + 1] = msg.scales[i * 3 + 1];
      scale_data_[vertex * 3 + 2] = msg.scales[i * 3 + 2];

      quat_data_[vertex * 4 + 0] = msg.quats[i * 4];
      quat_data_[vertex * 4 + 1] = msg.quats[i * 4 + 1];
      quat_data_[vertex * 4 + 2] = msg.quats[i * 4 + 2];
      quat_data_[vertex * 4 + 3] = msg.quats[i * 4 + 3];
    }
  }

  position_buffer_->writeData(
    0, position_buffer_->getSizeInBytes(), position_data_.data(), true);
  colour_buffer_->writeData(0, colour_buffer_->getSizeInBytes(), colour_data_.data(), true);
  scale_buffer_->writeData(0, scale_buffer_->getSizeInBytes(), scale_data_.data(), true);
  quat_buffer_->writeData(0, quat_buffer_->getSizeInBytes(), quat_data_.data(), true);

  mesh_->_setBounds(bounds);
  mesh_->_setBoundingSphereRadius(std::sqrt(radius_sq));

  // Force a fresh depth sort now that the geometry has changed.
  last_camera_position_ = Ogre::Vector3::ZERO;
  last_camera_direction_ = Ogre::Vector3::ZERO;
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

void GaussianSplattingDisplay::writeDrawIndices()
{
  if (!index_buffer_) {
    return;
  }

  std::size_t output = 0;
  for (const std::uint32_t splat : indices_) {
    const std::uint32_t base = splat * static_cast<std::uint32_t>(kVerticesPerSplat);
    draw_indices_[output++] = base;
    draw_indices_[output++] = base + 1;
    draw_indices_[output++] = base + 2;
    draw_indices_[output++] = base;
    draw_indices_[output++] = base + 2;
    draw_indices_[output++] = base + 3;
  }

  index_buffer_->writeData(0, index_buffer_->getSizeInBytes(), draw_indices_.data(), true);
}

void GaussianSplattingDisplay::sortIndexBuffer()
{
  if (!index_buffer_ || positions_.empty() || !context_ || !context_->getViewManager()) {
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
  if ((camera_position - last_camera_position_).squaredLength() < 0.0001f &&
    (camera_direction - last_camera_direction_).squaredLength() < 0.000001f)
  {
    return;
  }

  last_camera_position_ = camera_position;
  last_camera_direction_ = camera_direction;

  const Ogre::Vector3 local_sort_direction =
    splat_node_->_getDerivedOrientation().Inverse() * (-camera_direction);

  std::sort(indices_.begin(), indices_.end(), [&](std::uint32_t a, std::uint32_t b) {
    return positions_[a].dotProduct(local_sort_direction) <
           positions_[b].dotProduct(local_sort_direction);
  });

  writeDrawIndices();
}

}  // namespace gaussian_splatting_rviz_plugins

PLUGINLIB_EXPORT_CLASS(
  gaussian_splatting_rviz_plugins::GaussianSplattingDisplay,
  rviz_common::Display)
