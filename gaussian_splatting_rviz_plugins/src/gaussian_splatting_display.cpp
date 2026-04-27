#include "gaussian_splatting_rviz_plugins/gaussian_splatting_display.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include <OgreAxisAlignedBox.h>
#include <OgreCamera.h>
#include <OgreEntity.h>
#include <OgreHardwareBufferManager.h>
#include <OgreMeshManager.h>
#include <OgreResourceGroupManager.h>
#include <OgreSceneManager.h>
#include <OgreSceneNode.h>
#include <OgreSubMesh.h>
#include <OgreVertexIndexData.h>

#if __has_include(<GL/gl.h>)
#include <GL/gl.h>
#endif

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

#ifndef GL_PROGRAM_POINT_SIZE
#define GL_PROGRAM_POINT_SIZE 0x8642
#endif

void enableProgrammablePointSize()
{
#if __has_include(<GL/gl.h>)
  glEnable(GL_PROGRAM_POINT_SIZE);
#endif
}

template<typename T>
T clamp(T value, T low, T high)
{
  return std::max(low, std::min(value, high));
}

}  // namespace

GaussianSplattingDisplay::GaussianSplattingDisplay()
{
  mesh_name_ = "GaussianSplattingRvizMesh_" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
}

GaussianSplattingDisplay::~GaussianSplattingDisplay()
{
  clearMesh();

  if (splat_node_) {
    scene_manager_->destroySceneNode(splat_node_);
    splat_node_ = nullptr;
  }
}

void GaussianSplattingDisplay::onInitialize()
{
  rviz_common::MessageFilterDisplay<GaussianSplats>::onInitialize();

  registerOgreResources();
  enableProgrammablePointSize();
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
  enableProgrammablePointSize();
  sortIndexBuffer();
}

void GaussianSplattingDisplay::processMessage(GaussianSplats::ConstSharedPtr msg)
{
  if (!msg) {
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

  rebuildMesh(*msg);
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

  rgm.addResourceLocation(media_dir + "/materials/scripts", "FileSystem", kResourceGroup);
  rgm.addResourceLocation(media_dir + "/materials/programs/GLSL", "FileSystem", kResourceGroup);
  rgm.initialiseResourceGroup(kResourceGroup);

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

  index_buffer_.reset();
  positions_.clear();
  indices_.clear();
  last_camera_position_ = Ogre::Vector3::ZERO;
  last_camera_direction_ = Ogre::Vector3::ZERO;
}

void GaussianSplattingDisplay::rebuildMesh(const GaussianSplats & msg)
{
  clearMesh();

  const std::size_t count = msg.splats.size();
  if (count == 0) {
    return;
  }

  positions_.resize(count);
  indices_.resize(count);
  std::iota(indices_.begin(), indices_.end(), 0);

  std::vector<float> position_data(count * 3);
  std::vector<float> colour_data(count * 4);
  std::vector<float> cov_diag_data(count * 3);
  std::vector<float> cov_upper_data(count * 3);

  Ogre::AxisAlignedBox bounds;
  bounds.setNull();
  Ogre::Real radius_sq = 0.0f;

  for (std::size_t i = 0; i < count; ++i) {
    const auto & splat = msg.splats[i];

    const Ogre::Vector3 position(
      static_cast<Ogre::Real>(splat.position.x),
      static_cast<Ogre::Real>(splat.position.y),
      static_cast<Ogre::Real>(splat.position.z));
    positions_[i] = position;
    bounds.merge(position);
    radius_sq = std::max(radius_sq, position.squaredLength());

    position_data[i * 3 + 0] = static_cast<float>(splat.position.x);
    position_data[i * 3 + 1] = static_cast<float>(splat.position.y);
    position_data[i * 3 + 2] = static_cast<float>(splat.position.z);

    colour_data[i * 4 + 0] = clamp(splat.color.r, 0.0f, 1.0f);
    colour_data[i * 4 + 1] = clamp(splat.color.g, 0.0f, 1.0f);
    colour_data[i * 4 + 2] = clamp(splat.color.b, 0.0f, 1.0f);
    colour_data[i * 4 + 3] = clamp(splat.color.a, 0.0f, 1.0f);

    cov_diag_data[i * 3 + 0] = splat.cov_xx;
    cov_diag_data[i * 3 + 1] = splat.cov_yy;
    cov_diag_data[i * 3 + 2] = splat.cov_zz;

    cov_upper_data[i * 3 + 0] = splat.cov_xy;
    cov_upper_data[i * 3 + 1] = splat.cov_xz;
    cov_upper_data[i * 3 + 2] = splat.cov_yz;
  }

  mesh_ = Ogre::MeshManager::getSingleton().createManual(mesh_name_, kResourceGroup);
  Ogre::SubMesh * submesh = mesh_->createSubMesh();
  submesh->useSharedVertices = false;
  submesh->operationType = Ogre::RenderOperation::OT_POINT_LIST;
  submesh->setMaterialName(kMaterialName, kResourceGroup);
  submesh->vertexData = OGRE_NEW Ogre::VertexData();
  submesh->vertexData->vertexCount = count;

  Ogre::VertexDeclaration * declaration = submesh->vertexData->vertexDeclaration;
  declaration->addElement(0, 0, Ogre::VET_FLOAT3, Ogre::VES_POSITION);
  declaration->addElement(1, 0, Ogre::VET_FLOAT4, Ogre::VES_DIFFUSE);
  declaration->addElement(2, 0, Ogre::VET_FLOAT3, Ogre::VES_TEXTURE_COORDINATES, 0);
  declaration->addElement(3, 0, Ogre::VET_FLOAT3, Ogre::VES_TEXTURE_COORDINATES, 1);

  auto & hbm = Ogre::HardwareBufferManager::getSingleton();
  const Ogre::HardwareBuffer::Usage vertex_usage = Ogre::HardwareBuffer::HBU_STATIC_WRITE_ONLY;

  auto write_vertex_buffer =
    [&](unsigned short source, const std::vector<float> & data, std::size_t components) {
      Ogre::HardwareVertexBufferSharedPtr buffer = hbm.createVertexBuffer(
        sizeof(float) * components, count, vertex_usage);
      buffer->writeData(0, buffer->getSizeInBytes(), data.data(), true);
      submesh->vertexData->vertexBufferBinding->setBinding(source, buffer);
    };

  write_vertex_buffer(0, position_data, 3);
  write_vertex_buffer(1, colour_data, 4);
  write_vertex_buffer(2, cov_diag_data, 3);
  write_vertex_buffer(3, cov_upper_data, 3);

  index_buffer_ = hbm.createIndexBuffer(
    Ogre::HardwareIndexBuffer::IT_32BIT,
    count,
    Ogre::HardwareBuffer::HBU_DYNAMIC_WRITE_ONLY);
  index_buffer_->writeData(0, index_buffer_->getSizeInBytes(), indices_.data(), true);
  submesh->indexData->indexCount = count;
  submesh->indexData->indexBuffer = index_buffer_;

  mesh_->_setBounds(bounds);
  mesh_->_setBoundingSphereRadius(std::sqrt(radius_sq));
  mesh_->load();

  entity_ = scene_manager_->createEntity(mesh_name_);
  entity_->setMaterialName(kMaterialName, kResourceGroup);
  splat_node_->attachObject(entity_);

  sortIndexBuffer();
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

  index_buffer_->writeData(0, index_buffer_->getSizeInBytes(), indices_.data(), true);
}

}  // namespace gaussian_splatting_rviz_plugins

PLUGINLIB_EXPORT_CLASS(
  gaussian_splatting_rviz_plugins::GaussianSplattingDisplay,
  rviz_common::Display)
