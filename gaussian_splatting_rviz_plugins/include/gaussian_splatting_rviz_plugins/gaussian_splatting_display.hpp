#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <OgreHardwareIndexBuffer.h>
#include <OgreMesh.h>
#include <OgrePrerequisites.h>
#include <OgreVector.h>

#include "gaussian_splatting_msgs/msg/gaussian_splats.hpp"
#include "rviz_common/message_filter_display.hpp"

namespace gaussian_splatting_rviz_plugins
{

class GaussianSplattingDisplay
  : public rviz_common::MessageFilterDisplay<gaussian_splatting_msgs::msg::GaussianSplats>
{
public:
  GaussianSplattingDisplay();
  ~GaussianSplattingDisplay() override;

  void onInitialize() override;
  void reset() override;
  void update(float wall_dt, float ros_dt) override;

private:
  using GaussianSplats = gaussian_splatting_msgs::msg::GaussianSplats;

  void processMessage(GaussianSplats::ConstSharedPtr msg) override;
  void registerOgreResources();
  void clearMesh();
  void rebuildMesh(const GaussianSplats & msg);
  void sortIndexBuffer();

  Ogre::SceneNode * splat_node_ = nullptr;
  Ogre::Entity * entity_ = nullptr;
  Ogre::MeshPtr mesh_;
  Ogre::HardwareIndexBufferSharedPtr index_buffer_;
  std::vector<Ogre::Vector3> positions_;
  std::vector<std::uint32_t> indices_;
  Ogre::Vector3 last_camera_position_ = Ogre::Vector3::ZERO;
  Ogre::Vector3 last_camera_direction_ = Ogre::Vector3::ZERO;
  std::string mesh_name_;
  bool resources_registered_ = false;
};

}  // namespace gaussian_splatting_rviz_plugins
