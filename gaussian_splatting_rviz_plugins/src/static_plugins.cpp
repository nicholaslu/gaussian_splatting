// Copyright 2026 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gaussian_splatting_rviz_plugins/static_plugins.hpp"

#include <QString>
#include <QStringList>

#include "gaussian_splatting_rviz_plugins/gaussian_splatting_display.hpp"
#include "rviz_common/display.hpp"
#include "rviz_common/factory/static_plugin_registry.hpp"

namespace gaussian_splatting_rviz_plugins
{

void registerStaticPlugins()
{
  // The same five things plugins_description.xml says, in the form a linked
  // build can read: the base class, the class id, the package the icon and the
  // manifest belong to, the short name, and what it is.
  rviz_common::registerStaticPlugin<rviz_common::Display, GaussianSplattingDisplay>(
    QStringLiteral("rviz_common::Display"),
    QStringLiteral("gaussian_splatting_rviz_plugins/GaussianSplatting"),
    QStringLiteral("gaussian_splatting_rviz_plugins"),
    QStringLiteral("GaussianSplatting"),
    QStringLiteral("Displays Gaussian splatting."),
    QStringList{QStringLiteral("gaussian_splatting_msgs/msg/GaussianSplats")});
}

}  // namespace gaussian_splatting_rviz_plugins
