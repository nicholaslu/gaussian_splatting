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

#ifndef GAUSSIAN_SPLATTING_RVIZ_PLUGINS__STATIC_PLUGINS_HPP_
#define GAUSSIAN_SPLATTING_RVIZ_PLUGINS__STATIC_PLUGINS_HPP_

namespace gaussian_splatting_rviz_plugins
{

/// Make this display available to a statically linked application.
/**
 * Called once, early, by an application that links this library instead of
 * loading it - iPadOS, where there is nothing to dlopen. A plain call rather
 * than a global object: a linker drops an archive member nothing refers to,
 * and a registration in a constructor nobody calls is one that silently does
 * not happen. rviz_default_plugins registers its own displays the same way.
 *
 * Registering twice is harmless.
 */
void registerStaticPlugins();

}  // namespace gaussian_splatting_rviz_plugins

#endif  // GAUSSIAN_SPLATTING_RVIZ_PLUGINS__STATIC_PLUGINS_HPP_
