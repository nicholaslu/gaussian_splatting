cmake_minimum_required(VERSION 3.8)
project(gaussian_splatting_rviz_plugins)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(rviz_ogre_vendor REQUIRED)
find_package(pluginlib QUIET)
find_package(ament_index_cpp REQUIRED)
find_package(gaussian_splatting_msgs REQUIRED)

# iOS links everything into the signed application; elsewhere this stays a
# plug-in that RViz loads.
if(DEFINED BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS)
  set(gaussian_splatting_library_type STATIC)
else()
  set(gaussian_splatting_library_type SHARED)
endif()

add_library(${PROJECT_NAME} ${gaussian_splatting_library_type}
  src/gaussian_splatting_display.cpp
  src/static_plugins.cpp
)

# GPU view preparation needs Ogre's Metal render system. Without it the display
# still builds, with CPU view preparation only.
set(GSPLAT_METAL_PREPARATION OFF)
if(APPLE)
  enable_language(OBJCXX)
  get_filename_component(_rviz_prefix "${rviz_ogre_vendor_DIR}/../../.." REALPATH)
  set(_ogre_prefix "${_rviz_prefix}/opt/rviz_ogre_vendor")
  find_path(OGRE_METAL_INCLUDE_DIR OgreMetalRenderSystem.h
    HINTS "${_ogre_prefix}/include/OGRE/RenderSystems/Metal")
  # Ogre plug-ins intentionally omit the usual lib prefix, so find_library()
  # does not recognise this dylib even though it is linkable.
  unset(OGRE_METAL_LIBRARY CACHE)
  find_file(OGRE_METAL_LIBRARY RenderSystem_Metal.dylib
    HINTS "${_ogre_prefix}/lib/OGRE" NO_DEFAULT_PATH)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  # A static build is linked into an application that links the Metal render
  # system itself, so only the shared plug-in needs the dylib.
  if(gaussian_splatting_library_type STREQUAL "STATIC" OR OGRE_METAL_LIBRARY)
    set(_ogre_metal_linkable TRUE)
  else()
    set(_ogre_metal_linkable FALSE)
  endif()
  if(OGRE_METAL_INCLUDE_DIR AND _ogre_metal_linkable AND METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    set(GSPLAT_METAL_PREPARATION ON)
  else()
    message(WARNING
      "Ogre's Metal render system or Apple's Metal framework was not found; "
      "building with CPU view preparation only")
  endif()
endif()

if(GSPLAT_METAL_PREPARATION)
  set(gaussian_splatting_metal_sources
    src/metal_view_preparation.mm
    src/metal_view_preparation_core.mm)
  target_sources(${PROJECT_NAME} PRIVATE ${gaussian_splatting_metal_sources})
  # These hold Objective-C objects in C++ members and rely on ARC to release
  # them; compiled without it, every buffer a scene allocates is leaked.
  set_source_files_properties(${gaussian_splatting_metal_sources}
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_include_directories(${PROJECT_NAME} PRIVATE "${OGRE_METAL_INCLUDE_DIR}")
  target_link_libraries(${PROJECT_NAME} "${METAL_FRAMEWORK}" "${FOUNDATION_FRAMEWORK}")
  if(NOT gaussian_splatting_library_type STREQUAL "STATIC")
    # RViz can still select OpenGL on macOS. A normal dependency would make the
    # class loader require RenderSystem_Metal even in that process; weak linking
    # leaves the CPU/OpenGL fallback independently loadable.
    target_link_libraries(${PROJECT_NAME} "-Wl,-weak_library,${OGRE_METAL_LIBRARY}")
  endif()
  target_compile_definitions(${PROJECT_NAME} PRIVATE GSPLAT_HAS_METAL_PREPARATION=1)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rviz_common
  rviz_rendering
  rviz_ogre_vendor
  ament_index_cpp
  gaussian_splatting_msgs
)

if(pluginlib_FOUND)
  ament_target_dependencies(${PROJECT_NAME} pluginlib)
endif()

# A static build has no class loader to declare anything to, and pluginlib is
# not built for it either.
if(NOT gaussian_splatting_library_type STREQUAL "STATIC")
  pluginlib_export_plugin_description_file(rviz_common plugins_description.xml)
endif()

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY ogre_media
  DESTINATION share/${PROJECT_NAME}
)

# The notice travels with the shader it covers.
install(FILES plugins_description.xml THIRD_PARTY_NOTICES.md
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
# So an application can link this rather than load it, which is what iOS does.
ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  rviz_common
  rviz_rendering
  rviz_ogre_vendor
  pluginlib
  ament_index_cpp
  gaussian_splatting_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  if(GSPLAT_METAL_PREPARATION AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    # Runs the GPU kernels and the CPU path on identical synthetic views and
    # requires exact agreement; see test/verify_gpu_preparation.mm.
    add_executable(verify_gpu_preparation
      test/verify_gpu_preparation.mm
      src/metal_view_preparation_core.mm)
    set_source_files_properties(test/verify_gpu_preparation.mm
      PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_include_directories(verify_gpu_preparation PRIVATE include src)
    target_compile_features(verify_gpu_preparation PRIVATE cxx_std_17)
    target_link_libraries(verify_gpu_preparation "${METAL_FRAMEWORK}" "${FOUNDATION_FRAMEWORK}")
    add_test(NAME verify_gpu_preparation
      COMMAND verify_gpu_preparation --quick
        "${CMAKE_CURRENT_SOURCE_DIR}/ogre_media/materials/programs/Metal/gsplat_prepare.metal")
    # A machine with no Metal device skips rather than fails.
    set_tests_properties(verify_gpu_preparation PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()

ament_package()
