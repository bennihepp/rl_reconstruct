cmake_minimum_required(VERSION 2.8)
project(octomap_server_ext)

set(PACKAGE_DEPENDENCIES
  rospy
  roscpp
  visualization_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  nav_msgs
  std_msgs
  std_srvs
  octomap_ros
  octomap_msgs
  dynamic_reconfigure
  nodelet
  message_generation
)


find_package(catkin REQUIRED COMPONENTS ${PACKAGE_DEPENDENCIES})

find_package(PCL REQUIRED QUIET COMPONENTS common sample_consensus io segmentation filters)

find_package(octomap REQUIRED)
add_definitions(-DOCTOMAP_NODEBUGOUT)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

include("cmake/CheckC11.cmake")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# Add Eigen dependency
find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIR})

# Add boost dependency
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.56 COMPONENTS program_options REQUIRED)
include_directories(${BOOST_INCLUDE_DIR})
link_directories(${BOOST_LIBRARY_DIRS})
add_definitions(-DBOOST_ALL_DYN_LINK=1)

# Add mLib dependency
set(MLIB_ROOT "$ENV{HOME}/src/mLib")
set(MLIB_INCLUDE_DIR "${MLIB_ROOT}/include")
set(MLIB_SOURCE_DIR "${MLIB_ROOT}/src")
include_directories(${MLIB_INCLUDE_DIR} ${MLIB_SOURCE_DIR})

# Add libbh dependency
set(LIBBH_ROOT "$ENV{HOME}/src/libbh")
set(LIBBH_INCLUDE_DIR "${LIBBH_ROOT}/include")
set(LIBBH_SOURCE_DIR "${LIBBH_ROOT}/src")
set(LIBBH_SOURCES "${LIBBH_SOURCE_DIR}/mLib/mLib.cpp")
include_directories(${LIBBH_INCLUDE_DIR})
set(LIBBH_LIBRARIES "${LIBBH_ROOT}/libbh.so")

## Declare ROS messages and services
add_message_files(
  FILES
  Ray.msg
)

## Declare ROS messages and services
add_service_files(
  FILES
  InsertPointCloud.srv
  Raycast.srv
  RaycastCamera.srv
  QueryVoxels.srv
  ClearBoundingBox.srv
  OverrideBoundingBox.srv
)

## Generate added messages and services
generate_messages(
  DEPENDENCIES
  std_msgs sensor_msgs
)

generate_dynamic_reconfigure_options(cfg/OctomapServerExt.cfg)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS ${PACKAGE_DEPENDENCIES}
  DEPENDS octomap PCL
)

set(LINK_LIBS
  ${OCTOMAP_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_library(${PROJECT_NAME} src/OctomapServerExt.cpp)
target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

add_executable(octomap_server_ext_node src/octomap_server_ext_node.cpp)
target_link_libraries(octomap_server_ext_node ${PROJECT_NAME} ${LINK_LIBS})
add_dependencies(octomap_server_ext_node ${PROJECT_NAME}_gencfg)

#add_executable(octomap_color_server_node src/octomap_color_server_node.cpp)
#target_link_libraries(octomap_color_server_node ${PROJECT_NAME} ${LINK_LIBS})

add_executable(octomap_server_ext_static src/octomap_server_ext_static.cpp)
target_link_libraries(octomap_server_ext_static ${PROJECT_NAME} ${LINK_LIBS})

add_executable(octomap_saver src/octomap_saver.cpp ${LIBBH_SOURCES})
target_link_libraries(octomap_saver ${PROJECT_NAME} ${LINK_LIBS})

# Nodelet
add_library(octomap_server_ext_nodelet src/octomap_server_ext_nodelet.cpp ${LIBBH_SOURCES})
target_link_libraries(octomap_server_ext_nodelet ${PROJECT_NAME} ${LINK_LIBS})

# install targets:
install(TARGETS ${PROJECT_NAME}
  octomap_server_ext_node
  octomap_server_ext_static
  octomap_saver
  octomap_server_ext_nodelet
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
