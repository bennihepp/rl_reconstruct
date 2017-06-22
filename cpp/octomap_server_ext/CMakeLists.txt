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

## Declare ROS messages and services
add_service_files(
  FILES
  InsertPointCloud.srv
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

add_executable(octomap_saver src/octomap_saver.cpp)
target_link_libraries(octomap_saver ${PROJECT_NAME} ${LINK_LIBS})

# Nodelet
add_library(octomap_server_ext_nodelet src/octomap_server_ext_nodelet.cpp)
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
