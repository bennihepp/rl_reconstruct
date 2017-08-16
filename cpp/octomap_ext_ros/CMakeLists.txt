cmake_minimum_required(VERSION 2.8)
project(octomap_ext_ros)

include("cmake/CheckC11.cmake")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

find_package(catkin REQUIRED COMPONENTS sensor_msgs tf octomap_msgs)
find_package(octomap_ext REQUIRED)

catkin_package(
  INCLUDE_DIRS include ${OCTOMAP_EXT_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME} ${OCTOMAP_EXT_LIBRARIES}
  DEPENDS sensor_msgs tf octomap_msgs)

include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})
link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

include_directories(${OCTOMAP_EXT_INCLUDE_DIRS})
link_directories(${OCTOMAP_EXT_LIBRARY_DIRS})
link_libraries(${PROJECT_NAME} ${OCTOMAP_EXT_LIBRARIES})

include_directories(include)
include_directories(include ${catkin_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/conversions.cpp)

install(TARGETS ${PROJECT_NAME} 
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

install(DIRECTORY include/octomap_ext_ros/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}/octomap_ext_ros/
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE)
