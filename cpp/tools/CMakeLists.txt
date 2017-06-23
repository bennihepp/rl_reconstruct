# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required(VERSION 2.8.11)
project(rl_reconstruct_tools)

include("cmake/CheckC11.cmake")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# Enable local cmake macros
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

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

# Add PCL dependency
find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add octomap dependency
find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})
link_directories(${OCTOMAP_LIBRARY_DIRS})

add_subdirectory(surface_voxel_extractor)
