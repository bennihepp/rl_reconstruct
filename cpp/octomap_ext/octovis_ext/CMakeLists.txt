CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
PROJECT( octovis_ext )

include(CTest)

include("cmake/CheckC11.cmake")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# # version (e.g. for packaging)
set(OCTOVIS_EXT_MAJOR_VERSION 1)
set(OCTOVIS_EXT_MINOR_VERSION 9)
set(OCTOVIS_EXT_PATCH_VERSION 0)
set(OCTOVIS_EXT_VERSION ${OCTOVIS_EXT_MAJOR_VERSION}.${OCTOVIS_EXT_MINOR_VERSION}.${OCTOVIS_EXT_PATCH_VERSION})
set(OCTOVIS_EXT_SOVERSION ${OCTOVIS_EXT_MAJOR_VERSION}.${OCTOVIS_EXT_MINOR_VERSION})
# get rid of a useless warning:
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
  if(POLICY CMP0042)
    # Enable MACOSX_RPATH by default.
    cmake_policy(SET CMP0042 NEW)
  endif(POLICY CMP0042)
endif(COMMAND cmake_policy)

SET (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules")

# COMPILER SETTINGS (default: Release) and flags
INCLUDE(CompilerSettings)

# Set output directories for libraries and executables
SET( BASE_DIR ${CMAKE_SOURCE_DIR} )
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BASE_DIR}/lib )
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BASE_DIR}/lib )
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BASE_DIR}/bin )
# output dirs for multi-config builds (MSVC)
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
  STRING( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
  SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${BASE_DIR}/lib )
  SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${BASE_DIR}/lib )
  SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${BASE_DIR}/bin )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

# We need the main octomap library to link against (same version as octovis)
# Look at parent directory by default, in case the complete distribution
# (octomap & octovis together) is built.
#
# Otherwise you need to export octomap_DIR to the directory containing
# the file octomap-config.cmake
find_package(octomap_ext ${OCTOVIS_EXT_MAJOR_VERSION}.${OCTOVIS_EXT_MINOR_VERSION} REQUIRED
HINTS ${CMAKE_SOURCE_DIR}/lib/cmake/octomap_ext
${CMAKE_SOURCE_DIR}/../octomap_ext/lib/cmake/octomap_ext
)
MESSAGE(STATUS "Found octomap_ext version: " ${octomap_ext_VERSION})

INCLUDE_DIRECTORIES(BEFORE SYSTEM ${OCTOMAP_EXT_INCLUDE_DIRS})

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE octovis_ext)

set(INSTALL_TARGETS_DEFAULT_ARGS
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Builds the "octovis" viewer based on OpenGL and 
# libQGLViewer, if dependencies available
SET( BUILD_VIEWER 0)

# Look for required libraries:
FIND_PACKAGE(OpenGL)
if(NOT OCTOVIS_EXT_QT5)
  FIND_PACKAGE(Qt4)
endif(NOT OCTOVIS_EXT_QT5)

IF (OpenGL-NOTFOUND OR Qt4-NOTFOUND) 
	MESSAGE ( "OpenGL and QT4 are required for octovis but could not be found.")	
ELSE()
	FIND_PACKAGE(QGLViewer)
	IF(QGLViewer_FOUND)
	  SET( BUILD_VIEWER 1)
	ELSE()
	  MESSAGE ( "\n")
  	MESSAGE ( "libQGLViewer could not be found or generated.")
	ENDIF()
ENDIF()


IF(BUILD_VIEWER)
  MESSAGE(STATUS "\n")
  MESSAGE(STATUS "viewer octovis will be built")
  
  set(INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
  INCLUDE_DIRECTORIES(${INCLUDE_DIRS})
   
  INCLUDE( CMakeLists_src.txt )
    
  # Create an octovis-config.cmake file for the use from the build tree
  set(OCTOVIS_EXT_INCLUDE_DIRS "${INCLUDE_DIRS}")
  set(OCTOVIS_EXT_LIB_DIR     "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  # Set library names as absolute paths
  # Windows, spec. MSVC requires the .lib suffix for imported libs
  IF(WIN32)
    set(OCTOVIS_EXT_LIBRARY
      "${CMAKE_IMPORT_LIBRARY_PREFIX}octovis_ext${CMAKE_IMPORT_LIBRARY_SUFFIX}"
    )
  ELSE()
    set(OCTOVIS_EXT_LIBRARY
      "${CMAKE_SHARED_LIBRARY_PREFIX}octovis_ext${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )
  ENDIF()

  include(CMakePackageConfigHelpers)

  CONFIGURE_PACKAGE_CONFIG_FILE(
    octovis_ext-config.cmake.in
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/octovis_ext/octovis_ext-config.cmake"
    PATH_VARS OCTOVIS_EXT_INCLUDE_DIRS OCTOVIS_EXT_LIB_DIR
    INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/share/octovis_ext)

  WRITE_BASIC_PACKAGE_VERSION_FILE(
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/octovis_ext/octovis_ext-config-version.cmake"
    VERSION ${OCTOVIS_EXT_VERSION}
    COMPATIBILITY AnyNewerVersion)

  # not used right now (export depends?)
  #set(OCTOMAP_CMAKE_DIR "${PROJECT_BINARY_DIR}")
    
  # Create a octovis-config.cmake file for the use from the install tree
  # and install it
  set(OCTOVIS_EXT_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include")
  set(OCTOVIS_EXT_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
  #set(OCTOMAP_CMAKE_DIR "${INSTALL_DATA_DIR}/FooBar/CMake")

  set(OCTOVIS_EXT_INCLUDE_TARGETS
    "include(\${CMAKE_CURRENT_LIST_DIR}/octovis_ext-targets.cmake)")

  CONFIGURE_PACKAGE_CONFIG_FILE(
    octovis_ext-config.cmake.in
    "${PROJECT_BINARY_DIR}/InstallFiles/octovis_ext-config.cmake"
    PATH_VARS OCTOVIS_EXT_INCLUDE_DIRS OCTOVIS_EXT_LIB_DIR
    INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/share/octovis_ext)

  WRITE_BASIC_PACKAGE_VERSION_FILE(
    "${PROJECT_BINARY_DIR}/InstallFiles/octovis_ext-config-version.cmake"
    VERSION ${OCTOVIS_EXT_VERSION}
    COMPATIBILITY AnyNewerVersion)

  install(FILES
    "${PROJECT_BINARY_DIR}/InstallFiles/octovis_ext-config.cmake"
    "${PROJECT_BINARY_DIR}/InstallFiles/octovis_ext-config-version.cmake" 
    DESTINATION share/octovis_ext/)

  # #installation:
  # # store all header files to install:
  file(GLOB octovis_HDRS  *.h *.hxx *.hpp)
  install(FILES ${octovis_HDRS} DESTINATION include/octovis_ext)

  # Install catkin package.xml
  install(FILES package.xml DESTINATION share/octovis_ext)
  
ELSE()
    MESSAGE ( "Unfortunately, the viewer (octovis) can not be built because some requirements are missing.")
	MESSAGE ( "This will not affect the compilation of the stand-alone library and tools (octomap_ext)")
    MESSAGE ( "See README.txt or http://octomap.sf.net for further information.\n")
ENDIF()

