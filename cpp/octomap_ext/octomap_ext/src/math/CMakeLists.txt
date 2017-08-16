SET (octomath_SRCS
  Vector3.cpp
  Quaternion.cpp
  Pose6D.cpp
 )


ADD_LIBRARY( octomath_ext SHARED ${octomath_SRCS})

SET_TARGET_PROPERTIES( octomath_ext PROPERTIES
  VERSION ${OCTOMAP_EXT_VERSION}
  SOVERSION ${OCTOMAP_EXT_SOVERSION}
  INSTALL_NAME_DIR ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}   # this seems to be necessary for MacOS X
)
# INSTALL_NAME_DIR seems to be necessary for MacOS X

ADD_LIBRARY( octomath_ext-static STATIC ${octomath_SRCS})
SET_TARGET_PROPERTIES(octomath_ext-static PROPERTIES OUTPUT_NAME "octomath_ext")

if(NOT EXISTS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/octomap_ext")
  file(MAKE_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/octomap_ext")
endif()

export(TARGETS octomath_ext octomath_ext-static
  APPEND FILE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/octomap_ext/octomap_ext-targets.cmake")

install(TARGETS octomath_ext octomath_ext-static
  EXPORT octomap_ext-targets
  INCLUDES DESTINATION include
  ${INSTALL_TARGETS_DEFAULT_ARGS}
)
