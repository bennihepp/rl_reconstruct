SET (octomap_SRCS
  AbstractOcTree.cpp
  AbstractOccupancyOcTree.cpp
  Pointcloud.cpp
  ScanGraph.cpp
  CountingOcTree.cpp
  OcTree.cpp
  OcTreeExt.cpp
  OcTreeNode.cpp
  OcTreeExtNode.cpp
  OcTreeStamped.cpp
  ColorOcTree.cpp
  )

# dynamic and static libs, see CMake FAQ:
ADD_LIBRARY( octomap_ext SHARED ${octomap_SRCS})
set_target_properties( octomap_ext PROPERTIES
  VERSION ${OCTOMAP_EXT_VERSION}
  SOVERSION ${OCTOMAP_EXT_SOVERSION}
)
ADD_LIBRARY( octomap_ext-static STATIC ${octomap_SRCS})
SET_TARGET_PROPERTIES(octomap_ext-static PROPERTIES OUTPUT_NAME "octomap_ext") 

TARGET_LINK_LIBRARIES(octomap_ext octomath_ext)

if(NOT EXISTS "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/octomap_ext")
  file(MAKE_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/octomap_ext")
endif()

export(TARGETS octomap_ext octomap_ext-static
  APPEND FILE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cmake/octomap_ext/octomap_ext-targets.cmake")

ADD_SUBDIRECTORY( testing )

ADD_EXECUTABLE(graph2tree graph2tree.cpp)
TARGET_LINK_LIBRARIES(graph2tree octomap_ext)

ADD_EXECUTABLE(log2graph log2graph.cpp)
TARGET_LINK_LIBRARIES(log2graph octomap_ext)

ADD_EXECUTABLE(binvox2bt binvox2bt.cpp ../include/octomap_ext/OcTreeExtBase.h ../include/octomap_ext/TemplateOccupancyOcTree.h)
TARGET_LINK_LIBRARIES(binvox2bt octomap_ext)

ADD_EXECUTABLE(bt2vrml bt2vrml.cpp)
TARGET_LINK_LIBRARIES(bt2vrml octomap_ext)

ADD_EXECUTABLE(edit_octree edit_octree.cpp)
TARGET_LINK_LIBRARIES(edit_octree octomap_ext)

ADD_EXECUTABLE(convert_octree convert_octree.cpp)
TARGET_LINK_LIBRARIES(convert_octree octomap_ext)

ADD_EXECUTABLE(eval_octree_accuracy eval_octree_accuracy.cpp)
TARGET_LINK_LIBRARIES(eval_octree_accuracy octomap_ext)

ADD_EXECUTABLE(compare_octrees compare_octrees.cpp)
TARGET_LINK_LIBRARIES(compare_octrees octomap_ext)

ADD_EXECUTABLE(simple_example simple_example.cpp)
TARGET_LINK_LIBRARIES(simple_example octomap_ext)

ADD_EXECUTABLE(normals_example normals_example.cpp)
TARGET_LINK_LIBRARIES(normals_example octomap_ext)

ADD_EXECUTABLE(intersection_example intersection_example.cpp)
TARGET_LINK_LIBRARIES(intersection_example octomap_ext)

ADD_EXECUTABLE(octree2pointcloud octree2pointcloud.cpp)
TARGET_LINK_LIBRARIES(octree2pointcloud octomap_ext)

install(TARGETS octomap_ext octomap_ext-static
  EXPORT octomap_ext-targets
  INCLUDES DESTINATION include
  ${INSTALL_TARGETS_DEFAULT_ARGS}
)
install(EXPORT octomap_ext-targets DESTINATION share/octomap_ext/)

install(TARGETS
	graph2tree
	log2graph
	binvox2bt
	bt2vrml
	edit_octree
	convert_octree
	eval_octree_accuracy
	compare_octrees
	${INSTALL_TARGETS_DEFAULT_ARGS}
)

