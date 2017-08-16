CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
PROJECT( octomap-ext-distribution )

ENABLE_TESTING()  # enable CTest environment of subprojects

option(BUILD_OCTOVIS_SUBPROJECT "Build targets from subproject octovis_ext" ON)
option(BUILD_DYNAMICETD3D_SUBPROJECT  "Build targets from subproject dynamicEDT3D" ON)
option(OCTOVIS_QT5 "Link Octovis against Qt5?" NO)

if(OCTOVIS_QT5)
	# Compiling against QT5 requires C++11.
	set(CMAKE_CXX_STANDARD 11)
endif(OCTOVIS_QT5)

ADD_SUBDIRECTORY( octomap_ext )

if(BUILD_OCTOVIS_SUBPROJECT)
	ADD_SUBDIRECTORY( octovis_ext )
endif()		

if(BUILD_DYNAMICETD3D_SUBPROJECT)	
	ADD_SUBDIRECTORY( dynamicEDT3D )
endif()		


