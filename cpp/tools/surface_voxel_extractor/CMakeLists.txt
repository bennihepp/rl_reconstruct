add_executable(
	surface_voxel_extractor
	main.cpp
	#${LIBBH_SOURCES}
)

target_link_libraries(
	surface_voxel_extractor
	LINK_PUBLIC
	${LIBBH_LIBRARIES}
	${PCL_COMMON_LIBRARIES}
	${PCL_IO_LIBRARIES}
	${OCTOMAP_LIBRARIES}
	Boost::program_options
)
