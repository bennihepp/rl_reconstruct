add_executable(
	convert_bbox_to_mesh
	main.cpp
	#${LIBBH_SOURCES}
)

target_link_libraries(
	convert_bbox_to_mesh
	LINK_PUBLIC
	${LIBBH_LIBRARIES}
	${PCL_COMMON_LIBRARIES}
	${PCL_IO_LIBRARIES}
	Boost::program_options
)
