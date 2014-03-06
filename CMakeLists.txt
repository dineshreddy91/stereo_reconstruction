cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project( stereo_using_given_odo )

find_package( PCL 1.2 REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV REQUIRED )
#add_executable( 3d_points_2_clouds_using_dat_file 3d_points_2_clouds_using_dat_file.cpp)
#target_link_libraries( 3d_points_2_clouds_using_dat_file ${OpenCV_LIBS} ${PCL_LIBRARIES} )

add_executable( key_feat_bow key_feat_bow.cpp)
target_link_libraries( key_feat_bow ${OpenCV_LIBS} ${PCL_LIBRARIES} )


