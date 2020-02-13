cmake_minimum_required(VERSION 2.8)
project( orb )

# OpenCV 
find_package( OpenCV REQUIRED )

include_directories( 
${PROJECT_SOURCE_DIR}/include
${OpenCV_INCLUDE_DIRS} 
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/ORB_modify.cc
src/ORBextractor.cc
src/gms_matcher.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable( orb_modify ORB.cc )
target_link_libraries( orb_modify ${PROJECT_NAME} )

add_executable( orb_matcher ORB_matcher.cc )
target_link_libraries( orb_matcher ${PROJECT_NAME} )