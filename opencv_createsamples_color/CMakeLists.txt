cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
set(CMAKE_BUILD_TYPE Release )
add_definitions(-Wall -Wextra)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -march=native -flto")
project( opencv_createsamples_color )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( opencv_createsamples_color cvboost.cpp cvcommon.cpp cvhaarclassifier.cpp cvhaartraining.cpp cvsamples.cpp cvsamplesoutput.cpp createsamples.cpp )
target_link_libraries( opencv_createsamples_color ${OpenCV_LIBS} )

