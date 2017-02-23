cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
set(CMAKE_BUILD_TYPE Release)
add_definitions(-std=c++11 -Wall -Wextra -Wno-switch)
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Ofast -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} -Ofast -flto")
project( framegrabber )
include_directories(../zebravision)
find_package( OpenCV REQUIRED )

add_executable( FuelPipeline fuelPipeline.cpp )
target_link_libraries( FuelPipeline ${OpenCV_LIBS} )
