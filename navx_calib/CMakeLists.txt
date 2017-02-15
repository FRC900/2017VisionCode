cmake_minimum_required (VERSION 2.8)
project (nvxcalib)
add_definitions(-std=c++11 -Wall -Wextra -Wno-switch)
set(CMAKE_BUILD_TYPE Debug)
FIND_PACKAGE (Threads REQUIRED)
FIND_PACKAGE (OpenCV REQUIRED)


set (NAVX_SRCS
	../navXTimeSync/AHRS.cpp
	../navXTimeSync/ContinuousAngleTracker.cpp
	../navXTimeSync/InertialDataIntegrator.cpp
	../navXTimeSync/OffsetTracker.cpp
	../navXTimeSync/SerialIO.cpp
	../navXTimeSync/SerialPort.cpp )

file(GLOB SOURCES
	"*.h"
	"*.cpp"
)

include_directories(
	"."
	"../navXTimeSync"
	${OpenCV_INCLUDE_DIRS}
)

add_executable(nvxcalib ${SOURCES} ${NAVX_SRCS})

target_link_libraries(nvxcalib ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})
#target_compile_features(NavXTimeStamp PRIVATE cxx_range_for)

