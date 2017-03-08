cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
project( mark_goals )
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto --std=c++11")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -flto --std=c++11")

find_package(OpenCV REQUIRED)
find_package(PCL COMPONENTS common REQUIRED)
find_package(Boost COMPONENTS filesystem thread system serialization iostreams REQUIRED)

add_definitions(${PCL_DEFINITIONS})

include_directories(../zebravision .. ../navXTimeSync ${PCL_INCLUDE_DIRS})
set (NAVX_SRCS
	../navXTimeSync/AHRS.cpp
	../navXTimeSync/ContinuousAngleTracker.cpp
	../navXTimeSync/InertialDataIntegrator.cpp
	../navXTimeSync/OffsetTracker.cpp
	../navXTimeSync/SerialIO.cpp
	../navXTimeSync/SerialPort.cpp )

add_executable(mark_goals mark_goals.cpp ../zebravision/mediain.cpp ../zebravision/syncin.cpp ../zebravision/cameraparams.cpp ../zebravision/zmsin.cpp ../zebravision/portable_binary_iarchive.cpp ${NAVX_SRCS})
target_link_libraries( mark_goals ${OpenCV_LIBS} ${Boost_LIBRARIES} )
