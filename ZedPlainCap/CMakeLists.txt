cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)
project(ZedCap)
set(CMAKE_BUILD_TYPE Relase)
set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem system serialization iostreams thread REQUIRED)
find_package(ZED REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_library (LibTinyXML2 tinyxml2)

include_directories( ${OpenCV_INCLUDE_DIRS} ${ZED_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ../zebravision ../navXTimeSync  ..)
link_directories(/usr/local/cuda/lib64/stubs)
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -Wall -Ofast")
 
add_definitions(-DZED_SUPPORT)
set (NAVX_SRCS
	../navXTimeSync/AHRS.cpp
	../navXTimeSync/ContinuousAngleTracker.cpp
	../navXTimeSync/InertialDataIntegrator.cpp
	../navXTimeSync/OffsetTracker.cpp
	../navXTimeSync/SerialIO.cpp
	../navXTimeSync/SerialPort.cpp )

add_executable( ZedCap ZedCap.cpp ../zebravision/mediain.cpp ../zebravision/zedcamerain.cpp ../zebravision/syncin.cpp ../zebravision/asyncin.cpp ../zebravision/zmsin.cpp ../zebravision/cameraparams.cpp  ../zebravision/zedparams.cpp ../zebravision/ZvSettings.cpp ../zebravision/portable_binary_iarchive.cpp ../zebravision/portable_binary_oarchive.cpp ${NAVX_SRCS})
target_link_libraries( ZedCap ${OpenCV_LIBS} ${ZED_LIBRARIES} ${CUDA_LIBRARIES} ${Boost_LIBRARIES} ${LibTinyXML2})
