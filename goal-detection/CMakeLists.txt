cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Release)
project( goal_detect )
add_definitions(-std=c++11 -DZED_SUPPORT -Wall -ftrack-macro-expansion=0)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};../zebravision/cmake")
find_package(OpenCV REQUIRED )
find_package(ZED REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS filesystem system serialization iostreams thread REQUIRED)
find_package(ZMQ REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(.. ../navXTimeSync)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_library (LibTinyXML2 tinyxml2)

include_directories( ${OpenCV_INCLUDE_DIRS} ${ZMQ_INCLUDE_DIRS} ${ZED_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ../zebravision ..)

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS} /usr/local/cuda/lib64/stubs)
set (NAVX_SRCS
	../navXTimeSync/AHRS.cpp
	../navXTimeSync/ContinuousAngleTracker.cpp
	../navXTimeSync/InertialDataIntegrator.cpp
	../navXTimeSync/OffsetTracker.cpp
	../navXTimeSync/SerialIO.cpp
	../navXTimeSync/SerialPort.cpp )

add_executable( goal_detect goal_detect.cpp ../zebravision/GoalDetector.cpp ../zebravision/mediain.cpp ../zebravision/syncin.cpp ../zebravision/asyncin.cpp ../zebravision/zedcamerain.cpp ../zebravision/zedsvoin.cpp ../zebravision/zmsin.cpp ../zebravision/cameraparams.cpp ../zebravision/zedparams.cpp ../zebravision/Utilities.cpp ../zebravision/objtype.cpp ../zebravision/track3d.cpp ../zebravision/kalman.cpp ../zebravision/hungarian.cpp ../zebravision/portable_binary_iarchive.cpp ../zebravision/portable_binary_oarchive.cpp ../zebravision/ZvSettings.cpp ${NAVX_SRCS})
target_link_libraries( goal_detect ${OpenCV_LIBS} ${ZED_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY} ${Boost_LIBRARIES} ${ZMQ_LIBRARIES} ${LibTinyXML2} ${PCL_COMMON_LIBRARIES})
