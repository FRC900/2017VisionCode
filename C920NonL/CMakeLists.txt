cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11")
project( C920NonL )
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED )
find_package(Boost COMPONENTS filesystem system serialization iostreams thread REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} ../zebravision ../navXTimeSync ..)
find_library (LibTinyXML2 tinyxml2)

set (NAVX_SRCS
	../navXTimeSync/AHRS.cpp
	../navXTimeSync/ContinuousAngleTracker.cpp
	../navXTimeSync/InertialDataIntegrator.cpp
	../navXTimeSync/OffsetTracker.cpp
	../navXTimeSync/SerialIO.cpp
	../navXTimeSync/SerialPort.cpp )


add_executable( C920Cap C920Cap.cpp ../zebravision/mediain.cpp ../zebravision/asyncin.cpp ../zebravision/cameraparams.cpp ../zebravision/ZvSettings.cpp ${NAVX_SRCS})
target_link_libraries( C920Cap ${OpenCV_LIBS} ${Boost_LIBRARIES} ${LibTinyXML2})
