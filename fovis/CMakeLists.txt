cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=gnu++11 ")
project( fovis_zed_test )

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Ofast -flto")
if (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "arm-linux-gnueabihf")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4 -fvect-cost-model")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mcpu=cortex-a15 -mfpu=neon-vfpv4 -fvect-cost-model")
elseif (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "aarch64-linux-gnu")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a57")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -mcpu=cortex-a57")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -march=native")
endif()

find_package(OpenCV REQUIRED )
find_package(ZED REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${ZED_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${Boost_INCLUDE_DIR}
	/home/ubuntu/2016VisionCode/libfovis/build/include/fovis
	p1d
	../zebravision)
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(/home/ubuntu/2016VisionCode/libfovis/build/lib/)

add_executable( fovis_zed_test 
	fovis_testing.cpp 
	FovisLocalizer.cpp 
	../zebravision/mediain.cpp
	../zebravision/asyncin.cpp
	../zebravision/syncin.cpp
	../zebravision/zedcamerain.cpp
	../zebravision/cameraparams.cpp
	../zebravision/zedparams.cpp
	../zebravision/zedsvoin.cpp
	)
target_link_libraries( fovis_zed_test 
	fovis 
	${OpenCV_LIBS} 
	${ZED_LIBRARIES} 
	${CUDA_LIBRARIES} 
	${Boost_LIBRARIES})
