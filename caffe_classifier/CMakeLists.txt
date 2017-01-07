cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
project( CaffePrediction )
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -flto")
if (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "arm-linux-gnueabihf")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4 -fvect-cost-model")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mcpu=cortex-a15 -mfpu=neon-vfpv4 -fvect-cost-model")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -march=native")
endif()

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )
find_package( ZED QUIET )
if (${ZED_FOUND})
  add_definitions(-DZED_SUPPORT)
endif()
find_package( Boost COMPONENTS system filesystem iostreams serialization REQUIRED )

set (CAFFE_BASE_DIR "/home/ubuntu/caffe")

include_directories( ${OpenCV_INCLUDE_DIRS} ${ZED_INCLUDE_DIRS} ${CAFFE_BASE_DIR}/build/include ${CAFFE_BASE_DIR}/include ${Boost_INCLUDE_DIR} ../zebravision)

link_directories( ${ZED_LIBRARY_DIR} )

find_library (LibCaffe caffe PATH  ${CAFFE_BASE_DIR}/build/lib)
find_library (LibGLOG  glog)
find_library (LibProtobuf protobuf)

cuda_add_executable( CaffeBatchPrediction ../zebravision/CaffeBatchPrediction.cpp ../zebravision/fast_nms.cpp ../zebravision/scalefactor.cpp main.cpp ../zebravision/detect.cpp )
cuda_add_executable( DetectCheck ../zebravision/CaffeBatchPrediction.cpp detectcheck.cpp ../zebravision/fast_nms.cpp ../zebravision/scalefactor.cpp ../zebravision/detect.cpp ../zebravision/mediain.cpp ../zebravision/zedin.cpp ../zebravision/portable_binary_iarchive.cpp ../zebravision/portable_binary_oarchive.cpp)
target_link_libraries( CaffeBatchPrediction ${OpenCV_LIBS} ${LibCaffe} ${Boost_LIBRARIES} ${LibGLOG} ${LibProtobuf} )
target_link_libraries( DetectCheck ${ZED_LIBRARIES} ${OpenCV_LIBS} ${LibCaffe} ${Boost_LIBRARIES} ${LibGLOG} ${LibProtobuf} ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY} )
