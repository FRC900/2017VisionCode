CMake_minimum_required(VERSION 2.8)
project(zv)

set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
set(CMAKE_BUILD_TYPE Debug)
set(test ON)
add_definitions(-std=c++11 -Wall -Wextra -Wno-switch -ftrack-macro-expansion=0)

# Without -fno-finite-math-only, NaN checks are optimized away
# This is bad since the Zed outputs NaN for invalid depth data
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto=4 -fno-finite-math-only")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -flto=4 -fno-finite-math-only")
set(CMAKE_MODULE_PATH "/usr/share/cmake-2.8/Modules/")
if (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "arm-linux-gnueabihf")
  unset(CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
  option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4")
elseif (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "aarch64-linux-gnu")
  # Jetson TX1/TX2 are 64-bit ARM targets
  unset(CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -march=native -mtune=native")
  #link_directories($ENV{HOME}/GIE/lib)
  #find_library (LibNVCaffeParser nvcaffe_parser $ENV{HOME}/GIE/lib)
  #find_library (LibNVInfer nvinfer $ENV{HOME}/GIE/lib)
  #add_definitions(-DUSE_GIE=1)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -march=native -mtune=native")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
	include(CUDNN)
	detect_cuDNN()
endif()

find_package(ZED)
#Check if we are building with a GPU by checking CUDNN
#If we aren't, don't add ZED support
if (ZED_FOUND AND CUDNN_FOUND)
  add_definitions(-DZED_SUPPORT)
endif()

find_package(Boost COMPONENTS filesystem system serialization iostreams thread REQUIRED)
find_package(OpenCV REQUIRED)
find_package(MKL QUIET)
if (MKL_FOUND)
	add_definitions(-DUSE_MKL=1)
endif()
find_package(PCL COMPONENTS common)
add_definitions(${PCL_DEFINITIONS})

include (GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_describe(GIT_DESC "--long")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" @ONLY)

set (CAFFE_BASE_DIR "$ENV{HOME}/caffe")

# Caffe headers change the size of various
# objects based on these defines. Try to match
# how caffe itself was compiled
# Not sure how best to detect CPU-only builds
# but this should handle a lot of common cases
if (CUDNN_FOUND)
	add_definitions(-DUSE_CUDNN=1)
	MESSAGE("-- Using GPU-accelerated Caffe build")
else()
	add_definitions(-DCPU_ONLY=1)
	MESSAGE("-- Using CPU-only Caffe build")
	link_directories(/usr/local/cuda/lib64/stubs)
endif()

include_directories(
	${OpenCV_INCLUDE_DIRS}
	${ZED_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${CAFFE_BASE_DIR}/build/install/include
	${CAFFE_BASE_DIR}/build/include
	${CAFFE_BASE_DIR}/include
	${Boost_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
	.
	../fovis
	/home/$ENV{HOME}/2017VisionCode/libfovis/build/include/fovis
	#../../GIE/include
	..
	../navXTimeSync
	../googletest/googletest)

if ($MKL_FOUND)
	include_directories (${MKL_INCLUDE_DIR})
endif()

#link_directories($ENV{HOME}/GIE/lib)
link_directories(/opt/intel/ipp90legacy/linux/lib/intel64/threaded)
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(/home/$ENV{HOME}/2017VisionCode/libfovis/build/lib/)
link_directories(/usr/local/cuda/lib64/stubs)
link_directories(${PCL_LIBRARY_DIRS})

# Without this, nvcc grabs --std=c++11 which looks like
# it breaks builds for some reason
set (CUDA_PROPAGATE_HOST_FLAGS off)
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)

# Target appropriate GPU family when generating code
if (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "arm-linux-gnueabihf" OR
    ${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "aarch64-linux-gnu")
  LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=sm_53)
  LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
else()
  LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=compute_35)
  LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
  LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=compute_52)

  if (CUDA_VERSION_MAJOR GREATER 7)
    MESSAGE("-- CUDA8 support detected, including Pascal target")
    LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
	LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=compute_61)
  endif()
endif()
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler ,\"-O3\",\"-DNDEBUG\","-ffast-math")
LIST(APPEND CUDA_NVCC_FLAGS -O3 --default-stream per-thread)

set (NAVX_SRCS
	../navXTimeSync/AHRS.cpp
	../navXTimeSync/ContinuousAngleTracker.cpp
	../navXTimeSync/InertialDataIntegrator.cpp
	../navXTimeSync/OffsetTracker.cpp
	../navXTimeSync/SerialIO.cpp
	../navXTimeSync/SerialPort.cpp )

set (ZV_SRCS
	zca.cpp
	zca.cu
	cuda_utils.cpp
	Classifier.cpp
	CaffeClassifier.cpp
	GIEClassifier.cpp
	classifierio.cpp
	cascadeclassifierio.cpp
	detectstate.cpp
	objdetect.cpp
	scalefactor.cpp
	fast_nms.cpp
	depth_threshold.cu
	detect.cpp
	GoalDetector.cpp
	objtype.cpp
	track3d.cpp
	mediain.cpp
	asyncin.cpp
	syncin.cpp
	videoin.cpp
	camerain.cpp
	C920Camera.cpp
	c920camerain.cpp
	zedcamerain.cpp
	zedsvoin.cpp
	zmsin.cpp
	imagein.cpp
	cameraparams.cpp
	zedparams.cpp
	mediaout.cpp
	aviout.cpp
	pngout.cpp
	zmsout.cpp
	portable_binary_iarchive.cpp
	portable_binary_oarchive.cpp
	Args.cpp
	WriteOnFrame.cpp
	groundtruth.cpp
	Utilities.cpp
	#../fovis/FovisLocalizer.cpp
	FlowLocalizer.cpp
	kalman.cpp
	hungarian.cpp
	ZvSettings.cpp
	colormap.cpp
	${NAVX_SRCS} 
	${CMAKE_CURRENT_BINARY_DIR}/version.cpp)
CUDA_ADD_EXECUTABLE(zv ${ZV_SRCS} zv.cpp)
find_library (LibCaffe caffe PATH ${CAFFE_BASE_DIR}/build/lib)
find_library (LibGLOG  glog)
#find_library (LibProtobuf protobuf $ENV{HOME}/GIE/lib)
#find_library (LibWconv wconv $ENV{HOME}/GIE/lib)
find_library (LibTinyXML2 tinyxml2)
find_library (LibEFence efence)

set (ZV_LIBS
	${LibProtobuf}
	${OpenCV_LIBS}
	${LibCaffe}
	#fovis
	${CMAKE_THREAD_LIBS_INIT}
	${ZED_LIBRARIES}
	${Boost_LIBRARIES}
	${LibGLOG}
	${LibNVCaffeParser}
	${LibNVInfer}
	${LibTinyXML2}
	${MKL_LIBRARIES}
	#${LibEFence} 
	)
target_link_libraries( zv ${ZV_LIBS})
CUDA_ADD_CUBLAS_TO_TARGET(zv)

# Only build this if ZED libs are available - it converts
# from ZED's SVO format to our home-brewed ZMS video file
# format.
if (ZED_FOUND)
	add_executable(convertzms convertzms.cpp mediain.cpp syncin.cpp cameraparams.cpp zedparams.cpp zedsvoin.cpp zmsin.cpp mediaout.cpp zmsout.cpp portable_binary_oarchive.cpp portable_binary_iarchive.cpp ZvSettings.cpp ${NAVX_SRCS})
  target_link_libraries( convertzms ${Boost_LIBRARIES} ${OpenCV_LIBS} ${ZED_LIBRARIES} ${LibTinyXML2})
endif()
add_executable(mergezms mergezms.cpp mediain.cpp syncin.cpp cameraparams.cpp zedparams.cpp zedsvoin.cpp zmsin.cpp mediaout.cpp zmsout.cpp portable_binary_oarchive.cpp portable_binary_iarchive.cpp ZvSettings.cpp ${NAVX_SRCS})
target_link_libraries( mergezms ${Boost_LIBRARIES} ${OpenCV_LIBS} ${ZED_LIBRARIES} ${LibTinyXML2})
CUDA_ADD_EXECUTABLE(predict_one predict_one.cpp CaffeClassifier.cpp GIEClassifier.cpp Classifier.cpp zca.cpp zca.cu classifierio.cpp cuda_utils.cpp portable_binary_iarchive.cpp portable_binary_oarchive.cpp)
target_link_libraries( predict_one ${Boost_LIBRARIES} ${OpenCV_LIBS} ${LibCaffe} ${LibGLOG} ${LibProtobuf} ${MKL_LIBRARIES} ${LibNVCaffeParser} ${LibNVInfer})
CUDA_ADD_CUBLAS_TO_TARGET(predict_one)
CUDA_ADD_EXECUTABLE(rank_imagelist rank_imagelist.cpp CaffeClassifier.cpp GIEClassifier.cpp Classifier.cpp zca.cpp zca.cu classifierio.cpp cuda_utils.cpp portable_binary_iarchive.cpp portable_binary_oarchive.cpp)
target_link_libraries( rank_imagelist ${Boost_LIBRARIES} ${OpenCV_LIBS} ${LibCaffe} ${LibGLOG} ${LibProtobuf} ${MKL_LIBRARIES} ${LibNVCaffeParser} ${LibNVInfer})
CUDA_ADD_CUBLAS_TO_TARGET(rank_imagelist)
#add_executable(depthtest depthtest.cpp)
#target_link_libraries( depthtest ${OpenCV_LIBS} )

if(test)
  find_package(GTest REQUIRED)
  enable_testing()
  CUDA_ADD_EXECUTABLE(test_zv ${ZV_SRCS} unit_tests.cpp)
  list(REMOVE_ITEM ZV_LIBS opencv_ts)
  target_link_libraries(test_zv ${ZV_LIBS} ${GTEST_MAIN_LIBRARIES} ${GTEST_LIBRARIES})
  CUDA_ADD_CUBLAS_TO_TARGET(test_zv)
  add_test(NAME unit_tests COMMAND test_zv)
endif()
