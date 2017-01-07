cmake_minimum_required(VERSION 2.6.0)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME libfovis)
include(cmake/pods.cmake)

# Try to use pkg-config to find Eigen3.  If we don't have pkg-config, then
# just assume that Eigen3 is in the include path.
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(EIGEN REQUIRED eigen3)
endif()

set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
#cuda binaries are in different locations for different architectures
if(${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "arm-linux-gnueabihf")
	link_directories(/usr/local/cuda/lib)
else()
	link_directories(/usr/local/cuda/lib64)
endif()

# Do we want to use SSE?
option(USE_SSE "Enables SSE support" OFF)
if(USE_SSE)
    set(USE_SSE_MESSAGE "Enabled")
    add_definitions(-DFOVIS_USE_SSE)
else()
    set(USE_SSE_MESSAGE "Disabled")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -flto")
if (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "arm-linux-gnueabihf")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4 -fvect-cost-model")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4 -fvect-cost-model")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -march=native")
endif()

# build the main library
add_subdirectory(libfovis)

add_subdirectory(testers)
add_subdirectory(examples)

message("

Configuration (fovis):

           libfovis:    Enabled
        SSE Support:    ${USE_SSE_MESSAGE}
     OpenNI example:    ${OPENNI_EXAMPLE_MESSAGE}
 libfreenct example:    ${LIBFREENECT_EXAMPLE_MESSAGE}
   TUM-RGBD example:    ${TUM_RGBD_MESSAGE}

")


if(NOT PKG_CONFIG_FOUND)
    message("Couldn't find pkg-config.  Hoping that eigen3 is in the include path...")
    message("If you get compile errors related to Eigen, try setting CMAKE_CXX_FLAGS")
    message("")
endif()
