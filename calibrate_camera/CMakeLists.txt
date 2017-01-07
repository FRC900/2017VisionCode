# cmake needs this line
cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required

# Define project name
project(opencv_example)

find_package(ZED 0.9.2 QUIET)
if (${ZED_FOUND})
  add_definitions(-DZED_SUPPORT)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# Declare the executable target built from your sources
add_executable(opencv_example example.cpp)

# Link your application with OpenCV libraries
target_link_libraries(opencv_example ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY})
