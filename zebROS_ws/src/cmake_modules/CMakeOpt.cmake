# Various compiler optimizations to apply to all nodes


add_definitions(-std=c++11 -Wall -Wextra -Wno-switch -ftrack-macro-expansion=0)

# Without -fno-finite-math-only, NaN checks are optimized away
# This is bad since the Zed outputs NaN for invalid depth data
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto=4 -fno-finite-math-only")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -flto=4 -fno-finite-math-only")
set(CMAKE_MODULE_PATH "/usr/share/cmake-2.8/Modules/")
if (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "arm-linux-gnueabihf")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4")
  add_definitions(-DIS_JETSON=1)
elseif (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "aarch64-linux-gnu")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a57")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -mcpu=cortex-a57")
  add_definitions(-DIS_JETSON=1)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -march=native -mtune=native")
endif()

