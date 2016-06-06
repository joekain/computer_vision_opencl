cmake_minimum_required(VERSION 2.8)
project(computer_vision)

#Minimal OpenCL CMakeLists.txt by StreamComputing
# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

# OpenCV
find_package( OpenCV REQUIRED )

# Common settings
set(CMAKE_C_FLAGS "-std=c99 -Wall")
set(CMAKE_CXX_FLAGS "-std=c++11")

add_subdirectory(ps1)
