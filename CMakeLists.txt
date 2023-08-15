cmake_minimum_required(VERSION 3.16)
project(cuda-practice_v1 LANGUAGES CXX CUDA)

add_definitions(-DDEBUG=1)
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)
include_directories(${glog_INCLUDE_DIRS})

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIR})

# include_directories(${PROJECT_SOURCE_DIR}/Common)

enable_testing()
add_subdirectory(src)
add_subdirectory(tests)
# add_subdirectory(3rd_party)