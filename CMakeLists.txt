cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# set(MY_SYSTEM_INCLUDE_DIRS
#     /usr/include/c++/9
#     /usr/include/c++/10
# )
project(cuda-practice_v1 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

if(CMAKE_BUILD_TYPE MATCHES "debug|Debug|DEBUG")
    add_definitions(-DDEBUG=1)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
endif()


find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)
include_directories(${glog_INCLUDE_DIRS})

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIR})

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    add_compile_options(-fopenmp)
    add_compile_options(-Wunknown-pragmas )
endif()

enable_testing()
add_subdirectory(src)
add_subdirectory(tests)