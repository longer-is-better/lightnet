file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
add_library(all_in_one SHARED ${srcs})
target_link_libraries(
    all_in_one
    ${glog_LIBRARIES}
    OpenMP::OpenMP_CXX
)
target_include_directories(all_in_one PUBLIC .)
set_target_properties(all_in_one PROPERTIES CUDA_ARCHITECTURES "86")