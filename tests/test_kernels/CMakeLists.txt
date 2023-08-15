find_package(CUDA REQUIRED)
file(GLOB test_kernel_files ${CMAKE_CURRENT_SOURCE_DIR}/test_*.c*)
add_executable(test_kernels main.cpp ${test_kernel_files})
target_link_libraries(
  test_kernels
  all_in_one
  gflags
  GTest::gtest
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)

set_target_properties(test_kernels PROPERTIES CUDA_ARCHITECTURES "86")
include(GoogleTest)

gtest_discover_tests(test_kernels PROPERTIES DISCOVERY_TIMEOUT 60)