find_package(CUDA REQUIRED)
file(GLOB test_network_files ${CMAKE_CURRENT_SOURCE_DIR}/test_*.c*)
add_executable(test_network main.cpp ${test_network_files})


target_link_libraries(
  test_network
  all_in_one
  gflags
  GTest::gtest
  ${CUDA_LIBRARIES}
)

set_target_properties(test_network PROPERTIES CUDA_ARCHITECTURES "86")
include(GoogleTest)

gtest_discover_tests(test_network PROPERTIES DISCOVERY_TIMEOUT 60)