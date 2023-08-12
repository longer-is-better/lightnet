include(FetchContent)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
cmake_policy(SET CMP0135 NEW)
endif()

find_package(GTest REQUIRED)

# add_subdirectory(test_operators)
# add_subdirectory(test_utils)
add_subdirectory(test_network)
add_subdirectory(test_tensor)