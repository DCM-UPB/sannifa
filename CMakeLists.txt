cmake_minimum_required (VERSION 3.5)
include(FindPackageHandleStandardArgs)

project (sannifa LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

# find packages
message(STATUS "Configured FFNN_ROOT: ${FFNN_ROOT}")
message(STATUS "Configured TORCH_ROOT: ${TORCH_ROOT}")

find_package(Torch REQUIRED HINTS "${TORCH_ROOT}")
find_path(FFNN_INCLUDE_DIR FeedForwardNeuralNetwork.hpp HINTS "${FFNN_ROOT}/include")
find_library(FFNN_LIBRARY_DIR ffnn HINTS "${FFNN_ROOT}/lib/.libs")
find_package_handle_standard_args(libffnn DEFAULT_MSG FFNN_LIBRARY_DIR FFNN_INCLUDE_DIR)

# set header / library paths
include_directories(include/ "${FFNN_INCLUDE_DIR}" "${TORCH_INCLUDE_DIRS}") # headers
link_libraries("${FFNN_LIBRARY_DIR}" "${TORCH_LIBRARIES}") # libs

# continue with subdirectories
add_subdirectory (src) 
add_subdirectory (test)

#enable_testing ()
#add_test (NAME unittests COMMAND run_tests)
