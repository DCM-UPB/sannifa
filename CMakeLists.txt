cmake_minimum_required (VERSION 3.5)
project (sannifa LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include(FindPackageHandleStandardArgs)

find_package(Torch REQUIRED)

find_path(FFNN_INCLUDE_DIR FeedForwardNeuralNetwork.hpp HINTS "$ENV{FFNN_ROOT}/include")
find_library(FFNN_LIBRARY_DIR ffnn HINTS "$ENV{FFNN_ROOT}/lib/.libs")
find_package_handle_standard_args(LIBFFNN DEFAULT_MSG FFNN_LIBRARY_DIR FFNN_INCLUDE_DIR)


include_directories(include "${FFNN_INCLUDE_DIR}") # headers

add_subdirectory (src) 

add_subdirectory (test)
#enable_testing ()
#add_test (NAME unittests COMMAND run_tests)
