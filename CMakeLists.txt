cmake_minimum_required (VERSION 3.5)
project (sannifa)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include_directories(include) # headers
add_subdirectory (src) 

#add_subdirectory (test)
#enable_testing ()
#add_test (NAME unittests COMMAND run_tests)
