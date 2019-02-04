add_executable(torchtest torchtest.cpp PropagateBenchmark.hpp)
target_link_libraries(torchtest sannifa)

add_executable(ffnntest ffnntest.cpp PropagateBenchmark.hpp)
target_link_libraries(ffnntest sannifa)
