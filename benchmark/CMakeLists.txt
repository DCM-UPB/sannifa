add_executable(bench_ffnn bench_ffnn.cpp)
target_link_libraries(bench_ffnn sannifa)

add_executable(bench_torch bench_torch.cpp)
target_link_libraries(bench_torch sannifa)
