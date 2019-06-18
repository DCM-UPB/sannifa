add_executable(bench_qpoly bench_qpoly.cpp)
target_link_libraries(bench_qpoly sannifa)

add_executable(bench_qtempl bench_qtempl.cpp)
target_link_libraries(bench_qtempl sannifa)

add_executable(bench_torch bench_torch.cpp)
target_link_libraries(bench_torch sannifa)
