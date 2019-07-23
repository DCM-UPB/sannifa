add_executable(test_qpoly test_qpoly.cpp)
target_link_libraries(test_qpoly sannifa)

add_executable(test_qtempl test_qtempl.cpp)
target_link_libraries(test_qtempl sannifa)

add_executable(test_torch test_torch.cpp)
target_link_libraries(test_torch sannifa)