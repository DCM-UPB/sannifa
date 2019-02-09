add_executable(test_torch test_torch.cpp)
target_link_libraries(test_torch sannifa)

add_executable(test_ffnn test_ffnn.cpp)
target_link_libraries(test_ffnn sannifa)

add_executable(testleak testMemleak.cpp)
