file(GLOB SOURCES "*.cpp")
add_library(sannifa STATIC ${SOURCES})
#target_link_libraries(sannifa "${LIBQNETS}" "${TORCH_LIBRARIES}")
