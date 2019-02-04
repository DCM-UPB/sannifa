file(GLOB SOURCES "*.cpp")
add_library(sannifa SHARED ${SOURCES})
#target_link_libraries(sannifa "${LIBFFNN}" "${TORCH_LIBRARIES}")
